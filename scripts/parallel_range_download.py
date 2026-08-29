"""Resumable multi-connection HTTP range downloader with final SHA-256 verification."""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class Part:
    index: int
    start: int
    end: int
    path: Path

    @property
    def size(self) -> int:
        return self.end - self.start + 1


_PRINT_LOCK = threading.Lock()


def _log(message: str) -> None:
    with _PRINT_LOCK:
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}", flush=True)


def _download_part(url: str, part: Part, timeout: int) -> Path:
    part.path.parent.mkdir(parents=True, exist_ok=True)
    retry = 0
    while True:
        existing = part.path.stat().st_size if part.path.is_file() else 0
        if existing == part.size:
            return part.path
        if existing > part.size:
            raise RuntimeError(f"Part {part.index} is oversized: {existing} > {part.size}")

        range_start = part.start + existing
        headers = {
            "Range": f"bytes={range_start}-{part.end}",
            "Accept-Encoding": "identity",
            "User-Agent": "ava-vla-parallel-range-downloader/1.0",
        }
        try:
            with requests.get(
                url,
                headers=headers,
                stream=True,
                allow_redirects=True,
                timeout=(30, timeout),
            ) as response:
                if response.status_code != 206:
                    raise RuntimeError(
                        f"Part {part.index} expected HTTP 206, got {response.status_code}"
                    )
                content_range = response.headers.get("Content-Range", "")
                expected_prefix = f"bytes {range_start}-"
                if not content_range.startswith(expected_prefix):
                    raise RuntimeError(
                        f"Part {part.index} bad Content-Range {content_range!r}; "
                        f"expected prefix {expected_prefix!r}"
                    )

                remaining = part.size - existing
                wrote = 0
                with part.path.open("ab") as stream:
                    for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                        if not chunk:
                            continue
                        if len(chunk) > remaining - wrote:
                            chunk = chunk[: remaining - wrote]
                        stream.write(chunk)
                        wrote += len(chunk)
                        if wrote == remaining:
                            break
                    stream.flush()
                    os.fsync(stream.fileno())
                if wrote <= 0:
                    raise RuntimeError(f"Part {part.index} received no data")
                retry = 0
        except (OSError, requests.RequestException, RuntimeError) as error:
            retry += 1
            delay = min(30, max(2, retry * 2))
            _log(f"part={part.index} retry={retry} delay={delay}s error={error}")
            time.sleep(delay)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(32 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def download(
    url: str,
    partial_path: Path,
    output_path: Path,
    expected_size: int,
    expected_sha256: str,
    workers: int,
    timeout: int,
) -> None:
    partial_path = partial_path.resolve()
    output_path = output_path.resolve()
    if output_path.is_file():
        if output_path.stat().st_size != expected_size:
            raise RuntimeError(f"Existing output has wrong size: {output_path}")
        digest = _sha256(output_path)
        if digest != expected_sha256:
            raise RuntimeError(f"Existing output SHA-256 mismatch: {digest}")
        _log(f"already_complete path={output_path}")
        return
    if not partial_path.is_file():
        raise FileNotFoundError(f"Missing contiguous prefix file: {partial_path}")

    prefix_size = partial_path.stat().st_size
    if not 0 < prefix_size < expected_size:
        raise RuntimeError(
            f"Partial prefix must be between 1 and {expected_size - 1} bytes; got {prefix_size}"
        )
    workers = max(1, min(workers, expected_size - prefix_size))
    part_root = partial_path.with_name(f"{partial_path.name}.parallel_parts")
    part_root.mkdir(parents=True, exist_ok=True)
    lock_path = partial_path.with_name(f"{partial_path.name}.parallel_range.lock")
    lock_stream = lock_path.open("a+")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError(f"Another parallel downloader holds {lock_path}") from error

    remaining = expected_size - prefix_size
    part_span = (remaining + workers - 1) // workers
    parts = []
    for index in range(workers):
        start = prefix_size + index * part_span
        if start >= expected_size:
            break
        end = min(expected_size - 1, start + part_span - 1)
        parts.append(Part(index, start, end, part_root / f"part-{index:03d}-{start}-{end}"))

    _log(
        f"parallel_start prefix={prefix_size} remaining={remaining} "
        f"parts={len(parts)} workers={workers}"
    )
    started = time.monotonic()
    last_time = started
    last_bytes = prefix_size + sum(path.path.stat().st_size for path in parts if path.path.is_file())
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {
            executor.submit(_download_part, url, part, timeout): part
            for part in parts
        }
        while pending:
            done, not_done = concurrent.futures.wait(
                pending,
                timeout=10,
                return_when=concurrent.futures.FIRST_EXCEPTION,
            )
            now = time.monotonic()
            current_bytes = prefix_size + sum(
                part.path.stat().st_size for part in parts if part.path.is_file()
            )
            interval_speed = (current_bytes - last_bytes) / max(now - last_time, 1e-6)
            overall_speed = (current_bytes - prefix_size) / max(now - started, 1e-6)
            _log(
                f"progress={100.0 * current_bytes / expected_size:.2f}% "
                f"bytes={current_bytes}/{expected_size} interval={interval_speed / 1e6:.2f}MB/s "
                f"average={overall_speed / 1e6:.2f}MB/s"
            )
            last_time, last_bytes = now, current_bytes
            for future in done:
                future.result()
                pending.pop(future)
            pending = {future: part for future, part in pending.items() if future in not_done}

    if partial_path.stat().st_size != prefix_size:
        raise RuntimeError("Contiguous prefix changed while range parts were downloading")
    _log("all_parts_complete assembling=true")
    with partial_path.open("ab") as destination:
        for part in parts:
            if part.path.stat().st_size != part.size:
                raise RuntimeError(f"Part {part.index} is incomplete during assembly")
            with part.path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=32 * 1024 * 1024)
        destination.flush()
        os.fsync(destination.fileno())
    if partial_path.stat().st_size != expected_size:
        raise RuntimeError(f"Assembled file has wrong size: {partial_path.stat().st_size}")

    _log("sha256_verification_start")
    digest = _sha256(partial_path)
    if digest != expected_sha256:
        raise RuntimeError(f"SHA-256 mismatch: {digest} != {expected_sha256}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial_path, output_path)
    for part in parts:
        part.path.unlink(missing_ok=True)
    try:
        part_root.rmdir()
    except OSError:
        pass
    _log(f"complete path={output_path} size={expected_size} sha256={digest}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--partial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    download(
        args.url,
        args.partial,
        args.output,
        args.expected_size,
        args.expected_sha256.lower(),
        args.workers,
        args.timeout,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
