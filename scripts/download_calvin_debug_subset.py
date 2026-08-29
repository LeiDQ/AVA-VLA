#!/usr/bin/env python3
"""Download a runnable, language-aligned subset of the official CALVIN debug ZIP.

The official debug archive is about 1.3 GB.  This utility uses HTTP byte ranges
and Python's standard ``zipfile`` reader so CI/audits can fetch only metadata
and a few complete annotated demonstrations.  It preserves official frame
files and writes a correspondingly filtered ``auto_lang_ann.npy``.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_URL = "http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip"


class HTTPRangeReader(io.RawIOBase):
    """Minimal seekable HTTP reader backed by Range requests."""

    def __init__(self, url: str) -> None:
        self.url = url
        request = urllib.request.Request(url, method="HEAD")
        with self._open_with_retry(request, timeout=60) as response:
            self.length = int(response.headers["Content-Length"])
        self.position = 0

    @staticmethod
    def _open_with_retry(request: urllib.request.Request, timeout: int):
        last_error = None
        for attempt in range(8):
            try:
                return urllib.request.urlopen(request, timeout=timeout)
            except Exception as error:  # transient official-server/proxy 5xx responses
                last_error = error
                time.sleep(min(8.0, 0.5 * (2**attempt)))
        raise last_error

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            target = offset
        elif whence == os.SEEK_CUR:
            target = self.position + offset
        elif whence == os.SEEK_END:
            target = self.length + offset
        else:
            raise ValueError(f"Unsupported seek mode: {whence}")
        self.position = max(0, min(self.length, int(target)))
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.length:
            return b""
        if size is None or size < 0:
            end = self.length - 1
        else:
            end = min(self.length - 1, self.position + int(size) - 1)
        if end < self.position:
            return b""
        request = urllib.request.Request(
            self.url,
            headers={"Range": f"bytes={self.position}-{end}"},
        )
        with self._open_with_retry(request, timeout=180) as response:
            data = response.read()
        expected = end - self.position + 1
        if len(data) != expected:
            raise IOError(f"Range response returned {len(data)} bytes, expected {expected}")
        self.position = end + 1
        return data


def _safe_destination(root: Path, member_name: str) -> Path:
    relative = PurePosixPath(member_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe ZIP member path: {member_name}")
    destination = root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _extract(zip_file: zipfile.ZipFile, member_name: str, output_root: Path) -> Path:
    destination = _safe_destination(output_root, member_name)
    with zip_file.open(member_name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
    return destination


def _annotation_subset(payload: Dict, segment_count: int) -> Tuple[Dict, List[Tuple[int, int]]]:
    intervals = np.asarray(payload["info"]["indx"], dtype=np.int64)[:segment_count]
    if intervals.size == 0:
        raise ValueError("Official annotation file contains no segments")
    language = {}
    for key, values in payload["language"].items():
        language[key] = np.asarray(values)[: len(intervals)]
    subset = {"info": {"indx": intervals}, "language": language}
    return subset, [(int(row[0]), int(row[1])) for row in intervals]


def _find_member(names: Iterable[str], suffix: str) -> str:
    matches = [name for name in names if name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one ZIP member ending in {suffix!r}, found {matches}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--segments-per-split", type=int, default=1)
    args = parser.parse_args()
    if args.segments_per_split <= 0:
        raise ValueError("--segments-per-split must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    reader = HTTPRangeReader(args.url)
    extracted: List[str] = []
    with zipfile.ZipFile(reader) as archive:
        names = archive.namelist()
        for split in ("training", "validation"):
            annotation_member = _find_member(
                names,
                f"/{split}/lang_annotations/auto_lang_ann.npy",
            )
            with archive.open(annotation_member) as stream:
                payload = np.load(io.BytesIO(stream.read()), allow_pickle=True).item()
            subset, intervals = _annotation_subset(payload, args.segments_per_split)
            annotation_destination = _safe_destination(args.output, annotation_member)
            np.save(annotation_destination, subset, allow_pickle=True)
            extracted.append(annotation_member)

            prefix = annotation_member.split(f"/{split}/", 1)[0]
            for start, end in intervals:
                for frame_index in range(start, end + 1):
                    member = f"{prefix}/{split}/episode_{frame_index:07d}.npz"
                    if member not in archive.NameToInfo:
                        raise FileNotFoundError(f"Official annotation references absent member {member}")
                    _extract(archive, member, args.output)
                    extracted.append(member)

            hydra_prefix = f"{prefix}/{split}/.hydra/"
            for member in names:
                if member.startswith(hydra_prefix) and not member.endswith("/"):
                    _extract(archive, member, args.output)
                    extracted.append(member)

    print(
        f"Downloaded {len(extracted)} members from {reader.length} byte archive into "
        f"{args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
