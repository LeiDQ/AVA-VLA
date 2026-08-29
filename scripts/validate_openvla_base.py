"""Validate an immutable robot-pretrained OpenVLA Prismatic base directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


CHECKPOINT_NAME = "step-295000-epoch-40-loss=0.2200.pt"
NATIVE_SIZE = 30_165_309_772
NATIVE_SHA256 = "2c2497cd9e0ecced65e54b0771172f22e3ed64d0c0af339e094349715d3b3602"
HF_SOURCE_REPO = "openvla/openvla-7b"
HF_SOURCE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
HF_SOURCE_FILES = {
    "model-00001-of-00003.safetensors": {
        "size": 6_948_961_960,
        "sha256": "10d8636256018712c5e5c823d12e22b5797f99bb721bd123bf6bf2379892be85",
    },
    "model-00002-of-00003.safetensors": {
        "size": 6_971_232_040,
        "sha256": "2050b14f21d48904d269f48d5a980fecea87cd7b36641d9b0f015e72d1fe216a",
    },
    "model-00003-of-00003.safetensors": {
        "size": 1_162_406_824,
        "sha256": "ea65305a1577f36f721965bf84c8caec0a948ce7ce84d754701637376c531fef",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: dict, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def validate_config(base_dir: Path) -> tuple[Path, Path]:
    config_path = base_dir / "config.json"
    statistics_path = base_dir / "dataset_statistics.json"
    if not config_path.is_file() or not statistics_path.is_file():
        raise RuntimeError("missing config.json or dataset_statistics.json")
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    vla = config.get("vla", {})
    if config.get("stage") != "vla-full-train":
        raise RuntimeError(f"unexpected training stage: {config.get('stage')!r}")
    if "oxe" not in str(vla.get("data_mix", "")).lower():
        raise RuntimeError(f"base is not robot/OXE pretrained: {vla.get('data_mix')!r}")
    if "dinosiglip" not in str(vla.get("base_vlm", "")).lower():
        raise RuntimeError(f"base is not fused DINOv2+SigLIP: {vla.get('base_vlm')!r}")
    return config_path, statistics_path


def validate_manifest(base_dir: Path, checkpoint: Path, full_hash: bool) -> dict:
    manifest_path = base_dir / "BASE_VERIFIED.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") not in {
        "openvla-prismatic-base-v1",
        "openvla-prismatic-base-conversion-v1",
    }:
        raise RuntimeError(f"unsupported base manifest format: {manifest.get('format')!r}")
    checkpoint_record = manifest.get("checkpoint", {})
    if checkpoint_record.get("path") != f"checkpoints/{CHECKPOINT_NAME}":
        raise RuntimeError("base manifest selects an unexpected checkpoint")
    expected_size = int(checkpoint_record.get("size", -1))
    expected_hash = str(checkpoint_record.get("sha256", ""))
    if checkpoint.stat().st_size != expected_size or len(expected_hash) != 64:
        raise RuntimeError("checkpoint does not match its verified manifest")

    if manifest["format"] == "openvla-prismatic-base-v1":
        if expected_size != NATIVE_SIZE or expected_hash != NATIVE_SHA256:
            raise RuntimeError("native checkpoint is not the official OpenVLA Prismatic artifact")
    else:
        if manifest.get("source_repo") != HF_SOURCE_REPO:
            raise RuntimeError("converted checkpoint source repo is not official OpenVLA")
        if manifest.get("source_revision") != HF_SOURCE_REVISION:
            raise RuntimeError("converted checkpoint source revision is unexpected")
        if manifest.get("source_files") != HF_SOURCE_FILES:
            raise RuntimeError("converted checkpoint source hashes are not official OpenVLA")
        if manifest.get("tensor_values_changed") is not False:
            raise RuntimeError("converted checkpoint changed tensor values")
        if int(manifest.get("tensor_count", -1)) != 982:
            raise RuntimeError("converted checkpoint tensor count is incomplete")

    if full_hash and sha256(checkpoint) != expected_hash:
        raise RuntimeError("checkpoint SHA-256 no longer matches its manifest")
    return manifest


def validate(base_dir: Path, full_hash: bool, write_native_manifest: bool) -> dict:
    base_dir = base_dir.resolve()
    config_path, statistics_path = validate_config(base_dir)
    checkpoint = base_dir / "checkpoints" / CHECKPOINT_NAME
    if not checkpoint.is_file():
        raise RuntimeError(f"missing checkpoint: {checkpoint}")
    manifest_path = base_dir / "BASE_VERIFIED.json"
    if manifest_path.is_file():
        manifest = validate_manifest(base_dir, checkpoint, full_hash)
    else:
        if checkpoint.stat().st_size != NATIVE_SIZE:
            raise RuntimeError("unmanifested checkpoint has the wrong official byte size")
        if not full_hash:
            raise RuntimeError("unmanifested native checkpoint requires a full hash verification")
        actual_hash = sha256(checkpoint)
        if actual_hash != NATIVE_SHA256:
            raise RuntimeError(f"native checkpoint SHA-256 mismatch: {actual_hash}")
        manifest = {
            "format": "openvla-prismatic-base-v1",
            "source_repo": "openvla/openvla-7b-prismatic",
            "checkpoint": {
                "path": f"checkpoints/{CHECKPOINT_NAME}",
                "size": NATIVE_SIZE,
                "sha256": NATIVE_SHA256,
            },
            "config_sha256": sha256(config_path),
            "dataset_statistics_sha256": sha256(statistics_path),
        }
        if write_native_manifest:
            atomic_json(manifest, manifest_path)

    if manifest.get("config_sha256") != sha256(config_path):
        raise RuntimeError("base config hash mismatch")
    if manifest.get("dataset_statistics_sha256") != sha256(statistics_path):
        raise RuntimeError("base dataset statistics hash mismatch")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--write-native-manifest", action="store_true")
    args = parser.parse_args()
    manifest = validate(args.base_dir, args.full_hash, args.write_native_manifest)
    checkpoint = manifest["checkpoint"]
    print(
        f"valid format={manifest['format']} size={checkpoint['size']} "
        f"sha256={checkpoint['sha256']} base={args.base_dir.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
