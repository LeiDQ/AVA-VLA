"""Validate an AVA-VLA robot base and its configured vision resolution.

This validator checks checkpoint/config consistency without constructing the
7B model.  It intentionally does not convert a 224px checkpoint into a 384px
checkpoint: the selected Prismatic base must already contain compatible
robot-pretrained weights for the requested vision backbone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_BACKBONES = {
    "prism-dinosiglip-224px+7b": "dinosiglip-vit-so-224px",
    "prism-dinosiglip-224px-controlled+7b": "dinosiglip-vit-so-224px",
    "dinosiglip-224px-resize-naive+7b": "dinosiglip-vit-so-224px",
    "prism-dinosiglip+7b": "dinosiglip-vit-so-384px",
    "prism-dinosiglip-controlled+7b": "dinosiglip-vit-so-384px",
    "dinosiglip-384px-letterbox+7b": "dinosiglip-vit-so-384px",
    "dinosiglip-384px-resize-naive+7b": "dinosiglip-vit-so-384px",
}
BACKBONE_RESOLUTIONS = {
    "dinosiglip-vit-so-224px": 224,
    "dinosiglip-vit-so-384px": 384,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(32 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object in {path}")
    return payload


def resolve_model_metadata(config: dict[str, Any]) -> dict[str, Any]:
    if isinstance(config.get("vla"), dict):
        vla = config["vla"]
        model_id = str(vla.get("base_vlm", ""))
        vision_backbone_id = MODEL_BACKBONES.get(model_id, "")
        data_mix = str(vla.get("data_mix", ""))
    elif isinstance(config.get("model"), dict):
        model = config["model"]
        model_id = str(model.get("model_id", model.get("type", "")))
        vision_backbone_id = str(model.get("vision_backbone_id", ""))
        data_mix = str(config.get("data_mix", config.get("dataset", "")))
    else:
        raise RuntimeError("config must contain a top-level 'vla' or 'model' object")

    if not vision_backbone_id:
        raise RuntimeError(
            f"cannot resolve a supported DINOv2+SigLIP backbone from model {model_id!r}"
        )
    resolution = BACKBONE_RESOLUTIONS.get(vision_backbone_id)
    if resolution is None:
        raise RuntimeError(
            f"unsupported vision backbone {vision_backbone_id!r}; "
            f"expected one of {sorted(BACKBONE_RESOLUTIONS)}"
        )

    stage = str(config.get("stage", ""))
    robot_pretrained = stage == "vla-full-train" and "oxe" in data_mix.lower()
    return {
        "model_id": model_id,
        "vision_backbone_id": vision_backbone_id,
        "resolution": resolution,
        "stage": stage,
        "data_mix": data_mix,
        "robot_pretrained": robot_pretrained,
    }


def resolve_checkpoint(
    base_dir: Path,
    manifest: dict[str, Any] | None,
    allow_unverified: bool,
) -> tuple[Path, dict[str, Any] | None]:
    if manifest is not None:
        record = manifest.get("checkpoint")
        if not isinstance(record, dict):
            raise RuntimeError("BASE_VERIFIED.json has no checkpoint record")
        relative_path = Path(str(record.get("path", "")))
        if not relative_path.parts or relative_path.is_absolute():
            raise RuntimeError("manifest checkpoint path must be relative to the base directory")
        checkpoint = (base_dir / relative_path).resolve()
        if not checkpoint.is_relative_to(base_dir):
            raise RuntimeError("manifest checkpoint escapes the base directory")
        return checkpoint, record

    if not allow_unverified:
        raise RuntimeError(
            "missing BASE_VERIFIED.json; formal 384px runs require a verified base manifest"
        )
    candidates = sorted((base_dir / "checkpoints").glob("*.pt"))
    if not candidates:
        raise RuntimeError(f"no .pt checkpoint found under {base_dir / 'checkpoints'}")
    latest = base_dir / "checkpoints" / "latest-checkpoint.pt"
    return (latest if latest.is_file() else candidates[-1]).resolve(), None


def validate(
    base_dir: Path,
    expected_resolution: int,
    full_hash: bool,
    allow_unverified: bool,
) -> dict[str, Any]:
    base_dir = base_dir.resolve()
    config_path = base_dir / "config.json"
    statistics_path = base_dir / "dataset_statistics.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing config.json: {config_path}")
    config = read_json(config_path)
    metadata = resolve_model_metadata(config)
    if not metadata["robot_pretrained"]:
        raise RuntimeError(
            "base is not an OXE robot-pretrained Prismatic checkpoint "
            f"(stage={metadata['stage']!r}, data_mix={metadata['data_mix']!r})"
        )
    if metadata["resolution"] != expected_resolution:
        raise RuntimeError(
            f"requested {expected_resolution}px but config selects "
            f"{metadata['vision_backbone_id']} ({metadata['resolution']}px)"
        )
    if not statistics_path.is_file():
        raise RuntimeError(f"missing dataset_statistics.json: {statistics_path}")

    manifest_path = base_dir / "BASE_VERIFIED.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else None
    checkpoint, checkpoint_record = resolve_checkpoint(base_dir, manifest, allow_unverified)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise RuntimeError(f"missing or empty checkpoint: {checkpoint}")

    if checkpoint_record is not None:
        expected_size = int(checkpoint_record.get("size", -1))
        expected_hash = str(checkpoint_record.get("sha256", ""))
        if expected_size <= 0 or checkpoint.stat().st_size != expected_size:
            raise RuntimeError(
                f"checkpoint size mismatch: actual={checkpoint.stat().st_size}, expected={expected_size}"
            )
        if len(expected_hash) != 64:
            raise RuntimeError("manifest checkpoint SHA-256 is missing or malformed")
        config_hash = str(manifest.get("config_sha256", ""))
        statistics_hash = str(manifest.get("dataset_statistics_sha256", ""))
        if config_hash != sha256(config_path):
            raise RuntimeError("base config hash does not match BASE_VERIFIED.json")
        if statistics_hash != sha256(statistics_path):
            raise RuntimeError("dataset statistics hash does not match BASE_VERIFIED.json")
        if full_hash and sha256(checkpoint) != expected_hash:
            raise RuntimeError("checkpoint SHA-256 does not match BASE_VERIFIED.json")
    elif full_hash:
        raise RuntimeError("--full-hash requires a BASE_VERIFIED.json checkpoint hash")

    return {
        **metadata,
        "base_dir": str(base_dir),
        "checkpoint": str(checkpoint),
        "verified_manifest": checkpoint_record is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir", type=Path)
    parser.add_argument("--expected-resolution", type=int, choices=(224, 384), required=True)
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="Development only: accept a non-empty checkpoint without an integrity manifest.",
    )
    args = parser.parse_args()
    result = validate(
        args.base_dir,
        expected_resolution=args.expected_resolution,
        full_hash=args.full_hash,
        allow_unverified=args.allow_unverified,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
