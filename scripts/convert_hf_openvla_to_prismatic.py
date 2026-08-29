"""Convert the official HF OpenVLA weights back to the native Prismatic layout.

The official ``openvla/openvla-7b`` checkpoint was produced from the native
Prismatic checkpoint by ``convert_openvla_weights_to_hf.py``.  This utility
applies that published mapping in reverse without changing tensor values and
records source and output hashes in an atomic provenance manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict

import torch
from safetensors import safe_open


SOURCE_REPO = "openvla/openvla-7b"
SOURCE_REVISION = "47a0ec7fc4ec123775a391911046cf33cf9ed83f"
SOURCE_SHARDS = {
    "model-00001-of-00003.safetensors": (
        6_948_961_960,
        "10d8636256018712c5e5c823d12e22b5797f99bb721bd123bf6bf2379892be85",
    ),
    "model-00002-of-00003.safetensors": (
        6_971_232_040,
        "2050b14f21d48904d269f48d5a980fecea87cd7b36641d9b0f015e72d1fe216a",
    ),
    "model-00003-of-00003.safetensors": (
        1_162_406_824,
        "ea65305a1577f36f721965bf84c8caec0a948ce7ce84d754701637376c531fef",
    ),
}
PROJECTOR_INVERSE = {
    "projector.fc1.weight": "projector.0.weight",
    "projector.fc1.bias": "projector.0.bias",
    "projector.fc2.weight": "projector.2.weight",
    "projector.fc2.bias": "projector.2.bias",
    "projector.fc3.weight": "projector.4.weight",
    "projector.fc3.bias": "projector.4.bias",
}
EXPECTED_COMPONENT_COUNTS = {"vision_backbone": 685, "projector": 6, "llm_backbone": 291}
CHECKPOINT_NAME = "step-295000-epoch-40-loss=0.2200.pt"


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


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copyfile(source, temporary)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def map_key(hf_key: str) -> tuple[str, str]:
    if hf_key in PROJECTOR_INVERSE:
        return "projector", PROJECTOR_INVERSE[hf_key]
    if hf_key.startswith("language_model."):
        return "llm_backbone", hf_key.replace("language_model.", "llm.", 1)
    if hf_key.startswith("vision_backbone.fused_featurizer."):
        return "vision_backbone", hf_key.replace(
            "vision_backbone.fused_featurizer.", "siglip_featurizer.", 1
        )
    if hf_key.startswith("vision_backbone.featurizer."):
        native_key = hf_key.replace("vision_backbone.featurizer.", "dino_featurizer.", 1)
        if native_key.endswith(".scale_factor"):
            native_key = native_key[: -len(".scale_factor")] + ".gamma"
        return "vision_backbone", native_key
    raise ValueError(f"Unmapped HF OpenVLA key: {hf_key}")


def validate_source(hf_dir: Path) -> dict[str, dict[str, object]]:
    verified = {}
    for name, (expected_size, expected_hash) in SOURCE_SHARDS.items():
        path = hf_dir / name
        if not path.is_file() or path.stat().st_size != expected_size:
            raise RuntimeError(f"Missing or wrongly sized official source shard: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Source shard hash mismatch for {name}: {actual_hash}")
        verified[name] = {"size": expected_size, "sha256": actual_hash}
    return verified


def convert(hf_dir: Path, template_dir: Path, output_dir: Path) -> Path:
    source_files = validate_source(hf_dir)
    index_path = hf_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(index_path)
    with index_path.open(encoding="utf-8") as stream:
        weight_map = json.load(stream)["weight_map"]
    if len(weight_map) != 982:
        raise RuntimeError(f"Expected 982 official OpenVLA tensors, found {len(weight_map)}")

    components: Dict[str, Dict[str, torch.Tensor]] = {
        "vision_backbone": {},
        "projector": {},
        "llm_backbone": {},
    }
    seen = set()
    for shard_name in SOURCE_SHARDS:
        expected_keys = {key for key, filename in weight_map.items() if filename == shard_name}
        with safe_open(hf_dir / shard_name, framework="pt", device="cpu") as shard:
            actual_keys = set(shard.keys())
            if actual_keys != expected_keys:
                missing = sorted(expected_keys - actual_keys)[:5]
                extra = sorted(actual_keys - expected_keys)[:5]
                raise RuntimeError(f"Index mismatch in {shard_name}: missing={missing}, extra={extra}")
            for hf_key in sorted(actual_keys):
                component, native_key = map_key(hf_key)
                identity = (component, native_key)
                if identity in seen:
                    raise RuntimeError(f"Duplicate mapped tensor: {identity}")
                seen.add(identity)
                components[component][native_key] = shard.get_tensor(hf_key)

    counts = {name: len(values) for name, values in components.items()}
    if counts != EXPECTED_COMPONENT_COUNTS:
        raise RuntimeError(f"Unexpected mapped component counts: {counts}")

    config_source = template_dir / "config.json"
    statistics_source = template_dir / "dataset_statistics.json"
    if not config_source.is_file() or not statistics_source.is_file():
        raise FileNotFoundError("Native OpenVLA config/statistics template is incomplete")
    with config_source.open(encoding="utf-8") as stream:
        native_config = json.load(stream)
    if native_config.get("stage") != "vla-full-train" or "oxe" not in str(
        native_config.get("vla", {}).get("data_mix", "")
    ).lower():
        raise RuntimeError("Template is not the robot-pretrained OpenVLA Prismatic config")

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / CHECKPOINT_NAME
    temporary = checkpoint.with_name(f".{checkpoint.name}.conversion-tmp")
    temporary.unlink(missing_ok=True)
    torch.save(
        {
            "model": components,
            "provenance": {
                "source_repo": SOURCE_REPO,
                "source_revision": SOURCE_REVISION,
                "mapping": "official_convert_openvla_weights_to_hf.py inverse-v1",
                "tensor_values_changed": False,
            },
        },
        temporary,
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, checkpoint)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_copy(config_source, output_dir / "config.json")
    atomic_copy(statistics_source, output_dir / "dataset_statistics.json")
    checkpoint_hash = sha256(checkpoint)
    manifest = {
        "format": "openvla-prismatic-base-conversion-v1",
        "source_repo": SOURCE_REPO,
        "source_revision": SOURCE_REVISION,
        "source_files": source_files,
        "tensor_count": len(seen),
        "component_tensor_counts": counts,
        "tensor_values_changed": False,
        "checkpoint": {
            "path": f"checkpoints/{CHECKPOINT_NAME}",
            "size": checkpoint.stat().st_size,
            "sha256": checkpoint_hash,
        },
        "config_sha256": sha256(output_dir / "config.json"),
        "dataset_statistics_sha256": sha256(output_dir / "dataset_statistics.json"),
    }
    atomic_json(manifest, output_dir / "BASE_VERIFIED.json")
    print(
        f"complete checkpoint={checkpoint} size={checkpoint.stat().st_size} "
        f"sha256={checkpoint_hash} tensors={len(seen)}",
        flush=True,
    )
    return checkpoint


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-dir", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    convert(args.hf_dir.resolve(), args.template_dir.resolve(), args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
