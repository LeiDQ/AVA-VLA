#!/usr/bin/env python3
"""Validate AVA-VLA completed-stage archives and cross-stage freeze contracts."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict

import torch


STAGES = (
    "bc_complete",
    "latent_warmup_complete",
    "online_ppo_complete",
    "complete",
)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_torch(path: Path):
    kwargs = {"map_location": "cpu", "weights_only": True}
    try:
        return torch.load(str(path), mmap=True, **kwargs)
    except TypeError:
        return torch.load(path, **kwargs)


def required_file(archive: Path, manifest: Dict, prefix: str) -> Path:
    matches = [name for name in manifest["required_files"] if Path(name).name.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {prefix!r} artifact in {archive}, found {matches}")
    return archive / matches[0]


def state_digests(
    path: Path,
    selectors: Dict[str, Callable[[str], bool]],
) -> Dict[str, str]:
    state = load_torch(path)
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected a state dictionary in {path}")
    hashers = {name: hashlib.sha256() for name in selectors}
    counts = {name: 0 for name in selectors}
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        descriptor = f"{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode()
        for name, selector in selectors.items():
            if selector(key):
                hashers[name].update(descriptor)
                hashers[name].update(raw)
                counts[name] += 1
    del state
    gc.collect()
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        raise RuntimeError(f"No tensors selected for {missing} in {path}")
    return {name: hasher.hexdigest() for name, hasher in hashers.items()}


def validate_buffer(path: Path) -> int:
    rollouts = load_torch(path)
    if not isinstance(rollouts, list) or not rollouts:
        raise RuntimeError(f"Calibration buffer is empty or malformed: {path}")
    required = {"ppo_rewards", "next_latent_states", "valid_mask"}
    for index, rollout in enumerate(rollouts):
        if not isinstance(rollout, dict) or not required.issubset(rollout):
            raise RuntimeError(f"Malformed rollout {index} in {path}")
        if not all(torch.is_tensor(rollout[name]) for name in required):
            raise RuntimeError(f"Non-tensor calibration entry in rollout {index} of {path}")
        if rollout["next_latent_states"].shape[:2] != rollout["valid_mask"].shape:
            raise RuntimeError(f"Latent/mask shape mismatch in rollout {index} of {path}")
        if rollout["ppo_rewards"].shape[0] != rollout["valid_mask"].shape[0]:
            raise RuntimeError(f"Reward/mask batch mismatch in rollout {index} of {path}")
    count = len(rollouts)
    del rollouts
    gc.collect()
    return count


def validate_archive(archive: Path, schedule: Dict) -> Dict:
    manifest = load_json(archive / "CHECKPOINT_COMPLETE.json")
    stage_metadata = load_json(archive / "STAGE_CHECKPOINT.json")
    stage = archive.name
    if manifest.get("stage") != stage or stage_metadata.get("stage") != stage:
        raise RuntimeError(f"Stage identity mismatch in {archive}")
    if not stage_metadata.get("independently_resumable"):
        raise RuntimeError(f"Archive is not marked independently resumable: {archive}")
    for relative_name, expected_size in manifest.get("required_files", {}).items():
        artifact = archive / relative_name
        actual_size = artifact.stat().st_size if artifact.is_file() else -1
        if actual_size != int(expected_size) or actual_size <= 0:
            raise RuntimeError(
                f"Incomplete artifact {artifact}: size={actual_size}, expected={expected_size}"
            )

    training_path = required_file(archive, manifest, "training_state--")
    training_state = load_torch(training_path)
    paper_state = training_state.get("paper_state", {})
    log_step = int(manifest.get("log_step", -1))
    if paper_state.get("stage") != stage or int(paper_state.get("log_step", -2)) != log_step:
        raise RuntimeError(f"Manifest/training-state mismatch in {archive}")

    bc_steps = int(schedule["bc_steps"])
    warmup_steps = int(schedule["latent_warmup_steps"])
    if stage == "bc_complete":
        expected = bc_steps
        if int(paper_state.get("stage_step", -1)) != bc_steps:
            raise RuntimeError("BC stage counter is incomplete")
    elif stage == "latent_warmup_complete":
        expected = bc_steps + warmup_steps
        if int(paper_state.get("stage_step", -1)) != warmup_steps:
            raise RuntimeError("Latent-warmup stage counter is incomplete")
    else:
        ppo_update = int(paper_state.get("ppo_update", -1))
        if int(paper_state.get("global_env_steps", -1)) < int(schedule["ppo_environment_steps"]):
            raise RuntimeError(f"PPO environment-step budget is incomplete in {archive}")
        if ppo_update <= 0:
            raise RuntimeError(f"PPO update counter is invalid in {archive}")
        expected = bc_steps + warmup_steps + ppo_update
        if stage == "complete":
            exit_steps = int(schedule["exit_calibration_steps"])
            if int(paper_state.get("stage_step", -1)) != exit_steps:
                raise RuntimeError("Exit-calibration stage counter is incomplete")
            expected += exit_steps
    if log_step != expected:
        raise RuntimeError(f"Global-step mismatch in {archive}: actual={log_step}, expected={expected}")

    buffer_rollouts = {}
    if stage == "online_ppo_complete":
        buffer_names = sorted(
            name for name in manifest["required_files"]
            if Path(name).name.startswith("exit_calibration_buffer_rank")
        )
        if not buffer_names:
            raise RuntimeError("PPO completion archive has no calibration buffers")
        for name in buffer_names:
            buffer_rollouts[Path(name).name] = validate_buffer(archive / name)

    del training_state
    gc.collect()
    return {
        "log_step": log_step,
        "required_file_count": len(manifest["required_files"]),
        "boundary_checks": stage_metadata.get("boundary_checks", {}),
        "calibration_buffer_rollouts": buffer_rollouts,
        "action_path": required_file(archive, manifest, "action_head--"),
        "avavla_path": required_file(archive, manifest, "avavla--"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = load_json(run_dir / "avavla_config.json")
    schedule = config["paper_schedule"]
    archive_root = run_dir / "stage_checkpoints"
    present = [stage for stage in STAGES if (archive_root / stage).is_dir()]
    if not args.allow_partial and present != list(STAGES):
        raise RuntimeError(f"Expected completed archives {STAGES}, found {present}")
    if not present:
        raise RuntimeError(f"No completed-stage archives found under {archive_root}")

    stage_results = {
        stage: validate_archive(archive_root / stage, schedule)
        for stage in present
    }
    invariants = {}

    state_hashes = {}
    for stage in present:
        result = stage_results[stage]
        action_hash = state_digests(result.pop("action_path"), {"all": lambda _: True})["all"]
        avavla_hashes = state_digests(
            result.pop("avavla_path"),
            {
                "all": lambda _: True,
                "outside_warmup": lambda key: not key.startswith(
                    ("reasoning_policy.", "latent_transition.")
                ),
                "outside_exit_gate": lambda key: not key.startswith("exit_gate."),
                "exit_gate": lambda key: key.startswith("exit_gate."),
            },
        )
        state_hashes[stage] = {"action_head": action_hash, **avavla_hashes}

    if {"bc_complete", "latent_warmup_complete"}.issubset(state_hashes):
        bc = state_hashes["bc_complete"]
        warmup = state_hashes["latent_warmup_complete"]
        if bc["action_head"] != warmup["action_head"]:
            raise RuntimeError("Latent warmup changed the frozen action head")
        if bc["outside_warmup"] != warmup["outside_warmup"]:
            raise RuntimeError("Latent warmup changed parameters outside policy/transition dynamics")
        if bc["all"] == warmup["all"]:
            raise RuntimeError("Latent warmup completed without changing its trainable parameters")
        invariants["latent_warmup_freeze_contract"] = True

    if {"online_ppo_complete", "complete"}.issubset(state_hashes):
        ppo = state_hashes["online_ppo_complete"]
        final = state_hashes["complete"]
        if ppo["action_head"] != final["action_head"]:
            raise RuntimeError("Exit calibration changed the frozen action head")
        if ppo["outside_exit_gate"] != final["outside_exit_gate"]:
            raise RuntimeError("Exit calibration changed parameters outside the exit gate")
        if ppo["exit_gate"] == final["exit_gate"]:
            raise RuntimeError("Exit calibration completed without changing the exit gate")
        invariants["exit_calibration_freeze_contract"] = True

    report = {
        "passed": True,
        "run_dir": str(run_dir),
        "stages": stage_results,
        "cross_stage_invariants": invariants,
        "state_hashes": state_hashes,
    }
    if args.write_report:
        report_path = archive_root / "STAGE_VALIDATION.json"
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
