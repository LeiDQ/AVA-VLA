#!/usr/bin/env python3
"""CPU regression for completed-stage archives and independent PPO-buffer recovery."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def load_finetune_module():
    path = PROJECT_ROOT / "vla-scripts" / "finetune_avavla.py"
    spec = importlib.util.spec_from_file_location("finetune_avavla_stage_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeAVA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Linear(2, 2)
        self.reasoning_policy = nn.Linear(2, 2)
        self.latent_transition = nn.Linear(2, 2)
        self.exit_gate = nn.Linear(2, 1)

    def get_avavla_state_dict(self):
        return self.state_dict()

    def load_avavla_state_dict(self, state_dict, strict=True):
        self.load_state_dict(state_dict, strict=strict)


def make_paper_state(stage: str) -> tuple[int, dict]:
    if stage == "bc_complete":
        return 10, {"stage": stage, "log_step": 10, "stage_step": 10}
    if stage == "latent_warmup_complete":
        return 15, {"stage": stage, "log_step": 15, "stage_step": 5}
    if stage == "online_ppo_complete":
        return 18, {
            "stage": stage,
            "log_step": 18,
            "global_env_steps": 120,
            "ppo_update": 3,
        }
    if stage == "complete":
        return 20, {
            "stage": stage,
            "log_step": 20,
            "stage_step": 2,
            "global_env_steps": 120,
            "ppo_update": 3,
        }
    raise AssertionError(stage)


def main() -> int:
    finetune = load_finetune_module()
    with tempfile.TemporaryDirectory(prefix="avavla-stage-checkpoints-") as temp_dir:
        root = Path(temp_dir)
        run_dir = root / "run"
        source_config = root / "config.json"
        source_base = root / "source-base.pt"
        source_config.write_text(json.dumps({"model_id": "robot-base"}), encoding="utf-8")
        source_base.write_bytes(b"immutable-robot-base")
        finetune._link_or_copy_base_checkpoint(
            source_base,
            run_dir / "checkpoints" / "latest-checkpoint.pt",
        )

        cfg = SimpleNamespace(
            save_latest_checkpoint_only=True,
            use_l1_regression=True,
            enable_latent_reasoning=True,
            resume=False,
            vla_path=str(run_dir),
            resume_step=None,
            bc_steps=10,
            latent_warmup_steps=5,
            ppo_environment_steps=100,
            exit_calibration_steps=2,
        )
        distributed_state = SimpleNamespace(
            is_main_process=True,
            process_index=0,
            local_process_index=0,
            num_processes=1,
        )
        model = SimpleNamespace(module=FakeAVA())
        action_head = SimpleNamespace(module=nn.Linear(2, 2))
        optimizer = torch.optim.Adam(action_head.module.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        dataset = SimpleNamespace(dataset_statistics={})
        avavla_config = {
            "implementation_version": finetune.CHECKPOINT_IMPLEMENTATION_VERSION,
            "paper_schedule": {
                "bc_steps": 10,
                "latent_warmup_steps": 5,
                "ppo_environment_steps": 100,
                "exit_calibration_steps": 2,
            },
        }

        for stage in finetune.COMPLETED_STAGE_NAMES:
            with torch.no_grad():
                if stage == "latent_warmup_complete":
                    model.module.reasoning_policy.weight.add_(0.1)
                    model.module.latent_transition.weight.add_(0.1)
                elif stage == "online_ppo_complete":
                    model.module.projector.weight.add_(0.1)
                    model.module.reasoning_policy.weight.add_(0.1)
                    action_head.module.weight.add_(0.1)
                elif stage == "complete":
                    model.module.exit_gate.weight.add_(0.1)
            if stage == "online_ppo_complete":
                torch.save(
                    [{
                        "ppo_rewards": torch.ones(2),
                        "next_latent_states": torch.ones(2, 3, 4),
                        "valid_mask": torch.ones(2, 3),
                    }],
                    run_dir / "exit_calibration_buffer_rank0.pt",
                )
            step, paper_state = make_paper_state(stage)
            finetune.save_training_checkpoint(
                cfg,
                run_dir,
                step,
                model,
                source_config,
                action_head,
                dataset,
                optimizer,
                scheduler,
                distributed_state,
                avavla_config,
                paper_state,
            )
            archive = run_dir / "stage_checkpoints" / stage
            manifest = finetune._validate_checkpoint_manifest(archive)
            assert manifest["stage"] == stage
            assert json.loads((archive / "STAGE_CHECKPOINT.json").read_text())["independently_resumable"]
            assert os.stat(archive / "checkpoints" / "latest-checkpoint.pt").st_ino == os.stat(
                run_dir / "checkpoints" / "latest-checkpoint.pt"
            ).st_ino

            resumed_model = SimpleNamespace(module=FakeAVA())
            resumed_head = SimpleNamespace(module=nn.Linear(2, 2))
            resume_cfg = SimpleNamespace(
                resume=True,
                vla_path=str(archive),
                resume_step=None,
            )
            resumed = finetune.load_resume_state(
                resume_cfg,
                resumed_model,
                resumed_head,
                None,
                None,
                0,
            )
            assert resumed["paper_state"]["stage"] == stage

        ppo_archive = run_dir / "stage_checkpoints" / "online_ppo_complete"
        independent_run = root / "independent-resume"
        restore_cfg = SimpleNamespace(resume=True, vla_path=str(ppo_archive))
        restored_buffer = finetune._restore_stage_artifact_for_resume(
            restore_cfg,
            independent_run,
            "exit_calibration_buffer_rank0.pt",
        )
        assert restored_buffer.is_file()
        assert os.stat(restored_buffer).st_ino == os.stat(
            ppo_archive / "exit_calibration_buffer_rank0.pt"
        ).st_ino

        latest = finetune._validate_checkpoint_manifest(run_dir)
        assert latest["stage"] == "complete"
        component_files = [
            path.name for path in run_dir.iterdir()
            if path.name.startswith(("action_head--", "avavla--", "training_state--", "rng_state_rank"))
        ]
        assert component_files and all("step-20-complete" in name for name in component_files)

        validation = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "validate_avavla_stage_checkpoints.py"),
                str(run_dir),
                "--write-report",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if validation.returncode:
            raise RuntimeError(validation.stdout + validation.stderr)
        report = json.loads(
            (run_dir / "stage_checkpoints" / "STAGE_VALIDATION.json").read_text()
        )
        assert report["passed"]
        assert report["cross_stage_invariants"]["latent_warmup_freeze_contract"]
        assert report["cross_stage_invariants"]["exit_calibration_freeze_contract"]

    print("PASS: completed-stage archives and independent PPO-buffer recovery")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
