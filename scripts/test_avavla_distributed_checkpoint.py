"""Eight-rank regression for AVA-VLA atomic checkpoint publication."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)


def _load_finetune_module():
    path = PROJECT_ROOT / "vla-scripts" / "finetune_avavla.py"
    spec = importlib.util.spec_from_file_location("finetune_avavla_distributed_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeAVA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Linear(2, 2)

    def get_avavla_state_dict(self):
        return {"projector.weight": self.projector.weight.detach().cpu()}


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_avavla_distributed_checkpoint.py SHARED_TEMP_DIR")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 8

    finetune = _load_finetune_module()
    root = Path(sys.argv[1])
    run_dir = root / "run"
    source_config = root / "config.json"
    source_base = root / "source-base.pt"

    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        source_config.write_text(json.dumps({"model_id": "robot-base"}), encoding="utf-8")
        source_base.write_bytes(b"immutable-robot-base")
        finetune._link_or_copy_base_checkpoint(
            source_base,
            run_dir / "checkpoints" / "latest-checkpoint.pt",
        )
    dist.barrier()

    distributed_state = SimpleNamespace(
        is_main_process=rank == 0,
        process_index=rank,
        local_process_index=local_rank,
        num_processes=world_size,
    )
    cfg = SimpleNamespace(
        save_latest_checkpoint_only=True,
        use_l1_regression=True,
        enable_latent_reasoning=True,
    )
    model = SimpleNamespace(module=_FakeAVA())
    action_module = nn.Linear(2, 2)
    action_head = SimpleNamespace(module=action_module)
    optimizer = torch.optim.AdamW(action_module.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    dataset = SimpleNamespace(dataset_statistics={})
    avavla_config = {"implementation_version": finetune.CHECKPOINT_IMPLEMENTATION_VERSION}

    finetune.save_training_checkpoint(
        cfg,
        run_dir,
        17,
        model,
        source_config,
        action_head,
        dataset,
        optimizer,
        scheduler,
        distributed_state,
        avavla_config,
        {"stage": "bc"},
    )
    if rank == 0:
        torch.save({"partial": True}, run_dir / "avavla--step-999-crash_checkpoint.pt")
    dist.barrier()
    finetune.save_training_checkpoint(
        cfg,
        run_dir,
        18,
        model,
        source_config,
        action_head,
        dataset,
        optimizer,
        scheduler,
        distributed_state,
        avavla_config,
        {"stage": "ppo"},
    )

    rank_sum = torch.tensor(float(rank), device=torch.device("cuda", local_rank))
    dist.all_reduce(rank_sum)
    assert rank_sum.item() == 28.0

    if rank == 0:
        manifest = finetune._validate_checkpoint_manifest(run_dir)
        assert manifest["implementation_version"] == finetune.CHECKPOINT_IMPLEMENTATION_VERSION
        assert manifest["log_step"] == 18
        required = set(manifest["required_files"])
        assert len([name for name in required if name.startswith("rng_state_rank")]) == 8
        assert all("step-18-ppo" in name for name in required if "--step-" in name)
        leftovers = [
            path.name
            for path in run_dir.iterdir()
            if path.name.startswith(("action_head--", "avavla--", "training_state--", "rng_state_rank"))
            and "step-18-ppo" not in path.name
        ]
        assert not leftovers, leftovers
        print("PASS: 8-GPU NCCL and two-generation atomic checkpoint contract")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
