# AVA-VLA

Official implementation and reproduction toolkit for **AVA-VLA: Think Less, Act Early**.

[Paper](https://arxiv.org/abs/2606.15099) · [Detailed implementation notes](IMPLEMENTATION_REPRODUCTION_NOTES.md) · [LIBERO setup](LIBERO.md) · [General setup](SETUP.md)

AVA-VLA adds reinforced latent reasoning and dynamic early exit to an OpenVLA/OpenVLA-OFT policy. This repository provides the four-stage training pipeline, distributed online rollouts, checkpoint recovery, benchmark evaluation, and result aggregation used for the main experiments.

## Installation

Python 3.10 and CUDA-capable PyTorch are recommended. The reference environment uses PyTorch 2.2. The launchers use the repository-local `.venv` environment created below.

```bash
git submodule update --init third_party/LIBERO

python3.10 -m venv .venv
source .venv/bin/activate

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
pip install -e third_party/LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

Set the runtime environment from the repository root:

```bash
export LIBERO_CONFIG_PATH="$PWD/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD/third_party/LIBERO:$PWD"
export TOKENIZERS_PARALLELISM=false
```

For CALVIN, initialize its pinned submodule and install the simulator dependencies:

```bash
git submodule update --init --recursive third_party/calvin
bash scripts/install_calvin_runtime.sh
```

## Models and datasets

Place the robot-pretrained OpenVLA checkpoint, tokenizer, and LIBERO RLDS datasets under the following paths, or override the corresponding launcher environment variables:

```text
models/
├── openvla-7b-modelscope-prismatic/
│   ├── BASE_VERIFIED.json
│   ├── config.json
│   ├── dataset_statistics.json
│   └── checkpoints/step-295000-epoch-40-loss=0.2200.pt
├── openvla-7b-dinosiglip-384-prismatic/  # optional paper-resolution base
│   ├── BASE_VERIFIED.json
│   ├── config.json
│   ├── dataset_statistics.json
│   └── checkpoints/<verified-robot-checkpoint>.pt
└── llama2-7b-ms-tokenizer/

data/modified_libero_rlds/
├── libero_spatial_no_noops/1.0.0/
├── libero_object_no_noops/1.0.0/
├── libero_goal_no_noops/1.0.0/
└── libero_10_no_noops/1.0.0/
```

The training launchers expect an OpenVLA/OXE robot-pretrained base with a DINOv2+SigLIP vision backbone. Validate the checkpoint before starting a full run:

```bash
./.venv/bin/python scripts/validate_openvla_base.py \
  models/openvla-7b-modelscope-prismatic --full-hash
```

### Optional 384px vision encoder

The default launcher profile remains `openvla-224`, so existing runs continue to use the verified
`prism-dinosiglip-224px+7b` base. To use the paper-resolution visual encoder, select
`VISION_BACKBONE_PROFILE=dinosiglip-384`. This profile uses the fused
`DINOv2 + SigLIP-SO/14 @ 384px` backbone registered as `dinosiglip-vit-so-384px` and enables the
runtime `--require_384px_backbone true` guard.

A 224px OpenVLA checkpoint is **not** converted or partially loaded into the 384px encoder. Supply a
separate Prismatic checkpoint that is already OXE/robot-pretrained with `prism-dinosiglip+7b`, plus
matching dataset statistics and `BASE_VERIFIED.json`. The generic multimodal
`models/prism-dinosiglip+7b` artifact is not an OXE robot base and is intentionally rejected.
Validate a candidate before allocating GPUs:

```bash
./.venv/bin/python scripts/validate_avavla_vision_base.py \
  models/openvla-7b-dinosiglip-384-prismatic \
  --expected-resolution 384 --full-hash
```

Run one-policy-per-suite reproduction with 384px input:

```bash
mkdir -p logs/paper_reproduction_dinosiglip384
nohup env \
  VISION_BACKBONE_PROFILE=dinosiglip-384 \
  BASE_MODEL="$PWD/models/openvla-7b-dinosiglip-384-prismatic" \
  PAPER_SEEDS="0 1 2" EVAL_SHARDS=8 \
  bash scripts/run_paper_libero_per_suite.sh \
  >>logs/paper_reproduction_dinosiglip384/launcher.log 2>&1 </dev/null &
```

The 384px profile writes to `runs/paper_per_suite_dinosiglip384`,
`logs/paper_reproduction_dinosiglip384`, and `results/paper_per_suite_dinosiglip384` by default,
so it cannot reuse or overwrite an active 224px run. The all-suite launcher accepts the same two
environment variables and uses corresponding `paper_all_suites_dinosiglip384` directories.
Saved checkpoints record the 384px requirement; deployment reconstructs the encoder and image
transform from that saved configuration.

CALVIN ABC→D uses the official frame and language-annotation layout:

```text
calvin_data/task_ABC_D/
├── training/
│   ├── episode_*.npz
│   └── lang_annotations/auto_lang_ann.npy
└── validation/
    ├── episode_*.npz
    ├── lang_annotations/auto_lang_ann.npy
    └── .hydra/merged_config.yaml
```

A small interface-test subset can be downloaded without retrieving the full dataset:

```bash
./.venv/bin/python scripts/download_calvin_debug_subset.py \
  --output calvin_data/subset --segments-per-split 1
```

## Training protocol

AVA-VLA training consists of four consecutive stages:

1. Behavior cloning on robot demonstrations.
2. Latent reasoning warmup.
3. Joint online PPO with benchmark task rewards.
4. Exit-gate calibration.

The main experiment configuration is:

| Setting | Value |
|---|---:|
| GPUs | 8 |
| Training seeds | 0, 1, 2 |
| BC steps | 100,000 |
| Global BC batch size | 64 |
| Per-rank RLDS shuffle buffer | 2,048 examples |
| BC checkpoint interval | 1,000 steps |
| Latent warmup steps | 50,000 |
| PPO environment steps | 1,200,000 |
| PPO effective batch size | 512 |
| PPO minibatch size | 64 |
| PPO epochs | 4 |
| Policy learning rate | 3e-5 |
| Critic learning rate | 1e-4 |
| PPO clip ratio | 0.2 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Smoothness coefficient | 0.1 |
| Action PPO exploration std | 0.05 in normalized action space |
| Action chunk length | 8 |
| Observation history window | 9 frames |
| Maximum reasoning steps | 5 |
| Exit threshold | 0.55 |
| Exit calibration steps | 10,000 |
| Exit lookahead | 3 |
| Exit delta | 0.05 |
| Gradient clipping | 1.0 |

The installed reference robot base is `prism-dinosiglip-224px+7b`; the optional
`dinosiglip-384` profile selects `prism-dinosiglip+7b` from a separate compatible robot base.
Training, online rollout, and deterministic evaluation derive image resolution and transforms from the selected checkpoint and share the same proprioception statistics, action normalization, action chunking, and history convention.

## Reproducing Table 1

### One policy per suite

Run all four suites with three training seeds:

```bash
mkdir -p logs/paper_reproduction

nohup env \
  BASE_MODEL="$PWD/models/openvla-7b-modelscope-prismatic" \
  DATA_ROOT="$PWD/data/modified_libero_rlds" \
  PAPER_SEEDS="0 1 2" \
  EVAL_SHARDS=8 \
  bash scripts/run_paper_libero_per_suite.sh \
  >>logs/paper_reproduction/launcher.log 2>&1 </dev/null &
```

The launcher processes Spatial, Object, Goal, and Long sequentially. Each trained checkpoint is evaluated for 500 episodes: 10 tasks × 50 trials.

To run one suite and one seed, set a suite filter:

```bash
env \
  BASE_MODEL="$PWD/models/openvla-7b-modelscope-prismatic" \
  DATA_ROOT="$PWD/data/modified_libero_rlds" \
  PAPER_SEEDS="0" \
  PAPER_SUITE_FILTER="libero_spatial" \
  EVAL_SHARDS=8 \
  bash scripts/run_paper_libero_per_suite.sh
```

### One policy for all four suites

This setting uses the four-dataset RLDS mixture `libero_4_task_suites_no_noops`. The BC loader follows the dataset-size-balanced sampling implemented by the OpenVLA RLDS pipeline. Each DDP rank collects equal numbers of online rollouts from Spatial, Object, Goal, and Long while applying suite-specific normalization statistics.

```bash
mkdir -p logs/paper_all_suites

nohup env \
  BASE_MODEL="$PWD/models/openvla-7b-modelscope-prismatic" \
  DATA_ROOT="$PWD/data/modified_libero_rlds" \
  PAPER_SEEDS="0 1 2" \
  EVAL_SHARDS=8 \
  bash scripts/run_paper_libero_all_suites.sh \
  >>logs/paper_all_suites/launcher.log 2>&1 </dev/null &
```

For each seed, the launcher trains one checkpoint and evaluates it on all four suites. The aggregation step verifies checkpoint identity across the four result sets.

### Table 1 aggregation

The launchers aggregate results automatically after all requested runs finish. Existing evaluation files can also be aggregated directly:

```bash
./.venv/bin/python scripts/aggregate_table1_ours.py \
  --result-root results/paper_per_suite \
  --mode per_suite \
  --seeds 0 1 2 \
  --output results/paper_per_suite/table1_ours_per_suite.json

./.venv/bin/python scripts/aggregate_table1_ours.py \
  --result-root results/paper_all_suites \
  --mode all_policy \
  --seeds 0 1 2 \
  --output results/paper_all_suites/table1_ours_all_policy.json
```

## Reproducing Table 3: CALVIN ABC→D

The native CALVIN adapter streams `episode_XXXXXXX.npz` files and language annotations directly. It constructs 9-frame observation histories, 8-step relative-action targets, static/wrist RGB inputs, and AVA-VLA proprioception features. Dataset statistics are cached in `avavla_calvin_statistics.json` after the first scan.

Run training and the official 1,000-sequence evaluation:

```bash
CALVIN_DATA_ROOT="$PWD/calvin_data/task_ABC_D" \
SEED=0 EVAL_SHARDS=8 \
bash scripts/run_paper_calvin_abc.sh
```

Evaluation reports SR@1 through SR@5 and average completed sequence length. The task oracle supplies success rewards during online PPO.

To validate the official sequence protocol without loading a model:

```bash
PYTHONPATH="$PWD/third_party/calvin_runtime:$PWD/third_party/calvin/calvin_models:$PWD/third_party/calvin/calvin_env:$PWD" \
./.venv/bin/python experiments/robot/calvin/run_calvin_eval.py \
  --protocol-only --num-sequences 10 \
  --output results/calvin_protocol_check.json
```

## Validation and smoke tests

Run the CPU/static contracts before allocating a full training job:

```bash
./.venv/bin/python -u scripts/test_avavla_regressions.py
./.venv/bin/python -u scripts/test_avavla_safety_regressions.py
./.venv/bin/python -u scripts/test_libero_eval_sharding.py
./.venv/bin/python -u scripts/test_libero_multi_suite.py
./.venv/bin/python -u scripts/test_paper_table_adapters.py
```

Run the shortest 8-GPU four-stage integration test:

```bash
bash scripts/run_official_base_preflight.sh \
  models/openvla-7b-modelscope-prismatic
```

Run the Table 1 all-suite integration test with 200 BC steps, 50 warmup steps, one 4,096-environment-step PPO update, and 20 exit-calibration steps:

```bash
bash scripts/run_table1_all_suites_short.sh
```

These tests validate execution, distributed communication, online environment interaction, metrics, and checkpoint structure. Reproduction results should be reported only from the full training and evaluation budgets above.

## Monitoring and outputs

Monitor a per-suite run with:

```bash
tail -f logs/paper_reproduction/status.log
tail -f logs/paper_reproduction/libero_spatial.seed0.train.log
watch -n 2 nvidia-smi
```

The launcher writes `pipeline.pid` automatically. Start 30-second GPU, process, filesystem, and
host-memory telemetry with:

```bash
nohup bash scripts/record_paper_telemetry.sh \
  >>logs/paper_reproduction/telemetry.log 2>&1 </dev/null &
```

The CSV outputs are `gpu_telemetry.csv`, `process_telemetry.csv`, and
`host_memory_telemetry.csv` under the selected log directory. Set `LOG_ROOT` when monitoring an
all-suite run with a custom log directory.

Training metrics are written as JSON Lines:

```text
runs/paper_per_suite/paper_<suite>_seed<seed>/metrics.jsonl
```

Generate loss plots and CSV data with:

```bash
./.venv/bin/python scripts/plot_avavla_losses.py \
  runs/paper_per_suite/paper_spatial_seed0/metrics.jsonl \
  --output-dir results/paper_per_suite/libero_spatial/seed0/training_curves
```

Main result files:

```text
results/paper_per_suite/<suite>/seed<seed>/evaluation_results.json
results/paper_per_suite/table1_ours_per_suite.json
results/paper_all_suites/table1_ours_all_policy.json
```

LIBERO evaluation uses 10 tasks, 50 trials per task, center crop, proprioception normalization, latent history, and an open-loop action chunk of 8.

## Checkpoint recovery

Each resumable checkpoint contains a `CHECKPOINT_COMPLETE.json` manifest. The manifest records all required components and their byte sizes so interrupted or partially copied checkpoints are not resumed as complete runs.

BC checkpoints are written every 1,000 steps. PPO checkpoints are written every 10 updates and at the environment-step boundary. Exit calibration checkpoints are written every 1,000 steps. The paper launchers validate and resume compatible checkpoints automatically. `SHUFFLE_BUFFER_SIZE` and `BC_SAVE_FREQ` may be overridden for a different host-memory or storage profile.

For a manual distributed resume, keep all experiment arguments identical to the original run:

```bash
./.venv/bin/torchrun --standalone --nproc-per-node=8 \
  vla-scripts/finetune_avavla.py \
  --vla_path runs/paper_per_suite/paper_spatial_seed0 \
  --resume true \
  ...
```

## Repository structure

```text
prismatic/models/vlas/avavla.py             AVA-VLA model and training objectives
vla_scripts/online_policy.py                Online policy utilities and normalization
vla-scripts/finetune_avavla.py              Four-stage distributed trainer
vla-scripts/deploy_avavla.py                Deterministic inference and deployment
experiments/robot/libero/online_rollout.py   LIBERO online collectors
experiments/robot/calvin/dataset.py          CALVIN dataset adapter
experiments/robot/calvin/online_rollout.py   CALVIN online collector
experiments/robot/calvin/run_calvin_eval.py  CALVIN sequence evaluation
scripts/aggregate_table1_ours.py             Table 1 aggregation
```

Architecture details, PPO data flow, normalization contracts, checkpoint schema, and extended debugging instructions are documented in [IMPLEMENTATION_REPRODUCTION_NOTES.md](IMPLEMENTATION_REPRODUCTION_NOTES.md).

## Citation

If AVA-VLA is useful for your research, please cite the AVA-VLA paper together with the OpenVLA/OpenVLA-OFT works used by your experiment. Citation metadata is available in [CITATION.cff](CITATION.cff).

## Acknowledgements

This project builds on OpenVLA/OpenVLA-OFT, LIBERO, and CALVIN. Please follow the licenses and citation requirements of the corresponding upstream projects and datasets.
