# AVA-VLA 实现与复现实验说明

本文档保存训练实现、兼容性约束、实验参数和完整命令的详细说明，供开发、审阅与实验排查使用。对外项目介绍和标准复现入口见 [README.md](README.md)。

本仓库实现论文 [AVA-VLA: Think Less, Act Early](https://arxiv.org/abs/2606.15099) 的四阶段训练流程，并以 OpenVLA / OpenVLA-OFT 为机器人预训练基座。

Checkpoint 恢复不是只检查文件是否存在：训练和评测会验证原子 manifest、文件大小、动作策略参数化及当前 Stage 3 PPO 所需的 schema。与当前 PPO 或推理语义不兼容的旧 checkpoint 会被拒绝或由正式 launcher 归档，不能直接续训。

## 当前实现状态

Table 1 支持论文中的两种 Ours 设置：

- **One policy for all 4 suites**：一个 checkpoint 联合训练并评测四套。
- **One policy per suite**：Spatial、Object、Goal、Long 分别训练 checkpoint。

四个 LIBERO suite 为：

- `libero_spatial`
- `libero_object`
- `libero_goal`
- `libero_10`（论文表格中的 Long）

训练严格分为四个阶段：

1. Behavior Cloning：100,000 steps，全局 batch size 64。
2. Latent Reasoning Warmup：50,000 steps；冻结 action policy，以示范动作 L1 保持任务对齐，设置 `lambda_1=0`，并使用 latent smoothness 作为该阶段唯一正则项。
3. Joint PPO：约 1,200,000 个真实环境交互 steps。
4. Exit Gate Calibration：`k=3`，边际改善阈值 `delta=0.05`。

正式复现默认使用 8 张 GPU 和 3 个训练 seed。每个 checkpoint 在目标 suite 上执行 500 个标准评测回合（10 tasks × 50 trials）；all-four 设置必须用同一个 checkpoint 分别评测四套。

LIBERO 的 RLDS shuffle buffer 默认按 rank 配置为 2,048 个样本。8 卡进程各自持有独立的 TensorFlow/RLDS pipeline，因此该值不是跨卡共享的全局容量。正式入口每 1,000 个 BC step 写入一次原子 checkpoint；可通过 `SHUFFLE_BUFFER_SIZE` 和 `BC_SAVE_FREQ` 调整。

Table 3 使用官方 CALVIN ABC→D 协议：在 A/B/C 数据上训练，在 D 环境执行 1,000 条五任务连续序列，输出 SR@1..5 和平均连续任务长度。Table 5 不重新训练模型，而是使用 Table 1 的 all-four checkpoint 扫不同 early-exit threshold。

> 代码不会伪造或人为抬高成功率。论文表格中的结果只能通过完整训练和真实仿真评测验证，不由 smoke test 保证。

## 关键正确性与兼容性约束

### 端到端 Stage 3 PPO

在线 rollout 保存 latent update actions、旧策略 log-prob、FP32 latent state 和 observation encoding。每个 PPO minibatch 都在冻结的 rollout state 上使用当前 reasoning policy 重新计算：

```text
frozen rollout latent state + observation encoding
  -> current reasoning policy likelihood
  -> clipped PPO objective
```

PPO 的 likelihood ratio 使用 rollout 中冻结的 FP32 latent state 和 observation encoding，避免在同一 rollout 的多轮更新中通过已经变化的 transition 反复重建状态。结构化日志记录 ratio、clip fraction、更新前后 approximate KL 和回溯缩放比例。Adam 候选步若超过 target-KL 信赖域，会在写入 policy 前回溯缩小；后续 epoch 若已到达边界则提前停止。

### Action policy 的训练边界

Action head 直接预测归一化连续动作。论文第 3.5 节明确给出的 PPO policy 是 latent reasoning policy；因此正式配置不再人为给 56 维 OFT action chunk 添加论文未定义的固定方差 Gaussian PPO。在线动作使用 action head 的确定性均值，Stage 3 每轮 PPO 后的 demonstration BC auxiliary update 负责保持和更新 action policy、latent-to-action 接口与多模态投影。

代码仍保留可选的 action-space PPO 消融入口；只有显式设置 `action_ppo_coef > 0` 时才启用，并受同一个 KL guard 约束。正式 launcher 使用 `action_ppo_coef=0`。

### 其他已统一的训练/推理语义

- Observation encoder 不读取 demonstration action tokens，避免 BC 标签泄漏。
- Proprioception 在 BC、PPO 和评测中使用相同 RLDS 统计量归一化。
- Center crop 在训练、在线 rollout 和评测中一致生效。
- History 对齐为前一个 policy decision，即 8 个控制 steps，默认 `history_window_size=9`。
- 稀疏任务奖励只放在 action chunk 的最后一个有效 latent step，由 semi-Markov GAE 反向传播信用。
- 连续 Gaussian 的 entropy penalty 使用逐维、截断为非负的 differential entropy，避免负 entropy 反向成为奖励。
- Entropy 和 smoothness 只进入 composite reward 一次，不重复加入 loss。
- PPO replay 与 rollout 都使用 eval mode，首次 likelihood ratio 不受 dropout 扰动。
- PPO 使用提交前 target-KL 回溯和 epoch early stop；异常的首次 on-policy KL 会直接中止而不是静默训练。
- Exit gate 使用跨多个 rollout 的紧凑校准缓冲区。
- Checkpoint 原子写入并支持 BC、warmup、PPO 和 gate calibration 的真实断点续训。
- 评测必须完成准确的 500 episodes；异常或缺失 shard 不会被当作成功。

## 目录和必需资源

在仓库根目录应存在：

```text
.venv/
models/
├── openvla-7b-modelscope-prismatic/
│   ├── BASE_VERIFIED.json
│   ├── config.json
│   ├── dataset_statistics.json
│   └── checkpoints/step-295000-epoch-40-loss=0.2200.pt
└── llama2-7b-ms-tokenizer/
data/modified_libero_rlds/
├── libero_spatial_no_noops/1.0.0/
├── libero_object_no_noops/1.0.0/
├── libero_goal_no_noops/1.0.0/
└── libero_10_no_noops/1.0.0/
third_party/LIBERO/
```

不要使用只在 LLaVA 图文数据上训练的通用 `prism-dinosiglip+7b` 作为正式机器人基座。正式流水线会校验 OpenVLA/OXE 机器人预训练信息、DINOv2+SigLIP 配置以及 checkpoint manifest。

CALVIN 实验另外需要官方仓库和 ABC→D 数据：

```text
third_party/calvin/                  官方 CALVIN Git submodule
third_party/calvin_runtime/          本地安装的 simulator 运行时依赖，不进入 Git
calvin_data/task_ABC_D/
├── training/
│   ├── episode_*.npz
│   └── lang_annotations/auto_lang_ann.npy
└── validation/
    ├── episode_*.npz
    ├── lang_annotations/auto_lang_ann.npy
    └── .hydra/merged_config.yaml
```

如果只验证 CALVIN 接口，可以使用 `scripts/download_calvin_debug_subset.py` 按 HTTP Range 下载两个完整语言片段，无需获取整个 debug archive。该小数据只用于接口测试，不等价于 ABC→D 训练数据。

## 环境

推荐 Python 3.10。以下命令假定已经在仓库根目录创建 `.venv`：

```bash
source .venv/bin/activate
export LIBERO_CONFIG_PATH="$PWD/.libero"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD/third_party/LIBERO:$PWD"
export TOKENIZERS_PARALLELISM=false
```

如需从头安装：

```bash
git submodule update --init third_party/LIBERO
conda create -n avavla python=3.10 -y
conda activate avavla
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
pip install -e .
pip install packaging ninja
pip install "flash-attn==2.5.5" --no-build-isolation
pip install -e third_party/LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

初始化 CALVIN submodule 和隔离运行时：

```bash
git submodule update --init --recursive third_party/calvin
bash scripts/install_calvin_runtime.sh
```

## 1. 先运行静态与 CPU 回归测试

```bash
./.venv/bin/python scripts/validate_openvla_base.py \
  models/openvla-7b-modelscope-prismatic --full-hash

./.venv/bin/python -u scripts/test_avavla_regressions.py
./.venv/bin/python -u scripts/test_avavla_safety_regressions.py
./.venv/bin/python -u scripts/test_libero_eval_sharding.py
./.venv/bin/python -u scripts/test_libero_multi_suite.py
./.venv/bin/python -u scripts/test_paper_table_adapters.py
```

预期结果：

- AVA-VLA regression contracts：全部通过。
- Formal safety contracts：全部通过。
- LIBERO 8-shard coverage/merge/corruption rejection：通过。
- 四套分配、per-observation normalization 和 per-suite metrics：通过。
- Table 1 聚合、CALVIN adapter/protocol 和 Table 5 telemetry：通过。

有 CALVIN debug subset 时再运行真实数据和 simulator 测试：

```bash
./.venv/bin/python scripts/download_calvin_debug_subset.py \
  --output calvin_data/subset --segments-per-split 1

./.venv/bin/python scripts/test_calvin_debug_data.py \
  --dataset-root calvin_data/subset \
  --output audit_artifacts/calvin_debug_data_test.json
```

## 2. 单卡真实 smoke test

先用 GPU 0 验证单个 suite 的四阶段训练、真实 LIBERO `env.step`、checkpoint、结构化日志和 loss 曲线：

```bash
bash scripts/run_libero_suite_smoke.sh \
  libero_spatial \
  libero_spatial_no_noops \
  0 \
  models/openvla-7b-modelscope-prismatic \
  1 \
  5min
```

输出位置：

```text
runs/suite_smoke/official_openvla_smoke_libero_spatial_5min_1gpu/
logs/suite_smoke/libero_spatial_5min_1gpu/
```

查看进度：

```bash
tail -f logs/suite_smoke/libero_spatial_5min_1gpu/train.log
watch -n 2 nvidia-smi
```

该 wrapper 遇到瞬时失败会保留日志并自动重试。`5min` 是链路验证，不代表论文指标。

## 3. 8 卡全阶段预检

单卡通过后再执行 8 卡最短正式预检：

```bash
bash scripts/run_official_base_preflight.sh \
  models/openvla-7b-modelscope-prismatic
```

该命令会执行 BC、latent warmup、真实在线 PPO 和 exit calibration，并验证每个 stage 的必需 metrics、连续 `global_step`、有限 loss、PPO 路径重算标志及组件梯度。

日志与结果：

```text
logs/paper_reproduction/preflight.train.log
runs/preflight/official_openvla_avavla_preflight/metrics.jsonl
```

## 4. 四个 suite 的 8 卡 smoke test

四条命令依次执行；每条都会占用全部 8 张 GPU：

```bash
bash scripts/run_libero_suite_smoke.sh libero_spatial libero_spatial_no_noops 0,1,2,3,4,5,6,7 models/openvla-7b-modelscope-prismatic 8 100bc
bash scripts/run_libero_suite_smoke.sh libero_object  libero_object_no_noops  0,1,2,3,4,5,6,7 models/openvla-7b-modelscope-prismatic 8 object_curve
bash scripts/run_libero_suite_smoke.sh libero_goal    libero_goal_no_noops    0,1,2,3,4,5,6,7 models/openvla-7b-modelscope-prismatic 8 100bc
bash scripts/run_libero_suite_smoke.sh libero_10      libero_10_no_noops      0,1,2,3,4,5,6,7 models/openvla-7b-modelscope-prismatic 8 100bc
```

## 5. 正式论文复现

### 5.1 One policy per suite

#### 单 seed 快速启动

```bash
mkdir -p logs/paper_reproduction
nohup env \
  BASE_MODEL="$PWD/models/openvla-7b-modelscope-prismatic" \
  DATA_ROOT="$PWD/data/modified_libero_rlds" \
  PAPER_SEEDS="0" \
  PAPER_SUITE_FILTER="libero_spatial" \
  EVAL_SHARDS=8 \
  bash scripts/run_paper_libero_per_suite.sh \
  >>logs/paper_reproduction/launcher.log 2>&1 </dev/null &
```

去掉 `PAPER_SUITE_FILTER` 会依次训练四套。也可以将它设为由空格分隔的 suite 列表。

#### 论文级 3 seeds

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

正式配置：

| 项目 | 值 |
|---|---:|
| BC steps | 100,000 |
| Global BC batch | 64 |
| Latent warmup steps | 50,000 |
| PPO environment steps | 1,200,000 |
| PPO effective batch | 512 |
| PPO minibatch | 64 |
| PPO epochs | 4 |
| Policy LR | 3e-5 |
| Critic LR | 1e-4 |
| PPO clip | 0.2 |
| GAE lambda | 0.95 |
| Entropy coefficient | 0.01 |
| Smoothness coefficient | 0.1 |
| Gradient clipping | 1.0 |
| Exit calibration steps | 10,000 |
| Exit lookahead | 3 |
| Exit delta | 0.05 |

流水线顺序为：

```text
spatial train -> spatial 500-episode eval
object train  -> object 500-episode eval
goal train    -> goal 500-episode eval
long train    -> long 500-episode eval
```

每个 suite 完成后才会进入下一个；评测按 8 个 GPU shards 并行执行并严格合并。

四套和全部 seeds 完成后，launcher 会调用 `scripts/aggregate_table1_ours.py` 生成 per-suite Ours 汇总。

### 5.2 One policy for all 4 suites

all-four 使用四数据集 RLDS mixture `libero_4_task_suites_no_noops`。BC loader 延续 OpenVLA RLDS pipeline 的 dataset-size-balanced sampling；在线 PPO 在每个 DDP rank 上等量分配 Spatial/Object/Goal/Long 环境，并按 observation 选择对应的 proprio/action normalization statistics。

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

对每个 seed，脚本只训练一个 checkpoint，然后用 `scripts/evaluate_libero_checkpoint_all_suites.py` 在四套分别执行 500 episodes。聚合器会校验同一 seed 的四套结果确实来自同一个 checkpoint，避免把四个 per-suite checkpoint 误写成 all-four 结果。

8 卡截断链路测试入口：

```bash
bash scripts/run_table1_all_suites_short.sh
```

该命令执行 200 BC、50 warmup、4096 个真实环境 steps 和 20 个 exit calibration steps，用于验证完整执行链路，不用于报告论文成功率。

### 5.3 Table 1 自动汇总

如果已有原始 evaluation JSON，也可以单独聚合：

```bash
./.venv/bin/python scripts/aggregate_table1_ours.py \
  --result-root results/paper_all_suites \
  --mode all_policy \
  --seeds 0 1 2 \
  --output results/paper_all_suites/table1_ours_all_policy.json

./.venv/bin/python scripts/aggregate_table1_ours.py \
  --result-root results/paper_per_suite \
  --mode per_suite \
  --seeds 0 1 2 \
  --output results/paper_per_suite/table1_ours_per_suite.json
```

## 6. CALVIN ABC→D（Table 3）

原生 CALVIN adapter 直接流式读取官方 `episode_XXXXXXX.npz`，不需要先转换为 RLDS。每个样本包含 9 帧 observation history、8 个未来相对动作、双相机 RGB、8 维 AVA-VLA proprio 和 language instruction。训练 split 的 q01/q99 statistics 会在第一次运行时计算并缓存为 `avavla_calvin_statistics.json`。

正式训练和评测：

```bash
CALVIN_DATA_ROOT="$PWD/calvin_data/task_ABC_D" \
  SEED=0 EVAL_SHARDS=8 \
  bash scripts/run_paper_calvin_abc.sh
```

该 launcher 使用与 LIBERO 相同的四阶段预算。Stage 3 在官方 simulator 中按 task oracle 产生真实 success reward；训练结束后执行 1,000 条官方五任务序列，并由 `scripts/merge_calvin_eval_shards.py` 输出 SR@1..5 和 average sequence length。

只验证官方 sequence generator 而不加载模型或 simulator：

```bash
PYTHONPATH="$PWD/third_party/calvin_runtime:$PWD/third_party/calvin/calvin_models:$PWD/third_party/calvin/calvin_env:$PWD" \
  ./.venv/bin/python experiments/robot/calvin/run_calvin_eval.py \
  --protocol-only --num-sequences 10 \
  --output results/calvin_protocol_check.json
```

## 7. Early-exit threshold sweep（Table 5）

Table 5 使用训练完成的 all-four checkpoint，不为每个阈值重新训练。默认严格扫描论文中的阈值：

```text
0.30, 0.40, 0.50, 0.55, 0.65, 0.75, 0.85, 0.95, 1.00
```

```bash
./.venv/bin/python scripts/run_table5_threshold_sweep.py \
  --checkpoint runs/paper_all_suites/paper_all_suites_seed0 \
  --output-root results/table5 \
  --shards 8 \
  --num-trials-per-task 50
```

每个 policy query 都在 CUDA synchronize 后记录 latency 和实际 reasoning steps。分片合并器从原始 telemetry 重算 mean/P90 latency，最终生成 `table5_results.json` 和 CSV。不同 GPU 型号上的 latency 不应直接与论文 A100 数值比较。

## 监控

```bash
tail -f logs/paper_reproduction/status.log
tail -f logs/paper_reproduction/libero_spatial.seed0.train.log
watch -n 2 nvidia-smi
```

检查进程：

```bash
pgrep -af 'run_paper_libero_per_suite|finetune_avavla|run_libero_eval'
```

结构化训练日志：

```text
runs/paper_per_suite/paper_<suite>_seed<seed>/metrics.jsonl
```

每行至少包含：

- `global_step`
- `stage`
- `metrics` 中的全部数值 loss/diagnostics
- `grad_norm`
- 时间戳

## Loss 曲线

手动生成某个 run 的所有 loss 曲线：

```bash
./.venv/bin/python scripts/plot_avavla_losses.py \
  runs/paper_per_suite/paper_spatial_seed0/metrics.jsonl \
  --output-dir results/paper_per_suite/libero_spatial/seed0/training_curves
```

生成：

```text
loss_curves.png
loss_curves.csv
loss_curves_summary.json
```

正式流水线会在每个 suite 训练完成后自动生成这些文件。

## Checkpoint 与断点续训

Checkpoint manifest：

```text
CHECKPOINT_COMPLETE.json
```

只有 manifest 中的全部文件存在且字节大小匹配时，checkpoint 才可恢复。BC 每 1,000 steps 保存一次；PPO 每 10 次 update 保存一次，并在环境预算边界强制保存；exit calibration 每 1,000 steps 保存一次。

手动恢复：

```bash
./.venv/bin/torchrun --standalone --nproc-per-node=8 \
  vla-scripts/finetune_avavla.py \
  --vla_path runs/paper_per_suite/paper_spatial_seed0 \
  --resume true \
  ...其余参数保持与原 run 一致
```

推荐使用正式 launcher；它会自动验证当前 manifest/schema、恢复兼容 checkpoint，并归档 PPO 或推理语义不兼容的旧 run。不要通过手工复制部分 `.pt` 文件绕过原子 manifest。

## 评测结果

每个 suite/seed 的最终结果：

```text
results/paper_per_suite/<suite>/seed<seed>/evaluation_results.json
```

四套全部完成后：

```text
results/paper_per_suite/ALL_COMPLETE
results/paper_per_suite/table1_ours_per_suite.json
results/paper_all_suites/table1_ours_all_policy.json
```

评测固定使用：

- `seed=7`
- 10 tasks
- 每 task 50 trials
- 总计 500 episodes
- center crop 开启
- proprio normalization 开启
- history state 开启
- open-loop action chunk = 8

## 主要代码

```text
prismatic/models/vlas/avavla.py          AVA-VLA 模型、latent PPO、GAE、exit gate
vla_scripts/online_policy.py             跨 benchmark 在线 rollout、per-row normalization 与动作采样
vla-scripts/finetune_avavla.py           四阶段训练、checkpoint、结构化日志
vla-scripts/deploy_avavla.py             与训练一致的确定性推理
experiments/robot/libero/online_rollout.py LIBERO 单套/四套真实环境 collector
experiments/robot/calvin/dataset.py       官方 CALVIN frame/language 数据适配器
experiments/robot/calvin/online_rollout.py CALVIN simulator 与 task-oracle collector
experiments/robot/calvin/run_calvin_eval.py 官方 1000×5 连续任务评测
scripts/run_libero_suite_smoke.sh         单卡/多卡 smoke
scripts/run_official_base_preflight.sh    8 卡最短全阶段预检
scripts/run_paper_libero_per_suite.sh     四 suite 正式训练与 500 回合评测
scripts/run_paper_libero_all_suites.sh    同一策略联合四套训练与评测
scripts/aggregate_table1_ours.py           Table 1 两条 Ours 自动聚合
scripts/run_paper_calvin_abc.sh            Table 3 训练与分片评测
scripts/run_table5_threshold_sweep.py      Table 5 同 checkpoint 阈值扫描
scripts/plot_avavla_losses.py             Loss 曲线与 CSV 导出
```

环境和基准专项说明仍见 [SETUP.md](SETUP.md)、[LIBERO.md](LIBERO.md) 与 [ALOHA.md](ALOHA.md)。

## 引用

如果本项目对你的工作有帮助，请引用论文及 OpenVLA/OpenVLA-OFT 的相应工作。仓库引用信息见 [CITATION.cff](CITATION.cff)。
