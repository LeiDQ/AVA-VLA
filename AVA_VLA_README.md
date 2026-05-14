# AVA-VLA: Adaptive Variable Alignment Vision-Language-Action Model

这是 AVA-VLA（Adaptive Variable Alignment VLA）模型的官方实现。

## 核心特性

### 1. 隐式推理（Latent Reasoning）
- 将推理过程建模为连续的潜在状态演化，避免显式的CoT文本生成
- 通过潜在变量序列 $z_t$ 表示中间推理状态
- 不需要文本监督，端到端学习

### 2. 基于强化学习的去噪（RL-based Denoising）
- 将潜在推理生成建模为序列决策过程（POMDP）
- 使用任务级奖励信号优化推理轨迹
- 包含熵惩罚（抑制过度随机扰动）和平滑性正则化（保持时序连续性）

### 3. 自适应早期退出（Early-Exit Strategy）
- 通过退出门机制 $g_\omega$ 评估状态置信度
- 当置信度超过阈值时自适应终止推理
- 实现推理深度与计算效率的动态平衡

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│                     Multimodal Input                    │
│  ┌──────────────┐  ┌──────────────┐                │
│  │   Visual     │  │  Language    │                │
│  │   v_t        │  │  l_t         │                │
│  └──────┬───────┘  └──────┬───────┘                │
│         │                 │                           │
│         └────────┬────────┘                           │
│                  ▼                                    │
│         ┌─────────────────┐                           │
│         │  Multimodal    │                           │
│         │  Encoder ψ(·)   │                           │
│         └────────┬────────┘                           │
│                  ▼                                    │
│         ┌─────────────────┐                           │
│         │ Initial Latent │                           │
│         │    State z_0    │                           │
│         └────────┬────────┘                           │
│                  │                                    │
│                  ▼                                    │
│  ┌──────────────────────────────────────────┐            │
│  │   Latent Reasoning Loop (POMDP)      │            │
│  │  ┌─────────────────────────────────┐  │            │
│  │  │ Reasoning Policy π_φ          │  │            │
│  │  │ Generate update action u_t    │  │            │
│  │  └─────────────┬───────────────┘  │            │
│  │                ▼                    │            │
│  │  ┌─────────────────────────────────┐  │            │
│  │  │ Transition f_θ               │  │            │
│  │  │ Update latent: z_{t+1}      │  │            │
│  │  └─────────────┬───────────────┘  │            │
│  │                │                    │            │
│  │                ▼                    │            │
│  │  ┌─────────────────────────────────┐  │            │
│  │  │ Exit Gate g_ω                │  │            │
│  │  │ Confidence score e_t         │  │            │
│  │  └─────────────┬───────────────┘  │            │
│  │                │                    │            │
│  │         e_t > τ? (Early Exit)     │            │
│  └─────────────────┼──────────────────┘            │
│                    │                                    │
│                    ▼                                    │
│         ┌─────────────────┐                           │
│         │ Action Policy  │                           │
│         │   π_ψ          │                           │
│         │ Generate a_t   │                           │
│         └─────────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

## 目录结构

```
AVA-VLA/
├── prismatic/
│   └── models/
│       └── vlas/
│           ├── __init__.py              # 模块导出
│           ├── openvla.py              # 基础OpenVLA模型
│           └── avavla.py              # AVA-VLA核心实现
├── vla-scripts/
│   ├── finetune_avavla.py           # AVA-VLA训练脚本
│   └── deploy_avavla.py             # AVA-VLA推理部署脚本
├── scripts/
│   └── evaluate_avavla.py           # 评估脚本
└── AVA_VLA_README.md               # 本文档
```

## 安装依赖

```bash
# 基础依赖
pip install torch torchvision transformers
pip install pillow numpy tqdm wandb
pip install accelerate

# 可选：用于分布式训练
pip install draccus
```

## 快速开始

### 1. 训练AVA-VLA模型

```bash
python vla-scripts/finetune_avavla.py \
    --vla_path /path/to/prismatic-openvla-run \
    --data_root_dir datasets/rlds \
    --dataset_name your_dataset_name \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 200000 \
    --enable_latent_reasoning True \
    --use_rl_denoising True \
    --max_reasoning_steps 5 \
    --history_window_size 2 \
    --exit_threshold 0.8 \
    --wandb_entity your-entity \
    --wandb_project avavla-project
```

**主要训练参数：**

- `--enable_latent_reasoning`: 启用潜在推理机制
- `--use_rl_denoising`: 启用RL去噪
- `--max_reasoning_steps`: 最大推理步数（默认5）
- `--exit_threshold`: 早期退出阈值（默认0.8）
- `--latent_dim`: 潜在状态维度（默认512）
- `--rl_lr`: RL组件学习率（默认1e-4）
- `--reasoning_policy_type`: Softmax 更新模式策略（默认`softmax`）；连续高斯策略可用`gaussian`
- `--ppo_clip_ratio`: PPO裁剪系数（默认0.2）
- `--gae_lambda`: GAE参数（默认0.95）
- `--ppo_epochs`: 每个冻结rollout上的PPO更新轮数（默认4）
- `--ppo_minibatch_size`: PPO minibatch大小（默认64）
- `--history_window_size`: RLDS窗口长度，用于构造历史状态 `h_{t-1}`（默认2）
- `--reward_error_scale`: 当数据集没有环境reward/success字段时，将动作L1误差映射为离线proxy reward的尺度（默认0.25）
- `--trajectory_reward_weight`: 轨迹一致性奖励权重，约束预测动作序列的相邻变化与专家轨迹一致（默认0.25）
- `--vla_path`: 当前训练脚本需要本地Prismatic/OpenVLA格式checkpoint目录或`.pt`文件

### 1.1 只用10个样例训练/评测

```bash
RUN_DIR="${RUN_ROOT:-./runs}/avavla_libero_10sample/$(date +%Y%m%d-%H%M%S)"
python scripts/train_avavla_libero_10sample.py \
    --train-samples 10 \
    --eval-samples 10 \
    --bc-steps 10 \
    --latent-warmup-steps 5 \
    --ppo-steps 10 \
    --ppo-epochs 4 \
    --ppo-minibatch-size 4 \
    --exit-calibration-steps 5 \
    --exit-calibration-target-rate 0.5 \
    --batch-size 4 \
    --output-dir "$RUN_DIR"
```

这个脚本默认只消费10个LIBERO step样例，并执行BC预训练、latent warmup、基于冻结rollout的裁剪PPO联合RL微调、Exit Gate校准和自适应阈值标定。首次运行会保存`$RUN_DIR/samples_10.json`，后续可以通过`--samples-json "$RUN_DIR/samples_10.json"`只依赖这个小样例文件复跑。

### 2. 部署和推理

```bash
python vla-scripts/deploy_avavla.py \
    --checkpoint path/to/checkpoint \
    --image path/to/image.jpg \
    --instruction "Pick up the red block" \
    --enable-latent-reasoning \
    --max-reasoning-steps 5 \
    --exit-threshold 0.8
```

**输出示例：**
```
Loading AVA-VLA model from path/to/checkpoint
AVA-VLA model loaded successfully on cuda
  - Latent reasoning: True
  - Max reasoning steps: 5
  - Exit threshold: 0.8

Predicting action for instruction: Pick up the red block

=== Results ===
Predicted actions: [0.01 -0.02 0.05 ...]
Reasoning steps performed: 3
Exit threshold: 0.8
Exit scores history: [0.45 0.62 0.85]
```

### 3. 评估模型

```bash
python scripts/evaluate_avavla.py \
    --benchmark json \
    --avavla-checkpoint path/to/checkpoint \
    --dataset path/to/eval_dataset.json \
    --unnorm-key your_dataset \
    --num-samples 100 \
    --output results.json
```

真实LIBERO rollout评估：

```bash
python scripts/evaluate_avavla.py \
    --benchmark libero \
    --avavla-checkpoint path/to/checkpoint \
    --task-suite libero_spatial \
    --num-trials-per-task 50
```

CALVIN / LIBERO+ 不包含在本仓库环境依赖中，需要提供对应benchmark的外部评估脚本：

```bash
python scripts/evaluate_avavla.py \
    --benchmark calvin \
    --avavla-checkpoint path/to/checkpoint \
    --external-eval-script /path/to/calvin_eval.py \
    --task-suite abc_to_d
```

**评估指标：**
- 准确性：平均误差、中位数误差
- 效率：平均延迟、P90延迟、吞吐量
- 推理统计：平均推理步数、早期退出率

## 核心组件详解

### ReasoningPolicy (推理策略 π_φ)

```python
class ReasoningPolicy(nn.Module):
    """生成潜在更新动作 u_t"""
    
    def __init__(
        self, 
        latent_dim: int = 512,
        obs_dim: int = 768,
        hidden_dim: int = 1024,
        num_heads: int = 8
    ):
        # 使用Transformer编码器
        # 输出Softmax更新模式分布
```

### LatentTransition (状态转换 f_θ)

```python
class LatentTransition(nn.Module):
    """更新潜在状态 z_t → z_{t+1}"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        obs_dim: int = 768,
        hidden_dim: int = 1024
    ):
        # 使用Transformer上下文建模 + 更新动作门控机制
        # 实现平滑的状态演化
```

### ExitGate (退出门 g_ω)

```python
class ExitGate(nn.Module):
    """评估状态置信度，决定是否提前退出"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 256
    ):
        # 轻量级MLP
        # 输出(0,1)范围的置信度分数
```

### ValueFunction (值函数 V^π)

```python
class ValueFunction(nn.Module):
    """估计状态价值，用于RL优化"""
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512
    ):
        # 双层MLP
        # 输出状态价值估计
```

## RL去噪损失函数

总损失由以下部分组成：

1. **策略损失 (Policy Loss)**:
   ```
   L_policy = -E[log π_φ(u_t|z_t, o_t) * A_t]
   ```

2. **值函数损失 (Value Loss)**:
   ```
   L_value = MSE(V^π(z_t), R_t)
   ```

3. **熵惩罚 (Entropy Penalty)**:
   ```
   r_t = r_task(a_t) - λ_1 H[π_φ(·|z_t, o_t)] - λ_2 ||z_{t+1} - z_t||^2
   ```

4. **平滑性惩罚 (Smoothness Penalty)**:
   ```
   r_t = r_task(a_t) - λ_1 H[π_φ(·|z_t, o_t)] - λ_2 ||z_{t+1} - z_t||^2
   ```

5. **退出门监督 (Exit Gate Loss)**:
   ```
   L_exit = BCE(g_ω(z_t), clamp(r_task(a_t), 0, 1))
   ```

总损失：
```
L_total = L_action + L_policy + value_coef * L_value + exit_loss_coef * L_exit
```

## 实验配置

### LIBERO基准测试

```bash
python vla-scripts/finetune_avavla.py \
    --vla_path /path/to/prismatic-openvla-run \
    --data_root_dir datasets/rlds \
    --dataset_name libero_spatial \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 200000 \
    --latent_dim 512 \
    --max_reasoning_steps 5 \
    --history_window_size 2 \
    --exit_threshold 0.8
```

### CALVIN基准测试

本仓库没有内置CALVIN环境。可使用相同训练脚本训练CALVIN格式RLDS数据；完整成功率评估请通过 `scripts/evaluate_avavla.py --benchmark calvin --external-eval-script ...` 调用外部CALVIN evaluator。

```bash
python vla-scripts/finetune_avavla.py \
    --vla_path /path/to/prismatic-openvla-run \
    --data_root_dir datasets/rlds \
    --dataset_name calvin_abc_to_d \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --max_steps 200000 \
    --latent_dim 512 \
    --max_reasoning_steps 5 \
    --exit_threshold 0.8
```

## 消融实验

可以通过禁用不同组件进行消融研究：

```bash
# 不使用潜在推理
python vla-scripts/finetune_avavla.py \
    --enable_latent_reasoning False

# 不使用RL去噪
python vla-scripts/finetune_avavla.py \
    --use_rl_denoising False

# 固定推理步数（不使用早期退出）
python vla-scripts/deploy_avavla.py \
    --fixed-steps 5
```

## 性能优化建议

### 1. 推理阶段
- 使用自适应早期退出可以减少约54%的平均推理步数
- 对于简单任务，推理深度通常为1-2步
- 对于复杂任务，推理深度可达5步

### 2. 训练阶段
- AVA-VLA训练脚本直接优化基础VLA、AVA推理模块和L1动作头，并保存PPO/GAE配置到`avavla_config.json`
- 10样例脚本按BC、latent warmup、PPO joint RL、Exit Gate calibration四阶段执行
- PPO默认使用policy学习率3e-5、critic学习率1e-4、clip ratio 0.2、GAE λ=0.95
- 建议使用梯度累积来增加有效batch size

### 3. 内存优化
- 使用混合精度训练（bfloat16）
- 对于大batch size，启用梯度检查点
- 在推理时使用`torch.no_grad()`减少内存占用

## 常见问题

### Q: 如何调整推理深度与性能的权衡？

A: 主要通过两个参数：
1. `--max_reasoning_steps`: 设置最大推理步数
2. `--exit_threshold`: 调整退出阈值
   - 更高的阈值（如0.9）→ 更多推理步数 → 更好性能
   - 更低的阈值（如0.6）→ 更少推理步数 → 更快速度

### Q: RL训练需要环境交互吗？

A: 不强制需要。训练会优先使用RLDS里的 `reward` / `success` 字段作为任务级奖励，并叠加轨迹一致性奖励；如果离线数据没有环境反馈字段，才会降级使用基于动作误差的proxy reward。真实环境成功率评估通过LIBERO rollout入口完成。

### Q: 如何在现有VLA基础上添加AVA-VLA？

A: AVA-VLA继承自OpenVLA，可以直接：
1. 加载预训练的OpenVLA权重
2. 添加潜在推理组件
3. 进行微调训练

### Q: 潜在推理会增加多少参数？

A: 对于默认配置：
- ReasoningPolicy: ~2M 参数
- LatentTransition: ~4M 参数
- ExitGate: ~0.3M 参数
- ValueFunction: ~0.8M 参数
- 总计: ~7M 参数（相比7B VLA基模型，增加<0.1%）

## 引用

如果您使用此代码，请引用：

```bibtex
@inproceedings{avavla2026,
  title={Think Less, Act Early: Reinforced Latent Reasoning with Early Exit in VLA Models},
  author={Lei, Dianqiao and Shan, Lianlei},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## 许可证

本项目遵循原OpenVLA的许可证。

## 贡献

欢迎提交问题和拉取请求！

## 联系方式

如有问题，请通过GitHub Issues联系。
