# Setup Instructions

These instructions describe a generic local setup for the AVA-VLA source release. Adjust the PyTorch installation command for your CUDA driver, GPU, and package manager.

## Conda Environment

```bash
conda create -n avavla python=3.10 -y
conda activate avavla
```

Install PyTorch using the command recommended for your machine:

```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
```

Install AVA-VLA and its Python dependencies from the repository root:

```bash
pip install -e .
```

## Optional Training Dependencies

Flash Attention 2 can improve training throughput, but it is sensitive to the local CUDA and PyTorch versions. Install it after the editable install if your environment supports it:

```bash
pip install packaging ninja
ninja --version
pip install "flash-attn==2.5.5" --no-build-isolation
```

## Optional Benchmark Dependencies

For LIBERO evaluation:

```bash
git submodule update --init third_party/LIBERO
pip install -e third_party/LIBERO
pip install -r experiments/robot/libero/libero_requirements.txt
```

## Local Paths

Keep machine-specific data, checkpoint, and run paths outside the repository and pass them explicitly:

```bash
export DATA_ROOT=/path/to/datasets
export RUN_ROOT=/path/to/runs
export CHECKPOINT_DIR=/path/to/checkpoints
```
