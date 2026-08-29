"""RLDS input pipeline with TensorFlow isolated from PyTorch training GPUs."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

try:
    # TensorFlow is used only for RLDS decoding/augmentation. Let PyTorch own
    # every accelerator and avoid TF PTX JIT/memory allocation on new GPUs.
    tf.config.set_visible_devices([], "GPU")
except RuntimeError as error:
    raise RuntimeError("TensorFlow initialized a GPU before RLDS isolation") from error

from .dataset import make_interleaved_dataset, make_single_dataset
