#!/usr/bin/env python3
# Author: Joel Ye

from typing import List, Optional, Union

from yacs.config import CfgNode as CN

DEFAULT_CONFIG_DIR = "config/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100

# Name of experiment
_C.VARIANT = "experiment"
_C.TENSORBOARD_DIR = "tb/"
_C.MODEL_DIR = "models/"
_C.LOG_DIR = "logs/"

# -----------------------------------------------------------------------------
# System
# -----------------------------------------------------------------------------
_C.SYSTEM = CN()
# ! Unsupported
_C.SYSTEM.TORCH_GPU_ID = 0
# Auto-assign if you have free reign to GPUs. False if you're in a managed cluster that assigns GPUs.
_C.SYSTEM.GPU_AUTO_ASSIGN = False
_C.SYSTEM.NUM_GPUS = 1

# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
# Each dataset has its defines its own task, which requires the right dataloader as well as the right model.

_C.TASK = CN()
# Controls which downstream handler to use
_C.TASK.TASKS = ["mnli"]

_C.TASK.TASK_STRATEGY = "finetune"
# "finetune" - direct finetuning
# "tune_head" - fine tune + N additional layers
# TODO support additional layers ^ @ Joel
_C.TASK.HEAD_FIRST_EPOCHS = 0 # Number of layers to fine-tune head layers before unfreezing base [Ramasesh]
# TODO support head first epochs

# "features" - feature extraction
# TODO support feature extraction (RQ 4.3) @aysh
_C.TASK.FROZEN_LAYERS = -1 # Don't freeze any
# TODO support freezing @ aysh


# TODO implement nonsequential @ Joel
_C.TASK.MULTITASK_STRATEGY = "SEQUENTIAL"
# Most experiments can be done by training base -> A. (RQ 4.1, 4.2, 4.5).
# SAMPLED -- random task sampled, forward and back
# ROUND_ROBIN -- each task forward and back
# MIXED -- all tasks forward pass, one back pass
_C.TASK.MULTITASK_SAMPLER = "uniform"
# "uniform", "size" -- proportional to dataset size

_C.DATA = CN()
# `bert-representations` considers multiple tasks, and requires all datasets to share a common datapath. Mark it below.
# A given task's will receive a datapath pointing <DATAPATH>/<task_name> (symbolic links accepted)
_C.DATA.DATAPATH = 'data/'

# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.BASE = "bert-base-uncased" # distilbert-base-cased
_C.MODEL.MAX_LENGTH = 128
# -----------------------------------------------------------------------------
# Train Config
# -----------------------------------------------------------------------------
_C.TRAIN = CN()

_C.TRAIN.DO_VAL = True # Run validation while training
_C.TRAIN.BATCH_SIZE = 32
_C.TRAIN.TASK_LIMIT = 100000 # Not affected
_C.TRAIN.NUM_EPOCHS_PER_TASK = 1
_C.TRAIN.NUM_UPDATES_PER_TASK = -1 # Will override num_epochs_per_task if > 0
_C.TRAIN.CHECKPOINT_INTERVAL = 1000 # Num steps per checkpoint
_C.TRAIN.LOG_INTERVAL = 500
_C.TRAIN.LR_INIT = 2e-5 # Mosbach
_C.TRAIN.LR_WARMUP_STEPS = 0
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.EVAL_STEPS = 1000
_C.EVAL = CN()
_C.EVAL.BATCH_SIZE = 128
_C.EVAL.SAVE_FN = "{}.eval"
_C.EVAL.SPLIT = "validation" # There's no support for test! We'll just crash because there are no labels

def get_cfg_defaults():
  """Get default LFADS config (yacs config node)."""
  return _C.clone()

def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = get_cfg_defaults()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

