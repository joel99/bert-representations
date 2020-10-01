#!/usr/bin/env python3

import os
import os.path as osp
import shutil
from typing import Union, List
import argparse

from transformers import (
    set_seed,
)

from src import (
    get_train_func,
    logger
)

from src.config.default import get_config


# ! TODO test
# ! Address warnings (save optimizer)

DO_PRESERVE_RUNS = False # Whether to fail if runs exist

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"], # ! Eval not supported
        required=True,
        help="run type of the experiment (train or eval)",
    )

    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="full path to a ckpt (for eval or resumption)"
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_exp(**vars(args))

def check_exists(path, preserve=DO_PRESERVE_RUNS):
    if osp.exists(path):
        logger.warn(f"{path} exists")
        if not preserve:
            logger.warn(f"removing {path}")
            shutil.rmtree(path, ignore_errors=True)
        return True
    return False

def prepare_config(exp_config: Union[List[str], str], run_type: str, ckpt_path="", opts=None) -> None:
    r"""Prepare config node / do some preprocessing

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        ckpt_path: If training, ckpt to resume. If evaluating, ckpt to evaluate.
        opts: list of strings of additional config options.

    Returns:
        Runner, config, ckpt_path
    """
    config = get_config(exp_config, opts)

    # Default behavior is to pull experiment name from config file
    # Bind variant name to directories
    if isinstance(exp_config, str):
        variant_config = exp_config
    else:
        variant_config = exp_config[-1]
    variant_name = osp.split(variant_config)[1].split('.')[0]
    config.defrost()
    config.VARIANT = variant_name
    config.TENSORBOARD_DIR = osp.join(config.TENSORBOARD_DIR, config.VARIANT)
    config.MODEL_DIR = osp.join(config.MODEL_DIR, config.VARIANT)
    config.LOG_DIR = osp.join(config.LOG_DIR, config.VARIANT)
    config.freeze()
    os.makedirs(config.LOG_DIR, exist_ok=True)

    if ckpt_path is not None:
        if not osp.exists(ckpt_path):
            ckpt_path = osp.join(config.MODEL_DIR, ckpt_path)

    return config, ckpt_path

def run_exp(exp_config: Union[List[str], str], run_type: str, ckpt_path="", opts=None) -> None:
    config, ckpt_path = prepare_config(exp_config, run_type, ckpt_path, opts)

    logfile_path = osp.join(config.LOG_DIR, f"{config.VARIANT}.log")
    logger.add_filehandler(logfile_path)

    set_seed(config.SEED)

    if run_type == "train":
        if ckpt_path is not None:
            train_func = get_train_func(config, checkpoint_path=ckpt_path)
        else:
            if DO_PRESERVE_RUNS:
                if check_exists(config.TENSORBOARD_DIR) or \
                    check_exists(config.MODEL_DIR) or \
                    check_exists(config.LOG_DIR):
                    exit(1)
            else:
                check_exists(config.TENSORBOARD_DIR)
                check_exists(config.MODEL_DIR)
                check_exists(config.LOG_DIR)
            train_func = get_train_func(config)
        train_func()
    elif run_type == "eval":
        assert False, "not implemented"

if __name__ == "__main__":
    main()
