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
    get_runner_func,
    logger,
    TASK_DICT
)

from run import prepare_config, get_parser, indiv_run

# For each (trained on task A) model, tune on task B with N layers frozen

DO_PRESERVE_RUNS = False # Whether to fail if runs exist
ALL_EVAL_KEY = "all"
BERT_LAYERS = 12

def run_exp(exp_config: Union[List[str], str], run_type: str, ckpt_path="", run_id=None, eval_split=None, extract=False, opts=None) -> None:
    config, ckpt_path = prepare_config(exp_config, run_type, ckpt_path, run_id, eval_split, opts)

    assert not extract, "extract not supported for AuC runs"
    assert len(ckpt_path) == 0 or osp.exists(ckpt_path), "must provide valid ckpt path"

    logfile_path = osp.join(config.LOG_DIR, f"{config.VARIANT}.log")
    logger.add_filehandler(logfile_path)

    set_seed(config.SEED)
    for target_task in TASK_DICT:
        if len(config.TASK.TASKS) > 0 and target_task == config.TASK.TASKS[0]:
            continue
        for l in range(0, BERT_LAYERS): # TODO -1
            print(f"Starting AuC task {target_task} ... layer {l}")
            layer_cfg = config.clone()
            layer_cfg.defrost()
            layer_cfg.TASK.TASKS = [target_task]
            layer_cfg.MODEL.FROZEN_LAYERS = l
            layer_cfg.MODEL_DIR = osp.join(layer_cfg.MODEL_DIR, target_task, f"freeze_{l}")
            layer_cfg.TENSORBOARD_DIR = osp.join(layer_cfg.TENSORBOARD_DIR, target_task, f"freeze_{l}")
            layer_cfg.LOG_DIR = osp.join(layer_cfg.LOG_DIR, target_task, f"freeze_{l}")
            layer_cfg.EVAL.SAVE_FN = f"target_{target_task}-freeze_{l}-" + "{}.eval"
            layer_cfg.EVAL_ON_COMPLETION = True
            layer_cfg.TRAIN.TRANSFER_INIT = True
            if target_task != "mnli":
                layer_cfg.TRAIN.NUM_UPDATES_PER_TASK = 2000
            else:
                layer_cfg.TRAIN.NUM_UPDATES_PER_TASK = 6000
            layer_cfg.TRAIN.CHECKPOINT_INTERVAL = layer_cfg.TRAIN.NUM_UPDATES_PER_TASK
            layer_cfg.TRAIN.EVAL_STEPS = layer_cfg.TRAIN.NUM_UPDATES_PER_TASK
            layer_cfg.freeze()
            indiv_run(layer_cfg, run_type, ckpt_path=ckpt_path)

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_exp(**vars(args))

if __name__ == "__main__":
    main()
