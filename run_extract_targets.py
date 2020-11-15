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

# For each (trained on task A) model, extract representations on task B

ALL_EVAL_KEY = "all"
BERT_LAYERS = 12

def run_exp(exp_config: Union[List[str], str], run_type: str, ckpt_path="", run_id=None, eval_split=None, extract=False, opts=None) -> None:
    config, ckpt_path = prepare_config(exp_config, run_type, ckpt_path, run_id, eval_split, opts)
    assert run_type == "eval", "extract not supported for training"
    assert len(ckpt_path) == 0 or osp.exists(ckpt_path), "must provide valid ckpt path"

    logfile_path = osp.join(config.LOG_DIR, f"{config.VARIANT}.log")
    logger.add_filehandler(logfile_path)

    set_seed(config.SEED)
    for target_task in TASK_DICT:
        # if len(config.TASK.TASKS) > 0 and target_task == config.TASK.TASKS[0]:
        #     continue
        print(f"Starting extraction on task {target_task}")
        layer_cfg = config.clone()
        layer_cfg.defrost()
        layer_cfg.TASK.TASKS = [target_task]
        # Update model dir so extracted is dumped in the correct place
        layer_cfg.MODEL_DIR = osp.join(layer_cfg.MODEL_DIR, f"extract_{target_task}")
        layer_cfg.TENSORBOARD_DIR = osp.join(layer_cfg.TENSORBOARD_DIR, f"extract_{target_task}")
        layer_cfg.LOG_DIR = osp.join(layer_cfg.LOG_DIR, f"extract_{target_task}")
        # layer_cfg.EVAL.SAVE_FN = f"target_{target_task}_zero-shot" + "{}.eval"
        # layer_cfg.EVAL_ON_COMPLETION = False
        layer_cfg.TRAIN.TRANSFER_INIT = True
        layer_cfg.freeze()
        indiv_run(layer_cfg, run_type, ckpt_path=ckpt_path, extract=True)

def main():
    parser = get_parser()
    args = parser.parse_args()
    run_exp(**vars(args))

if __name__ == "__main__":
    main()
