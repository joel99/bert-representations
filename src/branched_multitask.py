# Src: https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb
# from operator import attrgetter
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from yacs.config import CfgNode as CN

from itertools import cycle

import transformers
from transformers.data.data_collator import DataCollator, InputDataClass, DataCollatorWithPadding # , DataCollatorForTokenClassification
import datasets as nlp

from src.registry import get_model_type, get_config, load_features_dict
from src.utils import (
    ModelArguments,
    get_eval_metrics_func,
    TASK_KEY_TO_NAME,
    DataCollatorForTokenClassification,
    FixedTrainer,
    rsetattr,
    rgetattr
)
from src.multitask import TaskDependentCollator, DataLoaderWithTaskname, MultitaskDataloader, MultitaskTrainer, MultitaskModel

BERT_LAYERS = 12
class BranchedMultitaskModel(MultitaskModel):
    # TODO branches
    # TODO test trunk out
    @classmethod
    def create(cls, model_args, model_type_dict, model_config_dict, config):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same trunk.

        Note, modules weights are as follows:
        - encoder.embeddings (BERT embedding layer)
        - encoder.encoder (BERT encoder core)
            - encoder.encoder.layer.{0..11} (layer weights)
        - encoder.pooler (BERT pooling?) (unsure)
        And cloned weights
        - taskmodels_dict.<task>.bert.embeddings
        - taskmodels_dict.<task>.bert.encoder
        - taskmodels_dict.<task>.bert.pooler
        And task-specific weights
        - taskmodels_dict.<task>.classifier (exactly this for all tasks we consider)
        """
        taskmodels_dict = {}
        is_ckpt = osp.exists(model_args.model_name_or_path)
        if is_ckpt:
            # Loading an existing model -- be careful. Following PretrainedModel loading scheme.
            # Note that we save redundant weights (just due to this architecture's setup)
            # We'll overwrite the central encoder repeatedly, as we
            total_state_dict = torch.load(osp.join(model_args.model_name_or_path, "pytorch_model.bin"), map_location="cpu")

        trunk_modules = None
        trunk_module_names = ["bert.embeddings", "bert.pooler"]
        trunk_layers = range(BERT_LAYERS - config.MODEL.HEAD_FIRST_LAYERS)
        trunk_module_names.extend([f"bert.encoder.layer.{i}" for i in trunk_layers])
        branch_layers = range(BERT_LAYERS - config.MODEL.HEAD_FIRST_LAYERS, BERT_LAYERS)
        branch_module_names = [f"bert.encoder.layer.{i}" for i in branch_layers] # Classifiers are still independent
        branches = config.MODEL.HEAD_BRANCHES
        branch_modules = [None] * len(branches)
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                config.MODEL.BASE,
                config=model_config_dict[task_name],
                state_dict=cls._extract_task_weights(total_state_dict, task_name) if is_ckpt else None
            )

            if trunk_modules is None:
                trunk_modules = [rgetattr(model, name) for name in trunk_module_names]
            else:
                for name, module in zip(trunk_module_names, trunk_modules):
                    rsetattr(model, name, module)
            if len(branches) > 0:
                group_index = list(map(lambda b: task_name in b, branches)).index(True)
                if branch_modules[group_index] is None:
                    branch_modules[group_index] = [rgetattr(model, name) for name in branch_module_names]
                else:
                    for name, module in zip(branch_module_names, branch_modules[group_index]):
                        rsetattr(model, name, module)
            taskmodels_dict[task_name] = model
        return cls(taskmodels_dict=taskmodels_dict)