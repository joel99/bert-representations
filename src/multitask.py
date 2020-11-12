# Src: https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb
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
    FixedTrainer
)

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, taskmodels_dict, encoder=None):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def _extract_task_weights(cls, total_state_dict, task_name):
        r""" Get the weights to load a single task"""
        prefix_str = f"taskmodels_dict.{task_name}."
        task_weights = {
            k[len(prefix_str):]: v for
            k, v in total_state_dict.items() if k.startswith(prefix_str)
        }
        return task_weights

    @classmethod
    def create(cls, model_args, model_type_dict, model_config_dict, config):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        is_ckpt = osp.exists(model_args.model_name_or_path)
        if is_ckpt:
            # Loading an existing model -- be careful. Following PretrainedModel loading scheme.
            # Note that we save redundant weights (just due to this architecture's setup)
            total_state_dict = torch.load(osp.join(model_args.model_name_or_path, "pytorch_model.bin"), map_location="cpu")

        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                config.MODEL.BASE,
                config=model_config_dict[task_name],
                state_dict=cls._extract_task_weights(total_state_dict, task_name) if is_ckpt else None
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, "bert")
            else:
                setattr(model, "bert", shared_encoder)
            taskmodels_dict[task_name] = model

        # Freeze layers
        if config.MODEL.FROZEN_LAYERS > -1:
            for param in shared_encoder.embeddings.parameters():
                param.requires_grad = False
            for l in range(config.MODEL.FROZEN_LAYERS + 1):
                for param in shared_encoder.encoder.layer[l].parameters():
                    param.requires_grad = False
        return cls(taskmodels_dict=taskmodels_dict, encoder=shared_encoder)

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

def create_multitask_model(model_args, config: CN, model_cls: nn.Module):
    # if checkpoint_path is not None:
    #     # Since there is no diff b/n loading TokenClassifiers and SequenceClassifiers, we can just call BertPreTrainedModel
    #     return BertPretrainedModel.from_pretrained(checkpoint_path) # fingers crossed

    model_types = {}
    model_configs = {}
    for task in config.TASK.TASKS:
        model_types[task] = get_model_type(task, config)
        model_configs[task] = get_config(task, config)[0]
    return model_cls.create(
        model_args=model_args,
        model_type_dict=model_types,
        model_config_dict=model_configs,
        config=config
    )

def TaskDependentCollator(tokenizer):
    token_collator = DataCollatorForTokenClassification(tokenizer)
    default_collator = DataCollatorWithPadding(tokenizer)
    def inner(task_key: str, features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
        if task_key == "pos":
            return token_collator(features)
        else:
            return default_collator(features)
    return inner

class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch

class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, config, dataloader_dict):
        self.config = config
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset)
            for dataloader in self.dataloader_dict.values()
        )
        self.epoch_factor = 1
        if self.config.TASK.MULTITASK_STRATEGY == "FULL_SEQUENTIAL":
            # Cram all epoch's updates into one list
            self.epoch_factor = self.config.TRAIN.NUM_EPOCHS_PER_TASK

    def __len__(self):
        if "EQUAL" in self.config.TASK.MULTITASK_STRATEGY:
            return len(self.task_name_list) * self.config.TRAIN.NUM_UPDATES_PER_TASK
        return sum(self.num_batches_dict.values()) * self.epoch_factor

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        if "EQUAL" in self.config.TASK.MULTITASK_STRATEGY:
            for i, task_name in enumerate(self.task_name_list):
                task_choice_list += [i] * self.config.TRAIN.NUM_UPDATES_PER_TASK
        else:
            for i, task_name in enumerate(self.task_name_list):
                task_choice_list += [i] * self.num_batches_dict[task_name] * self.epoch_factor
            task_choice_list = np.array(task_choice_list)
        if "SAMPLE" in self.config.TASK.MULTITASK_STRATEGY:
            np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: cycle(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

class MultitaskTrainer(FixedTrainer):
    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

    def get_single_train_dataloader(self, task_key, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        data_loader = DataLoaderWithTaskname(
            task_name=task_key,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=lambda f: self.data_collator(task_key, f),
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader(self.config, {
            task_key: self.get_single_train_dataloader(task_key, task_dataset)
            for task_key, task_dataset in self.train_dataset.items()
        })


def run_multitask(cfg, multitask_model_cls, model_args, training_args, tokenizer, mode="train", extract=False, *args, **kwargs):
    multitask_model = create_multitask_model(model_args, cfg, multitask_model_cls)

    features_dict = load_features_dict(tokenizer, cfg)
    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in features_dict.items()
    }
    task_collator = TaskDependentCollator(tokenizer)
    trainer = MultitaskTrainer(
        config=cfg,
        model=multitask_model,
        args=training_args,
        data_collator=task_collator,
        train_dataset=train_dataset
    )
    if mode == "train":
        trainer.train()
    if mode == "eval":
        # *nb It'd be tough to use compute_metrics, as it expects individual EvalPredictions and we'd need to aggregate appropriately. We just extract
        # Print individual evaluations
        preds_dict = {}
        split_key = "validation"
        extract_path = None
        if extract:
            extract_path = get_extract_path(cfg, model_args)
        for task_key in cfg.TASK.TASKS:
            split_key = cfg.EVAL.SPLIT
            if task_key == "mnli":
                split_key = f"{split_key}_matched"
            eval_dataloader = DataLoaderWithTaskname(
                task_key,
                data_loader=DataLoader(
                    features_dict[task_key][split_key],
                    batch_size=cfg.EVAL.BATCH_SIZE,
                    collate_fn=lambda f: task_collator(task_key, f)
                )
            )
            preds_dict[task_key] = trainer.prediction_loop(
                eval_dataloader,
                description=f"Validation: {task_key}",
                extract_path=extract_path,
                limit_tokens=cfg.TASK.EXTRACT_TOKENS_LIMIT
            )
        predictions_file = osp.join('./eval/', cfg.EVAL.SAVE_FN.format(f"{cfg.VARIANT}_{osp.split(model_args.model_name_or_path)[1]}_{split_key}"))
        results = {}
        for task_key in cfg.TASK.TASKS:
            evaluator = get_eval_metrics_func(task_key)
            evaluation = evaluator(preds_dict[task_key])
            task_name = TASK_KEY_TO_NAME[task_key]
            results[task_name] = evaluation
            print(task_name, evaluation)
        torch.save(results, predictions_file)
