# Src: https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb

import numpy as np
import torch
import torch.nn as nn
import json
import dataclasses
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from yacs.config import CfgNode as CN

import transformers
from transformers.data.data_collator import DataCollator, InputDataClass, default_data_collator
import datasets as nlp

from src.registry import get_model_type, get_config, load_features_dict
from src.utils import ModelArguments

class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_args, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_args.model_name_or_path,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, "bert")
            else:
                setattr(model, "bert", shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

def create_multitask_model(model_args, config: CN):
    # if checkpoint_path is not None:
    #     # Since there is no diff b/n loading TokenClassifiers and SequenceClassifiers, we can just call BertPreTrainedModel
    #     return BertPretrainedModel.from_pretrained(checkpoint_path) # fingers crossed

    model_types = {}
    model_configs = {}
    for task in config.TASK.TASKS:
        model_types[task] = get_model_type(task, config)
        model_configs[task] = get_config(task, config, model_args)[0]
    return MultitaskModel.create(
        model_args=model_args,
        model_type_dict=model_types,
        model_config_dict=model_configs,
    )

def NLPDataCollator(features: List[Union[InputDataClass, Dict]]) -> Dict[str, torch.Tensor]:
    """
    Extending the existing DataCollator to work with NLP dataset batches
    """
    first = features[0]
    if isinstance(first, dict):
        # NLP data sets current works presents features as lists of dictionary
        # (one per example), so we  will adapt the collate_batch logic for that
        if "labels" in first and first["labels"] is not None:
            if first["labels"].dtype == torch.int64:
                labels = torch.tensor([f["labels"] for f in features], dtype=torch.long)
            else:
                labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
            batch = {"labels": labels}
        for k, v in first.items():
            if k != "labels" and v is not None and not isinstance(v, str):
                batch[k] = torch.stack([f[k] for f in features])
        return batch
    else:
        # otherwise, revert to using the default collate_batch
        return default_data_collator(features)

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
    def __init__(self, dataloader_dict):
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

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader)
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])

class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
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
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=self.data_collator,
            ),
        )

        return data_loader

    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each
        task Dataloader
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })


def run_multitask(cfg, model_args, training_args, tokenizer, mode="train", *args, **kwargs):
    multitask_model = create_multitask_model(model_args, cfg)
    features_dict = load_features_dict(tokenizer, cfg)
    train_dataset = {
        task_name: dataset["train"]
        for task_name, dataset in features_dict.items()
    }
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=training_args,
        data_collator=NLPDataCollator,
        train_dataset=train_dataset
    )
    if mode == "train":
        trainer.train()
    if mode == "eval":
        # *nb It'd be tough to use compute_metrics, as it expects individual EvalPredictions and we'd need to aggregate appropriately. We just extract
        # Print individual evaluations
        preds_dict = {}
        for task_name in cfg.TASK.TASKS:
            split_key = "validation"
            if task_name == "mnli":
                split_key = "validation_matched"
            eval_dataloader = DataLoaderWithTaskname(
                task_name,
                trainer.get_eval_dataloader(eval_dataset=features_dict[task_name][split_key])
            )
            preds_dict[task_name] = trainer.prediction_loop( # ! careful, I changed the method
                eval_dataloader,
                description=f"Validation: {task_name}",
            )
        print(preds_dict)
        json.dump(preds_dict, cfg.EVAL.SAVE_FN.format(model_args.model_name_or_path))
