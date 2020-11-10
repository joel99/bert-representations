# Src: https://colab.research.google.com/github/zphang/zphang.github.io/blob/master/files/notebooks/Multi_task_Training_with_Transformers_NLP.ipynb
#%%
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data.dataloader import DataLoader
from transformers.trainer import get_tpu_sampler
from transformers.data.data_collator import DataCollator, InputDataClass, default_data_collator
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict

import transformers
import datasets as nlp
from yacs.config import CfgNode as CN

import os
import os.path as osp

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)


from src.registry import get_model_type, get_config
from src.utils import ModelArguments


#%%
dataset_dict = {
    # Keyd by task key
    "stsb": nlp.load_dataset('glue', name="stsb", cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/glue_data/stsb"),
    "sst2": nlp.load_dataset('glue', name="sst2", cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/glue_data/sst2"),
    # "mnli": nlp.load_dataset('glue', name="mnli", cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/glue_data/MNLI"),
    # "pos": nlp.load_dataset('conll2003', cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/POS/"),
}
#%%

for task_name, dataset in dataset_dict.items():
    print(task_name)
    batch = dataset_dict[task_name]["validation"]
    print(batch)
    # print(tokenizer.batch_encode_plus(batch))
    print()

#%%
batch = dataset_dict['pos']['train'][:3]
print(tokenizer(batch['words']))
#%%
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
        # TODO hook this up with our checkpoints
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_args.model_name_or_path,
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)

def create_multitask_model(config: CN):
    # More piping of our config to model config
    # ! Support checkpointing?
    model_args = ModelArguments(
        model_name_or_path=config.MODEL.BASE,
    )

    # Models are initialized either with Transformers configs or paths
    model_types = {}
    model_configs = {}
    for task in config.TASK.TASKS:
        model_types[task] = get_model_type(task, config)
        model_configs[task] = get_config(task, config, model_args)
    return MultitaskModel.create(
        model_args=model_args,
        model_type_dict=model_types,
        model_config_dict=model_configs,
    )

#%%
from src.config.default import get_config as get_yacs_cfg
config = get_yacs_cfg('../configs/mnli_test.yaml', None)

multitask_model = create_multitask_model(config)
print(multitask_model.encoder.embeddings.word_embeddings.weight.data_ptr())
#%%
print(multitask_model.taskmodels_dict['mnli'].bert.embeddings.word_embeddings.weight.data_ptr())
# print(multitask_model.taskmodels_dict["stsb"].bert.embeddings.word_embeddings.weight.data_ptr())
# print(multitask_model.taskmodels_dict["rte"].bert.embeddings.word_embeddings.weight.data_ptr())
# print(multitask_model.taskmodels_dict["commonsense_qa"].bert.embeddings.word_embeddings.weight.data_ptr())

#%%
# ---
# Data and dataloading
# ---
max_length = 128
tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased') # we'll have this already

# Taking at face value that we need to encode like so for NLP to work properly
def convert_to_stsb_features(example_batch):
    inputs = list(zip(example_batch['sentence1'], example_batch['sentence2']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_mnli_features(example_batch):
    inputs = list(zip(example_batch['hypothesis'], example_batch['premise']))
    features = tokenizer.batch_encode_plus(
        inputs, max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_sst2_features(example_batch):
    features = tokenizer.batch_encode_plus(
        example_batch["sentence"], max_length=max_length, pad_to_max_length=True
    )
    features["labels"] = example_batch["label"]
    return features

def convert_to_pos_features(example_batch):
    # This is the most naive guess, `utils_ner` suggest the actual procedure is harder
    # TODO use the utils_ner conversion -- it'll also process the labels correctly..
    features = tokenizer(example_batch['words'])
    features["labels"] = example_batch["pos"]
    return features

convert_func_dict = {
    "stsb": convert_to_stsb_features,
    "sst2": convert_to_sst2_features,
    "mnli": convert_to_mnli_features,
    # "pos": convert_to_pos_features
}


#%% get cached features
columns_dict = {
    "sst2": ['input_ids', 'attention_mask', 'labels'],
    "stsb": ['input_ids', 'attention_mask', 'labels'],
    "mnli": ['input_ids', 'attention_mask', 'labels'],
    # "pos": ['input_ids', 'attention_mask', 'labels'],
}

features_dict = {}
for task_name, dataset in dataset_dict.items():
    features_dict[task_name] = {}
    for phase, phase_dataset in dataset.items():
        features_dict[task_name][phase] = phase_dataset.map(
            convert_func_dict[task_name],
            batched=True,
            load_from_cache_file=True,
            cache_file_name=f"/srv/share/svanga3/bert-representations/nlp_datasets/cached_batches/{task_name}.cache"
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))
        features_dict[task_name][phase].set_format(
            type="torch",
            columns=columns_dict[task_name],
        )
        print(task_name, phase, len(phase_dataset), len(features_dict[task_name][phase]))

#%%
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
              collate_fn=self.data_collator.collate_batch,
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


#%%
# Now do it with POS
pos_dataset = nlp.load_dataset('conll2003', cache_dir="/srv/share/svanga3/bert-representations/nlp_datasets/POS/")
def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

label_list = get_label_list(pos_dataset["train"]["pos"])
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

#%%
label_list