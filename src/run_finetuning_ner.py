from transformers import AutoConfig, AutoModelForSequenceClassification, GlueDataset
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    Trainer,
    glue_tasks_num_labels,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from src import (
    get_eval_metrics_func,
    logger
)
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask

def run_ner(cfg, model_args, training_args, tokenizer, ckpt_path=None):
    r"""
        cfg: YACS cfg node
        ckpt_path: Unsupported
    """
