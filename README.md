# bert-representations
A Unified View of BERT Representations in Fine-Tuning, Transfer, and Forgetting

A normal project might have main scripts, trainers, and model files. HuggingFace graciously takes care of all of this infrastructure. Our codebase largely builds on top of this, consisting of scripts that run the exact experiments, and extract + analyze representations.

We still use a configuration system in order to track our experiments. Our main script will forward to a fully configured downstream script (e.g. the finetuning tasks should also be a part of configuration).

## Tasks for Joel
- do it again (match reported)

Q: Do the models being finetuned has their base saved separately from their head?

## Installation
Install requirements in a virtual environment of your preference, using

`pip install -r requirements.txt`

## Tasklog
- cite huggingface
- Pull HuggingFace and set up pretrained BERT
- Pull in GLUE + other tasks
- Setup fine-tuning pipeline
- CKA utilities