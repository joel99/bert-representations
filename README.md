# bert-representations
A Unified View of BERT Representations in Fine-Tuning, Transfer, and Forgetting

A normal project might have main scripts, trainers, and model files. HuggingFace graciously takes care of all of this infrastructure. Our codebase largely builds on top of this, consisting of scripts that run the exact experiments, and extract + analyze representations.

We still use a configuration system in order to track our experiments. Our main script will forward to a fully configured downstream script (e.g. the finetuning tasks should also be a part of configuration).

## Installation
Install requirements in a virtual environment of your preference, using

`pip install -r requirements.txt`

## Tasklog
- cite huggingface
- CKA utilities

### Experimental Notes
- [set cache](https://huggingface.co/transformers/installation.html#caching-models)
- use pip to install HuggingFace