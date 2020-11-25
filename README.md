# bert-representations
A Unified View of BERT Representations in Fine-Tuning, Transfer, and Forgetting
Course project for Deep Learning for Text. 
Work by Joel Ye, Sri Vivek Vanga, Ayush Shrivastava.

Writeup (with citations) in `bert-representations.pdf`.

A normal project might have main scripts, trainers, and model files. HuggingFace graciously takes care of all of this infrastructure. Our codebase largely builds on top of this, consisting of scripts that run the exact experiments, and extract + analyze representations.

We still use a configuration system in order to track our experiments. Our main script will forward to a fully configured downstream script (e.g. the finetuning tasks should also be a part of configuration).

## Installation
Install requirements in a virtual environment of your preference, using

`pip install -r requirements.txt`
