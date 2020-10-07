#!/bin/bash

if [[ $# -eq 2 ]]
then
    python -u run.py --run-type eval --exp-config configs/$1.yaml --ckpt-path $2
elif [[ $# -eq 3 ]]
then
    python -u run.py --run-type eval --exp-config configs/$1.yaml --run-id $2 --ckpt-path $3
else
    echo "Expected args <variant> (ckpt)"
fi