#!/bin/bash

if [[ $# -eq 1 ]]
then
    python -u run.py --run-type train --exp-config configs/$1.yaml
elif [[ $# -eq 2 ]]
then
    python -u run.py --run-type train --exp-config configs/$1.yaml --ckpt-path $2
else
    echo "Expected args <variant> (ckpt)"
fi