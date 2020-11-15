#!/bin/bash
#SBATCH --job-name=br-eval
#SBATCH --gres gpu:1
#SBATCH --nodes 1
#SBATCH --cpus-per-task 3
#SBATCH --ntasks-per-node 1
#SBATCH --partition=short
#SBATCH --account=overcap
#SBATCH --output=slurm_logs/run-%j.out
#SBATCH --error=slurm_logs/run-%j.err

if [[ $# -eq 2 ]]
then
    python -u run_extract_targets.py --run-type eval --exp-config configs/$1.yaml --ckpt-path $2
elif [[ $# -eq 3 ]]
then
    python -u run.py --run-type eval --exp-config configs/$1.yaml --run-id $2 --ckpt-path $3
else
    echo "Expected args <variant> (ckpt)"
fi