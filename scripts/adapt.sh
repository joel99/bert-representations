#!/bin/bash
#SBATCH --job-name=br-tune
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
    python -u run_auc.py --run-type train --exp-config configs/$1.yaml --ckpt-path $2
else
    echo "Expected args <variant> (ckpt)"
fi