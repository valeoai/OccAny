#!/bin/bash
#SBATCH --job-name=eval_occany
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --hint=nomultithread
#SBATCH -A eu-25-92
#SBATCH --partition=qgpu
#SBATCH --output=slurm/output/eval_occany_%A_%a.out
#SBATCH --error=slurm/output/eval_occany_%A_%a.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export EXP_LIST="${EXP_LIST:-occany_plus}"
export EXP_ID="${EXP_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
export WORLD="${WORLD:-20}"

bash sh/eval_occany.sh
