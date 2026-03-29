#!/bin/bash

#SBATCH --job-name=eval_traj
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu


#SBATCH --output=slurm/output/eval_traj_%A_%a.out
#SBATCH --error=slurm/output/eval_traj_%A_%a.err

#SBATCH --array=0,1

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany


if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon.pth \
        bash sh/infer_occany_nuscenes_traj.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    OCCANY_PLUS_RECON_CKPT=./checkpoints/occany_plus_recon_1B.pth \
        bash sh/infer_occany_nuscenes_traj.sh
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi
