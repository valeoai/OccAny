#!/bin/bash

#SBATCH --job-name=eval_recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu

#SBATCH --output=slurm/output/eval_recon_%A_%a.out
#SBATCH --error=slurm/output/eval_recon_%A_%a.err

#SBATCH --array=0,1,2,3

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany


case "$SLURM_ARRAY_TASK_ID" in
    0)
        python eval_recon.py \
            --model occany_da3 \
            --dataset kitti \
            --setting 5frames \
            --exp_name occany_plus_recon_1B \
            --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth
        ;;
    1)
        python eval_recon.py \
            --model occany_da3 \
            --dataset nuscenes \
            --setting surround \
            --exp_name occany_plus_recon_1B \
            --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth
        ;;
    2)
        python eval_recon.py \
            --model da3 \
            --dataset kitti \
            --setting 5frames \
            --exp_name da3_recon
        ;;
    3)
        python eval_recon.py \
            --model da3 \
            --dataset nuscenes \
            --setting surround \
            --exp_name da3_recon
        ;;
    *)
        echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
        exit 1
        ;;
esac
