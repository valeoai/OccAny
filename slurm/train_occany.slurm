#!/bin/bash
#SBATCH --job-name=occany
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --array=0-1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --partition=qgpu
#SBATCH -A eu-25-92
#SBATCH --output=slurm/output/train_occany_%A_%a.out
#SBATCH --error=slurm/output/train_occany_%A_%a.err

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TRAIN_TASK_ID="${TRAIN_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
export NUM_NODE="${NUM_NODE:-${SLURM_NNODES:-2}}"
export NUM_GPU_PER_NODE="${NUM_GPU_PER_NODE:-8}"

case "$TRAIN_TASK_ID" in
    0)
        NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" NUM_NODE="$NUM_NODE" N_WORKERS="${N_WORKERS:-10}" BATCH_SIZE="${BATCH_SIZE:-2}" bash sh/train_occany_recon.sh
        ;;
    1)
        NUM_GPU_PER_NODE="$NUM_GPU_PER_NODE" NUM_NODE="$NUM_NODE" N_WORKERS="${N_WORKERS:-10}" BATCH_SIZE="${BATCH_SIZE:-2}" bash sh/train_occany_gen.sh
        ;;
    *)
        echo "ERROR: Unsupported TRAIN_TASK_ID '$TRAIN_TASK_ID'. Valid values: 0 (recon), 1 (gen)." >&2
        exit 1
        ;;
esac
