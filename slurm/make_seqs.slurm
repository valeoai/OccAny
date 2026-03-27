#!/bin/bash

#SBATCH --job-name=make_seqs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=qgpu


#SBATCH --output=slurm/output/make_seqs_%A_%a.out
#SBATCH --error=slurm/output/make_seqs_%A_%a.err

#SBATCH --array=0-9

#SBATCH -A eu-25-92

source ~/.bashrc

mkdir -p slurm/output

eval "$(conda shell.bash hook)"
conda activate occany

# Toggle test-prefixed outputs, e.g. testseq_exact_len_sub5_stride9_all.pkl
# export SEQ_PREFIX="${SEQ_PREFIX:-test_}"



if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
    DATASET=waymo SEQ_MODE=temporal bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 1 ]; then
    DATASET=once SEQ_MODE=temporal bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 2 ]; then
    DATASET=ddad SEQ_MODE=temporal bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    DATASET=pandaset SEQ_MODE=temporal bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 4 ]; then
    DATASET=vkitti SEQ_MODE=temporal bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 5 ]; then
    DATASET=kitti SEQ_MODE=temporal bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 6 ]; then
    DATASET=waymo SEQ_MODE=surround bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 7 ]; then
    DATASET=once SEQ_MODE=surround bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 8 ]; then
    DATASET=ddad SEQ_MODE=surround bash sh/make_seqs.sh
elif [ "$SLURM_ARRAY_TASK_ID" -eq 9 ]; then
    DATASET=pandaset SEQ_MODE=surround bash sh/make_seqs.sh
else
    echo "Invalid SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
    exit 1
fi
