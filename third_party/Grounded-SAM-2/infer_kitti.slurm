#!/bin/bash

#SBATCH --job-name=infer_kitti
##SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00

#SBATCH --array=0-1

##SBATCH --time=02:00:00
##SBATCH --qos=qos_gpu-dev

##SBATCH --output=log_infer_kitti_%j.out
##SBATCH --error=log_infer_kitti_%j.err

#SBATCH --output=log_infer_kitti_%A_%a.out
#SBATCH --error=log_infer_kitti_%A_%a.err
#SBATCH --hint=nomultithread

#SBATCH --gres=gpu:1

#SBATCH -A trg@v100


##SBATCH -C v100-32g
##SBATCH --cpus-per-task=10

#SBATCH --cpus-per-task=3
#SBATCH --partition=gpu_p2

##SBATCH --signal=USR1@60

source ~/.bashrc

module purge
module load pytorch-gpu/py3/2.5.0
export PYTHONUSERBASE=$TRG_WORK/python_envs/grounded_sam2


cd $TRG_WORK/code/Grounded-SAM-2

if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    BOX_THRESHOLD=0.0
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    BOX_THRESHOLD=0.1
fi


python infer_kitti.py --box_threshold $BOX_THRESHOLD 

# python infer_kitti.py --box_threshold 0.1 --no_high_res_feature