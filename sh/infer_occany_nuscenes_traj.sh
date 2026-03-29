#!/usr/bin/bash
#
# NuScenes trajectory evaluation with OccAny model.
#
# Computes ADE (Average Displacement Error) between predicted and
# ground-truth trajectories on NuScenes Vista validation set.
#
# Outputs:
#   ${output_dir}/ade_metrics.json          – per-sample and mean ADE
#   ${output_dir}/trajectory_plots/*.png    – trajectory comparison plots
#
# Usage:
#   bash sh/infer_occany_nuscenes_traj.sh
#   OCCANY_PLUS_RECON_CKPT=checkpoints/occany_plus_recon.pth bash sh/infer_occany_nuscenes_traj.sh

export OMP_NUM_THREADS=4

# ─── PYTHONPATH setup ───────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${REPO_ROOT}/sh/train_common.sh"
occany_prepend_pythonpath "${REPO_ROOT}"
cd "${REPO_ROOT}"

# ─── Paths ──────────────────────────────────────────────────────────────────
data_folder="$PROJECT/data/nuscenes"       # NuScenes data root
anno_file="vista_nuscenes_val_traj.json"        # Vista annotation JSON
: "${PRETRAINED_CKPTS:=./checkpoints}"
: "${OCCANY_PLUS_RECON_CKPT:=}"

if [ -n "${OCCANY_PLUS_RECON_CKPT}" ]; then
    occany_ckpt="${OCCANY_PLUS_RECON_CKPT}"
elif [ -f "${PRETRAINED_CKPTS}/occany_plus_recon.pth" ]; then
    occany_ckpt="${PRETRAINED_CKPTS}/occany_plus_recon.pth"
else
    occany_ckpt="${PRETRAINED_CKPTS}/occany_plus_recon_1B.pth"
fi

ckpt_name="$(basename "${occany_ckpt%.pth}")"
: "${OUTPUT_DIR:=./outputs/${ckpt_name}_nuscenes_traj}"
output_dir="${OUTPUT_DIR}"

if [ ! -f "${occany_ckpt}" ]; then
    echo "ERROR: OccAny checkpoint not found at ${occany_ckpt}" >&2
    echo "Set OCCANY_PLUS_RECON_CKPT to the reconstruction checkpoint you want to evaluate." >&2
    exit 1
fi

# ─── Image settings ────────────────────────────────────────────────────────
img_size="294 518"
crop_img_size="294 518"

# ─── Inference controls ────────────────────────────────────────────────────
batch_size=4
num_workers=16
n_frames=25
max_batches=-1        # -1 = evaluate all
plot_every=20         # save a trajectory plot every N samples (0 disables)

# ─── Run ────────────────────────────────────────────────────────────────────
python3 infer_trajectory.py \
    --data-folder "${data_folder}" \
    --anno-file "${anno_file}" \
    --occany-ckpt "${occany_ckpt}" \
    --img-size ${img_size} \
    --crop-img-size ${crop_img_size} \
    --n-frames "${n_frames}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --max-batches "${max_batches}" \
    --plot-every "${plot_every}" \
    --output-dir "${output_dir}"
