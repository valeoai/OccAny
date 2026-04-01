#!/bin/bash
PROJECT="${PROJECT:-$PWD}"

EXP_LIST="${EXP_LIST:-recon}"

case "$EXP_LIST" in
    recon)
        source "sh/exp_lists/recon.sh"
        ;;
    *)
        echo "ERROR: Unknown EXP_LIST '$EXP_LIST'. Valid: recon"
        exit 1
        ;;
esac

EXP_ID="${EXP_ID:-0}"
if [ -z "${exp_extra_args[$EXP_ID]:-}" ]; then
    echo "ERROR: EXP_ID '$EXP_ID' is out of range for EXP_LIST '$EXP_LIST'"
    exit 1
fi

PID="${SLURM_ARRAY_TASK_ID:-0}"
WORLD="${WORLD:-1}"

kitti_root="${KITTI_ROOT:-$PROJECT/data/kitti}"
nuscenes_root="${NUSCENES_ROOT:-$PROJECT/data/nuscenes}"

python extract_recon.py \
    --output_dir "${RECON_OUTPUT:-./outputs}" \
    --kitti_root "$kitti_root" \
    --nuscenes_root "$nuscenes_root" \
    --world="$WORLD" --pid="$PID" \
    --silent \
    ${exp_extra_args[$EXP_ID]}
