#!/bin/bash
PROJECT="${PROJECT:-$PWD}"
SCRATCH="${SCRATCH:-$PWD/eval_output}"

mkdir -p "$SCRATCH/ssc_voxel_pred" "$SCRATCH/ssc_output" "$SCRATCH/data"

EXP_LIST="${EXP_LIST:-occany}"

case "$EXP_LIST" in
    occany)
        source "sh/exp_lists/occany.sh"
        MODEL="occany_must3r"
        ;;
    occany_plus)
        source "sh/exp_lists/occany_plus.sh"
        MODEL="occany_da3"
        ;;
    *)
        echo "ERROR: Unknown EXP_LIST '$EXP_LIST'. Valid: occany | occany_plus"
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

SCALE_BY_GT_ARG="${SCALE_BY_GT_ARG:-}"
USE_RENDER_OUTPUT="${USE_RENDER_OUTPUT:---use_render_output}"
KEY_TO_GET_PTS3D="${KEY_TO_GET_PTS3D:---key_to_get_pts3d pts3d}"


kitti_root="${KITTI_ROOT:-$PROJECT/data/kitti}"
nuscenes_root="${NUSCENES_ROOT:-$PROJECT/data/nuscenes}"

python extract_output_occany.py \
    --output_dir "$SCRATCH/ssc_voxel_pred" \
    --vis_output_dir "$SCRATCH/ssc_output" \
    --kitti_root "$kitti_root" \
    --nuscenes_root "$nuscenes_root" \
    --model "$MODEL" \
    --silent ${exp_extra_args[$EXP_ID]} \
    --world="$WORLD" --pid="$PID" ${SCALE_BY_GT_ARG} ${USE_RENDER_OUTPUT} ${KEY_TO_GET_PTS3D}
