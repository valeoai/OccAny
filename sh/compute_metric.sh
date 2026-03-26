#!/bin/bash
set -euo pipefail

PROJECT="${PROJECT:-$PWD}"
SCRATCH="${SCRATCH:-$PWD/eval_output}"

mkdir -p "$SCRATCH/ssc_voxel_pred" "$SCRATCH/ssc_output" "$SCRATCH/data"

USE_MAJORITY_POOLING="${USE_MAJORITY_POOLING:-1}"
MAJORITY_POOLING_ARG=""
POOLING_MODE="${POOLING_MODE:-separate}"
if [ "$USE_MAJORITY_POOLING" = "1" ]; then
    MAJORITY_POOLING_ARG="--apply_majority_pooling --pooling_mode ${POOLING_MODE}"
fi

EXP_LIST="${EXP_LIST:-metric_occany}"

case "$EXP_LIST" in
    metric_occany)
        source "sh/exp_lists/metric_occany.sh"
        ;;
    metric_occany_plus)
        source "sh/exp_lists/metric_occany_plus.sh"
        ;;
    *)
        echo "ERROR: Unknown EXP_LIST '$EXP_LIST'. Valid: metric_occany | metric_occany_unifiedpool"
        exit 1
        ;;
esac

EXP_ID="${EXP_ID:-0}"
if [ -z "${exp_extra_args[$EXP_ID]:-}" ]; then
    echo "ERROR: EXP_ID '$EXP_ID' is out of range for EXP_LIST '$EXP_LIST'"
    exit 1
fi

if [ -n "${DSDIR:-}" ]; then
    kitti_root="${KITTI_ROOT:-$DSDIR/SemanticKITTI}"
    nuscenes_root="${NUSCENES_ROOT:-$SCRATCH/data/occ3d_nuscenes}"
else
    kitti_root="${KITTI_ROOT:-$PROJECT/data/kitti}"
    nuscenes_root="${NUSCENES_ROOT:-$PROJECT/data/nuscenes}"
fi

NO_MAJORITY_POOLING_ARG=""
if [ "${NO_MAJORITY_POOLING:-0}" = "1" ]; then
    NO_MAJORITY_POOLING_ARG="--no_majority_pooling"
fi

python compute_metrics_from_saved_voxels.py \
    --kitti_root "$kitti_root" \
    --nuscenes_root "$nuscenes_root" \
    ${exp_extra_args[$EXP_ID]} \
    ${NO_MAJORITY_POOLING_ARG}
