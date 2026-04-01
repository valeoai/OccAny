#!/bin/bash
set -euo pipefail

EXP_LIST="${EXP_LIST:-metric_recon}"

case "$EXP_LIST" in
    metric_recon)
        source "sh/exp_lists/metric_recon.sh"
        ;;
    *)
        echo "ERROR: Unknown EXP_LIST '$EXP_LIST'. Valid: metric_recon"
        exit 1
        ;;
esac

EXP_ID="${EXP_ID:-0}"
if [ -z "${exp_extra_args[$EXP_ID]:-}" ]; then
    echo "ERROR: EXP_ID '$EXP_ID' is out of range for EXP_LIST '$EXP_LIST'"
    exit 1
fi

python compute_recon_metrics.py \
    ${exp_extra_args[$EXP_ID]}
