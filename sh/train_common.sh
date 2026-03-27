#!/bin/bash

occany_prepend_pythonpath() {
    local repo_root="$1"
    local -a vendored_paths=(
        "$repo_root/third_party"
        "$repo_root/third_party/dust3r"
        "$repo_root/third_party/croco/models/curope"
        "$repo_root/third_party/Grounded-SAM-2"
        "$repo_root/third_party/Grounded-SAM-2/grounding_dino"
        "$repo_root/third_party/sam3"
        "$repo_root/third_party/Depth-Anything-3/src"
    )
    local prefix=""
    local vendored_path
    for vendored_path in "${vendored_paths[@]}"; do
        if [ -d "$vendored_path" ]; then
            if [ -n "$prefix" ]; then
                prefix="${prefix}:$vendored_path"
            else
                prefix="$vendored_path"
            fi
        fi
    done
    if [ -n "${PYTHONPATH:-}" ]; then
        prefix="${prefix}:$PYTHONPATH"
    fi
    export PYTHONPATH="$prefix"
}

occany_prepare_train_env() {
    local repo_root="$1"
    cd "$repo_root"
    PROJECT="${PROJECT:-$repo_root}"
    if [ -n "${WORK:-}" ] && [ -n "${TRG_WORK:-}" ]; then
        PROJECT="$TRG_WORK"
    fi
    SCRATCH="${SCRATCH:-$PROJECT}"
    export PROJECT SCRATCH
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    occany_prepend_pythonpath "$repo_root"
}

occany_compute_accum_iter() {
    local effective_batch_size="$1"
    local num_gpu_per_node="$2"
    local batch_size="$3"
    local num_node="$4"
    local denom=$((num_gpu_per_node * batch_size * num_node))
    if [ "$denom" -le 0 ] || [ $((effective_batch_size % denom)) -ne 0 ]; then
        echo "ERROR: EFFECTIVE_BATCH_SIZE=$effective_batch_size is not divisible by NUM_GPU_PER_NODE * BATCH_SIZE * NUM_NODE = $denom" >&2
        return 1
    fi
    echo $((effective_batch_size / denom))
}

occany_select_train_cmd() {
    local launcher="$1"
    if tty -s; then
        echo "Running in interactive mode" >&2
        if [ "${NUM_GPU_PER_NODE:-1}" -eq 1 ] && [ "${NUM_NODE:-1}" -eq 1 ]; then
            printf 'python %s' "$launcher"
        else
            printf 'srun python %s' "$launcher"
        fi
    else
        echo "Running in SLURM job" >&2
        if [ "${NUM_GPU_PER_NODE:-1}" -eq 1 ] && [ "${NUM_NODE:-1}" -eq 1 ]; then
            printf 'srun python %s' "$launcher"
        else
            printf 'srun python %s' "$launcher"
        fi
    fi
}

occany_log_train_config() {
    local exp_name="$1"
    echo "NUM_NODE: $NUM_NODE"
    echo "NUM_GPU_PER_NODE: $NUM_GPU_PER_NODE"
    echo "ACCUM_ITER: $ACCUM_ITER"
    echo "BATCH_SIZE: $BATCH_SIZE"
    echo "EPOCHS: $EPOCHS"
    echo "EFFECTIVE_BATCH_SIZE: $EFFECTIVE_BATCH_SIZE"
    echo "N_WORKERS: $N_WORKERS"
    echo "Training $exp_name"
    echo "Using $NUM_GPU_PER_NODE GPUs per node ($NUM_NODE nodes)"
}
