export SEQ_PREFIX="${SEQ_PREFIX:-}"

if [ "$SEQ_MODE" = "surround" ]; then
    export CAMERA=surround
    export SEQ_ARGS=""
elif [ "$SEQ_MODE" = "temporal" ]; then
    export CAMERA=all
    export SEQ_ARGS="--subsampling_rate 5 --max_stride 9"
else
    echo "Invalid seq mode: $SEQ_MODE"
    exit 1
fi

if [ "$DATASET" = "waymo" ]; then
    echo "Making seq for Waymo"
    python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset waymo \
        --camera $CAMERA \
        --seq_mode $SEQ_MODE
elif [ "$DATASET" = "vkitti" ]; then
    echo "Making seq for Virtual Kitti"
    if [ "$SEQ_MODE" = "surround" ]; then
        echo "Skipping Virtual KITTI - surround mode not applicable (single camera only)"
    else
        python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset vkitti \
            --seq_mode $SEQ_MODE
    fi
elif [ "$DATASET" = "ddad" ]; then
    echo "Making seq for DDAD"
    python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset ddad \
        --camera $CAMERA \
        --seq_mode $SEQ_MODE
elif [ "$DATASET" = "pandaset" ]; then
    echo "Making seq for Pandaset"
    python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset pandaset \
        --camera $CAMERA \
        --seq_mode $SEQ_MODE
elif [ "$DATASET" = "kitti" ]; then
    echo "Making seq for Kitti"
    # KITTI only has 1 camera, so surround modes are not applicable
    if [ "$SEQ_MODE" = "surround" ]; then
        echo "Skipping KITTI - surround mode not applicable (single camera only)"
    else
        python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset kitti \
            --camera $CAMERA \
            --seq_mode $SEQ_MODE
    fi
elif [ "$DATASET" = "once" ]; then
    echo "Making seq for ONCE"
    python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset once \
        --camera $CAMERA \
        --seq_mode $SEQ_MODE
elif [ "$DATASET" = "occ3d_nuscenes" ]; then
    echo "Making seq for Occ3D-nuScenes"
    # occ3d_nuscenes only supports surround mode (no temporal)
    if [ "$SEQ_MODE" = "temporal" ]; then
        echo "Skipping occ3d_nuscenes - temporal mode not applicable"
    else
        python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset occ3d_nuscenes \
            --camera $CAMERA \
            --seq_mode $SEQ_MODE
    fi
elif [ "$DATASET" = "occ3d_nuscenes_all" ]; then
    echo "Making seq for Occ3D-nuScenes at 10Hz"
    # occ3d_nuscenes only supports surround mode (no temporal)
    if [ "$SEQ_MODE" = "temporal" ]; then
        echo "Skipping occ3d_nuscenes - temporal mode not applicable"
    else
        python dataset_setup/base_make_seq.py --prefix "$SEQ_PREFIX" $SEQ_ARGS --dataset occ3d_nuscenes_all \
            --camera $CAMERA \
            --seq_mode $SEQ_MODE
    fi
else
    echo "Invalid dataset: $DATASET"
    exit 1
fi
