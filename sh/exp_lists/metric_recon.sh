RECON_OUTPUT="${RECON_OUTPUT:-./outputs}"

exp_extra_args=(
    # 0: KITTI 5frames — OccAny+ 1B
    "--exp_dir $RECON_OUTPUT/occany_plus_recon_1B_occany_da3_kitti_5frames_img512"

    # 1: nuScenes surround — OccAny+ 1B
    "--exp_dir $RECON_OUTPUT/occany_plus_recon_1B_occany_da3_nuscenes_surround_img512"

    # 2: KITTI 5frames — plain DA3 Giant
    "--exp_dir $RECON_OUTPUT/da3_recon_da3_kitti_5frames_img512"

    # 3: nuScenes surround — plain DA3 Giant
    "--exp_dir $RECON_OUTPUT/da3_recon_da3_nuscenes_surround_img512"
)
