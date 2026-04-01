common_args="--exp_name occany_plus_recon_1B --occany_recon_ckpt ./checkpoints/occany_plus_recon_1B.pth"

exp_extra_args=(
    # 0: KITTI 5frames — OccAny+ 1B
    "$common_args --model occany_da3 --dataset kitti --setting 5frames"

    # 1: nuScenes surround — OccAny+ 1B
    "$common_args --model occany_da3 --dataset nuscenes --setting surround"

    # 2: KITTI 5frames — plain DA3 Giant
    "--exp_name da3_recon --model da3 --dataset kitti --setting 5frames"

    # 3: nuScenes surround — plain DA3 Giant
    "--exp_name da3_recon --model da3 --dataset nuscenes --setting surround"
)
