common_args="--exp_name OccAnyPlus --sam3_conf_th 0.15 --pose_from_depth_ray"

exp_extra_args=(
    # 0: kitti 5frames
    "$common_args --batch_gen_view 12 --dataset kitti --setting 5frames --gen -rot 60 -vpi 10 -fwd 3 -seed_trans 2 --vis_interval 50 --recon_conf_thres 12.0 --gen_conf_thres 8.0"

    # 1: nuscenes surround
    "$common_args --batch_gen_view 12 --dataset nuscenes --setting surround --gen -rot 60 -vpi 10 -fwd 3 --seed_trans 2 --vis_interval 400 --recon_conf_thres 4.0 --gen_conf_thres 6.0"

    # 2: kitti 5frames semantic distill
    "$common_args --compute_segmentation_masks -nsr --merge_masks --batch_gen_view 2 --dataset kitti --setting 5frames --gen -rot 60 -vpi 10 -fwd 1 -seed_trans 2 --semantic distill@SAM3 --sam3_conf_th 0.15 --vis_interval 50 --recon_conf_thres 6.0 --gen_conf_thres 6.0"

    # 3: nuscenes surround semantic distill
    "$common_args --compute_segmentation_masks -nsr --merge_masks --batch_gen_view 2 --dataset nuscenes --setting surround --semantic distill@SAM3 --gen -rot 60 -vpi 10 -fwd 1 --sam3_conf_th 0.15 --vis_interval 400 --recon_conf_thres 2.0 --gen_conf_thres 6.0"

    # 4: kitti 5frames semantic pretrained
    "$common_args --compute_segmentation_masks -nsr -osr --merge_masks --batch_gen_view 2 --dataset kitti --setting 5frames --gen -rot 60 -vpi 10 -fwd 1 -seed_trans 2 --semantic pretrained@SAM3 --sam3_conf_th 0.15 --vis_interval 50 --recon_conf_thres 6.0 --gen_conf_thres 6.0"

    # 5: nuscenes surround semantic pretrained
    "$common_args --compute_segmentation_masks -nsr -osr --merge_masks --batch_gen_view 2 --dataset nuscenes --setting surround --semantic pretrained@SAM3 --gen -rot 60 -vpi 10 -fwd 1 --sam3_conf_th 0.15 --vis_interval 400 --recon_conf_thres 2.0 --gen_conf_thres 6.0"

)