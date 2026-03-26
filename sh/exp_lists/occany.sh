common_exp_name="--exp_name OccAny --image_size 512"

exp_extra_args=(
    # 0: kitti 5frames
    "$common_exp_name --batch_gen_view 12 --dataset kitti --setting 5frames --gen -rot 60 -vpi 10 -fwd 3 -seed_trans 2 --boxes_folder resized_1216_box5_text5_DINOB --box_conf_thres 0.15 --recon_conf_thres 2.5 --gen_conf_thres 2.5"

    # 1: kitti 1frame
    "$common_exp_name --batch_gen_view 16 --dataset kitti --setting 1frame --gen -rot 60 -vpi 50 -fwd 1 -seed_trans 2 --boxes_folder resized_1216_box5_text5_DINOB --recon_conf_thres 2.0 --gen_conf_thres 2.0"

    # 2: nuscenes 5frames
    "$common_exp_name --batch_gen_view 12 --dataset nuscenes --setting 5frames --gen -rot 60 -vpi 10 -fwd 3 -seed_trans 2 --boxes_folder resized_1328_box5_text5_DINOB --recon_conf_thres 1.1 --gen_conf_thres 1.1"

    # 3: nuscenes surround
    "$common_exp_name --batch_gen_view 12 --dataset nuscenes --setting surround --gen -rot 60 -vpi 10 -fwd 3 --seed_trans 2 --boxes_folder resized_1328_box5_text5_DINOB --recon_conf_thres 1.1 --gen_conf_thres 1.1"

    # 4: kitti 5frames semantic distill
    "$common_exp_name --compute_segmentation_masks -nsr --merge_masks --batch_gen_view 12 --dataset kitti --setting 5frames --gen -rot 60 -vpi 10 -fwd 1 -seed_trans 2 --semantic distill@SAM2_large --boxes_folder resized_1216_box5_text5_DINOB --box_conf_thres 0.15 --recon_conf_thres 2.0 --gen_conf_thres 2.0"

    # 5: nuscenes surround semantic distill
    "$common_exp_name --compute_segmentation_masks -nsr --merge_masks --batch_gen_view 12 --dataset nuscenes --setting surround --semantic distill@SAM2_large --gen -rot 60 -vpi 10 -fwd 1 --boxes_folder resized_1328_box5_text5_DINOB --box_conf_thres 0.15 --recon_conf_thres 1.01 --gen_conf_thres 1.01"

    # 6: kitti 5frames semantic pretrained
    "$common_exp_name --compute_segmentation_masks -nsr -osr --merge_masks --batch_gen_view 12 --dataset kitti --setting 5frames --gen -rot 60 -vpi 10 -fwd 1 -seed_trans 2 --semantic pretrained@SAM2_large --boxes_folder resized_1216_box5_text5_DINOB --box_conf_thres 0.15 --recon_conf_thres 2.0 --gen_conf_thres 2.0"

    # 7: nuscenes surround semantic pretrained
    "$common_exp_name --compute_segmentation_masks -nsr -osr --merge_masks --batch_gen_view 1 --dataset nuscenes --setting surround --semantic pretrained@SAM2_large --gen -rot 60 -vpi 10 -fwd 1 --boxes_folder resized_1328_box5_text5_DINOB --box_conf_thres 0.15 --recon_conf_thres 1.01 --gen_conf_thres 1.01"
)

