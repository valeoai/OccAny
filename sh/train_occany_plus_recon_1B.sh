#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

export EXP_NAME="occany_plus_recon_1B"


: ${BATCH_SIZE:=2}
: ${EFFECTIVE_BATCH_SIZE:=64}
: ${N_WORKERS:=12}

export EPOCHS=100

# Default values for multi-node setup
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch_da3.py')"
echo "Starting training..."
echo "$CMD"

WIDTH=518
HEIGHT=168



RAY_MAP_PROB=0.4

$CMD \
    --train_dataset="6000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=10, ray_map_prob=$RAY_MAP_PROB,\
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3') + \
        5000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$SCRATCH/data/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=10, ray_map_prob=$RAY_MAP_PROB,\
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3') + \
        6000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl',\
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=10, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3') + \
        6000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=10, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3') + \
        6000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=10, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True, distill_model_name='SAM3', base_model='da3') + \
        4000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=5, ray_map_prob=$RAY_MAP_PROB,\
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0, distill_model_name='SAM3', base_model='da3') + \
        4000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_surround_all.pkl',\
        min_memory_num_views=6, frame_interval=1, max_memory_num_views=6, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0, distill_model_name='SAM3', base_model='da3') + \
        4000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_surround_all.pkl', \
        min_memory_num_views=6, frame_interval=1, max_memory_num_views=6, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0, distill_model_name='SAM3', base_model='da3') + \
        4000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_surround_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=5, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=128, z_far=50, split='train', \
        resolution=[(518, 294), (518, 280), (518, 266), (518, 210), (518, 168)], \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0, distill_model_name='SAM3', base_model='da3')"  \
    --test_dataset="206 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, reverse_seq=False, \
        z_far=50, split='val', recon_view_idx=[0, 2, 4, 6, 8], ray_map_idx=[1, 3, 5, 7], \
        resolution=[(518, 168)], distill_model_name='SAM3', base_model='da3')" \
    --lr=5e-5 --min_lr=1e-6 --warmup_epochs=3 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=3 --keep_freq=5 --eval_freq=1  --num_workers=$N_WORKERS --multiview \
    --amp bf16 --fixed_eval_set --loss_enc_feat  \
    --output_dir="$PROJECT/tb_log_occany/$EXP_NAME" \
    --training_objective pointmap_depth_ray --fine_tune_layers 34,35,36,37,38,39 \
    --sam3_proj_lr_mult 10.0 \
    --da3_model_name depth-anything/DA3-GIANT-1.1 \
    --loss_type L1 --pointmap_lambda_c 1.0 --depth_lambda_c 0.0 --raymap_lambda_c 1.0 \
    --distill_model SAM3 --distill_criterion "DistillLoss(nn.L1Loss(), use_conf=False)" \
    --aux_branch_layers 6 --scale_inv_depth_loss --lambda_scale_inv_depth 1.0 \
     --lambda_depth 1.0 --lambda_pointmap 1.0 --sam3_use_dpt_proj
