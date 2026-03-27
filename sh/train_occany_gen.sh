#!/bin/bash
set -euo pipefail

source sh/train_common.sh
occany_prepare_train_env "$PWD"

export EXP_NAME="occany_gen"
: ${MUST3R_PRETRAINED_CKPT:="checkpoints/MUSt3R_512.pth"}
: ${OCCANY_RECON_CKPT:="checkpoints/occany_recon.pth"}

: ${BATCH_SIZE:=4}
: ${EFFECTIVE_BATCH_SIZE:=64}
: ${N_WORKERS:=12}

export EPOCHS=100

# Default values for multi-node setup
: ${NUM_NODE:=1}
: ${NUM_GPU_PER_NODE:=1}

export ACCUM_ITER
ACCUM_ITER="$(occany_compute_accum_iter "$EFFECTIVE_BATCH_SIZE" "$NUM_GPU_PER_NODE" "$BATCH_SIZE" "$NUM_NODE")"
occany_log_train_config "$EXP_NAME"
CMD="$(occany_select_train_cmd 'launch.py --mode=train')"
echo "Starting training..."
echo "$CMD"

if [ ! -f "$MUST3R_PRETRAINED_CKPT" ]; then
    echo "ERROR: Must3R checkpoint not found at $MUST3R_PRETRAINED_CKPT" >&2
    echo "Set MUST3R_PRETRAINED_CKPT or download the base checkpoint into checkpoints/." >&2
    exit 1
fi

if [ ! -f "$OCCANY_RECON_CKPT" ]; then
    echo "ERROR: Reconstruction checkpoint not found at $OCCANY_RECON_CKPT" >&2
    echo "Train the reconstruction stage first or override OCCANY_RECON_CKPT." >&2
    exit 1
fi

WIDTH=512
HEIGHT=160


RAY_MAP_PROB=0.5

$CMD \
    --train_dataset="6000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=9, ray_map_prob=$RAY_MAP_PROB,\
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True) + \
        5000 @ VKittiSeqMultiView(VKITTI_PROCESSED_ROOT='$SCRATCH/data/vkitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=9, ray_map_prob=$RAY_MAP_PROB,\
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True) + \
        6000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl',\
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=9, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True) + \
        6000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=9, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True) + \
        6000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=9, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, reverse_seq=True) + \
        4000 @ WaymoSeqMultiView(ROOT='$SCRATCH/data/waymo_processed', \
        seq_pkl_name='seq_surround_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=5, ray_map_prob=$RAY_MAP_PROB,\
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0) + \
        4000 @ DDADSeqMultiView(DDAD_PREPROCESSED_ROOT='$SCRATCH/data/ddad_processed', \
        seq_pkl_name='seq_surround_all.pkl',\
        min_memory_num_views=6, frame_interval=1, max_memory_num_views=6, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0) + \
        4000 @ PandasetSeqMultiView(PANDASET_PREPROCESSED_ROOT='$SCRATCH/data/pandaset_processed', \
        seq_pkl_name='seq_surround_all.pkl', \
        min_memory_num_views=6, frame_interval=1, max_memory_num_views=6, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0) + \
        4000 @ OnceSeqMultiView(ONCE_PREPROCESSED_ROOT='$SCRATCH/data/once_processed', \
        seq_pkl_name='seq_surround_all.pkl', \
        min_memory_num_views=5, frame_interval=1, max_memory_num_views=5, ray_map_prob=$RAY_MAP_PROB,
        aug_crop=112, z_far=50, split='train', \
        resolution=[(512, 336), (512, 288), (512, 256), (512, 160)], img_size=512, \
        transform=SeqColorJitter, aug_focal=0.9, shuffle_seq_prob=1.0)"  \
    --test_dataset="206 @ KittiSeqMultiView(KITTI_PREPROCESSED_ROOT='$SCRATCH/data/kitti_processed', \
        seq_pkl_name='seq_exact_len_sub5_stride9.pkl', frame_interval=1, \
        min_memory_num_views=10, max_memory_num_views=10, reverse_seq=False, \
        z_far=50, split='val', recon_view_idx=[0, 2, 4, 6, 8], ray_map_idx=[1, 3, 5, 7], \
        resolution=[(512, 160)])" \
    --decoder="Must3rDecoder(img_size=($WIDTH, $WIDTH), enc_embed_dim=1024, embed_dim=768, \
               pointmaps_activation=ActivationType.LINEAR, pred_sam_features=True, \
               feedback_type='single_mlp', memory_mode='kv', ray_map_encoder_depth=6, use_multitask_token=True)" \
    --train_criterion="ConfLoss_multiview(Regr3D_multiview(L21, norm_mode='?avg_dis', loss_in_log=False, \
                        pose_loss_value=1), alpha=0.2)" \
    --test_criterion="Regr3D_multiview(L21, norm_mode='?avg_dis', gt_scale=True, \
                      pose_loss_value=1)" \
    --train_criterion_gen="ConfLoss_multiview(Regr3D_multiview(L21, norm_mode='?avg_dis', loss_in_log=False, \
                        pose_loss_value=1), alpha=0.2)" \
    --test_criterion_gen="Regr3D_multiview(L21, norm_mode='?avg_dis', gt_scale=True, \
                          pose_loss_value=1)" \
    --pretrained="$MUST3R_PRETRAINED_CKPT"   \
    --pretrained_occany="$OCCANY_RECON_CKPT"   \
    --lr=7e-5 --min_lr=1e-6 --warmup_epochs=3 --epochs=$EPOCHS \
    --batch_size=$BATCH_SIZE --accum_iter=$ACCUM_ITER \
    --save_freq=3 --keep_freq=5 --eval_freq=1  --num_workers=$N_WORKERS --multiview \
    --amp bf16 --fixed_eval_set --loss_enc_feat  \
    --output_dir="$PROJECT/tb_log_occany/$EXP_NAME" \
    --gen \
    --sam_model SAM2 \
    --distill_model SAM2_large --distill_criterion "DistillLoss(MSELoss(), use_conf=True)"
