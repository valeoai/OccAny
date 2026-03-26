exp_extra_args=(
    # 0: DA3 SAM3 kitti 5frames
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAnyPlus_5frames_kitti512_rot60_vpi10_fwd3_sTrans2 --dataset kitti --setting 5frames --output_type render_recon_gen --recon_threshold 12.0 --gen_threshold 8.0 --geometry_only $MAJORITY_POOLING_ARG"

    # 1: DA3 SAM3 nuscenes surround
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAnyPlus_surround_nuscenes512_rot60_vpi10_fwd3_sTrans2 --dataset nuscenes --setting surround --output_type render_recon_gen --recon_threshold 4.0 --gen_threshold 6.0 --geometry_only $MAJORITY_POOLING_ARG"

    # 2: DA3 SAM3 kitti 5frames distill semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAnyPlus_5frames_kitti512_sam3th15_res1008_rot60_vpi10_fwd1_sTrans2_nsr_mm --dataset kitti --setting 5frames --output_type render_recon_gen --recon_threshold 6.0 --gen_threshold 6.0 --eval_superclass $MAJORITY_POOLING_ARG"

    # 3: DA3 SAM3 nuscenes surround  distill semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAnyPlus_surround_nuscenes512_sam3th15_res1008_rot60_vpi10_fwd1_nsr_mm --dataset nuscenes --setting surround --output_type render_recon_gen --recon_threshold 2.0 --gen_threshold 6.0 --eval_superclass $MAJORITY_POOLING_ARG"

    # 4: DA3 SAM3 kitti 5frames pretrained semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAnyPlus_5frames_kitti512_sam3th15_res1008_rot60_vpi10_fwd1_sTrans2_nsr_mm_osfr --dataset kitti --setting 5frames --output_type render_recon_gen --recon_threshold 6.0 --gen_threshold 6.0 --eval_superclass $MAJORITY_POOLING_ARG"

    # 5: DA3 SAM3 nuscenes surround pretrained semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAnyPlus_surround_nuscenes512_sam3th15_res1008_rot60_vpi10_fwd1_nsr_mm_osfr --dataset nuscenes --setting surround --output_type render_recon_gen --recon_threshold 2.0 --gen_threshold 6.0 --eval_superclass $MAJORITY_POOLING_ARG"
    
)
