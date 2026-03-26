exp_extra_args=(
    # 0: OccAny SAM2 kitti 5frames geometry
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_5frames_kitti512_resized_1216_box5_text5_DINOB_boxth15_rot60_vpi10_fwd3_sTrans2 --dataset kitti --setting 5frames --output_type render_recon_gen --recon_threshold 2.5 --gen_threshold 2.5 --geometry_only $MAJORITY_POOLING_ARG"

    # 1: OccAny SAM2 kitti 1frame geometry
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_1frame_kitti512_resized_1216_box5_text5_DINOB_boxth5_rot60_vpi50_fwd1_sTrans2 --dataset kitti --setting 1frame --output_type render_recon_gen --recon_threshold 2.0 --gen_threshold 2.0 --geometry_only $MAJORITY_POOLING_ARG"

    # 2: OccAny SAM2 nuscenes 5frames geometry
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_5frames_nuscenes512_resized_1328_box5_text5_DINOB_boxth5_rot60_vpi10_fwd3_sTrans2 --dataset nuscenes --setting 5frames --output_type render_recon_gen --recon_threshold  1.1 --gen_threshold  1.1 --geometry_only $MAJORITY_POOLING_ARG"

    # 3: OccAny SAM2 nuscenes surround geometry
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_surround_nuscenes512_resized_1328_box5_text5_DINOB_boxth5_rot60_vpi10_fwd3_sTrans2 --dataset nuscenes --setting surround --output_type render_recon_gen --recon_threshold 1.1 --gen_threshold 1.1 --geometry_only $MAJORITY_POOLING_ARG"

    # 4: OccAny SAM2 kitti 5frames  distill semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_5frames_kitti512_resized_1216_box5_text5_DINOB_boxth15_rot60_vpi10_fwd1_sTrans2_nsr_mm --dataset kitti --setting 5frames --output_type render_recon_gen --recon_threshold 2.0 --gen_threshold 2.0 --eval_superclass $MAJORITY_POOLING_ARG"

    # 5: OccAny SAM2 nuscenes surround  distill semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_surround_nuscenes512_resized_1328_box5_text5_DINOB_boxth15_rot60_vpi10_fwd1_nsr_mm --dataset nuscenes --setting surround --output_type render_recon_gen --recon_threshold 1.01 --gen_threshold 1.01 --eval_superclass $MAJORITY_POOLING_ARG"


    # 6: OccAny SAM2 kitti 5frames pretrained semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_5frames_kitti512_resized_1216_box5_text5_DINOB_boxth15_rot60_vpi10_fwd1_sTrans2_nsr_mm_osfr --dataset kitti --setting 5frames --output_type render_recon_gen --recon_threshold 2.0 --gen_threshold 2.0 --eval_superclass $MAJORITY_POOLING_ARG"

    # 7: OccAny SAM2 nuscenes surround pretrained semantic
    "--exp_dir $SCRATCH/ssc_voxel_pred/OccAny_surround_nuscenes512_resized_1328_box5_text5_DINOB_boxth15_rot60_vpi10_fwd1_nsr_mm_osfr --dataset nuscenes --setting surround --output_type render_recon_gen --recon_threshold 1.01 --gen_threshold 1.01 --eval_superclass $MAJORITY_POOLING_ARG"
    
)