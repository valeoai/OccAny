# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import argparse
import os
import sys
import types
from pathlib import Path
import torch
import numpy as np
import copy
import pickle
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
VENDORED_IMPORT_PATHS = [
    REPO_ROOT / "third_party",
    REPO_ROOT / "third_party" / "dust3r",
    REPO_ROOT / "third_party" / "croco" / "models" / "curope",
    REPO_ROOT / "third_party" / "Grounded-SAM-2",
    REPO_ROOT / "third_party" / "Grounded-SAM-2" / "grounding_dino",
    REPO_ROOT / "third_party" / "sam3",
    REPO_ROOT / "third_party" / "Depth-Anything-3" / "src",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)

import matplotlib.pyplot as pl
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor
from occany.utils.helpers import (
    apply_majority_pooling,
    build_fine_prompt_metadata,
    create_voxel_prediction,
    generate_intermediate_poses,
    transform_points_torch,
    save_semantic_2d_images,
)
from occany.utils.image_util import ImgNorm, get_SAM2_transforms, get_SAM3_transforms, convert_images_to_uint8_hwc
from PIL import Image
from occany.utils.io_da3 import setup_da3_models
from occany.utils.inference_helper import (
    build_demo_reconstruction_views,
    build_intrinsics_from_focal,
    convert_da3_output_to_occany_format,
    denormalize_da3_imgs_to_minus1_1,
    extract_demo_rgb_images,
    get_allowed_gen_view_ids,
    get_pts3d_from_voxel,
    is_distill_source,
    normalize_demo_rgb_image,
    parse_semantic_mode,
    populate_demo_sam2_box_dicts,
    uses_sam3_projection_features,
)

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
BILINEAR_RESAMPLE = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
MODEL_FAMILY_RESIZE_SETTINGS = {
    "da3": {"max_dimension": 518, "divisor": 14},
    "must3r": {"max_dimension": 512, "divisor": 16},
}


def round_to_nearest_divisible(value: float, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    return max(divisor, int(np.floor((value / divisor) + 0.5)) * divisor)


def get_output_resolution(image_size: Tuple[int, int], model_family: str) -> Tuple[int, int]:
    if model_family not in MODEL_FAMILY_RESIZE_SETTINGS:
        raise ValueError(f"Unsupported model family '{model_family}'")

    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {image_size}")

    resize_settings = MODEL_FAMILY_RESIZE_SETTINGS[model_family]
    max_dimension = resize_settings["max_dimension"]
    divisor = resize_settings["divisor"]

    if width >= height:
        scaled_short_side = height * max_dimension / width
        return max_dimension, round_to_nearest_divisible(scaled_short_side, divisor)

    scaled_short_side = width * max_dimension / height
    return round_to_nearest_divisible(scaled_short_side, divisor), max_dimension


def get_output_resolution_from_image(image_path: str, model_family: str) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return get_output_resolution(image.size, model_family=model_family)


COLORS = np.array([
            [0, 0, 0, 255],
            [112, 128, 144, 255],
            [220, 20, 60, 255],
            [255, 127, 80, 255],
            [255, 158, 0, 255],
            [233, 150, 70, 255],
            [255, 61, 99, 255],
            [0, 0, 230, 255],
            [47, 79, 79, 255],
            [255, 140, 0, 255],
            [255, 98, 70, 255],
            [0, 207, 191, 255],
            [175, 0, 75, 255],
            [75, 0, 75, 255],
            [112, 180, 60, 255],
            [222, 184, 135, 255],
            [0, 175, 0, 255],
            [135, 206, 235, 255], # sky, empty
        ])
CLASS_NAMES = [
            "other",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]
empty_class = 17
other_class = 0
n_classes = 18
# GaussTR
OCC3D_CATEGORIES = (
    ['other'],
    ['barrier', 'concrete barrier', 'metal barrier', 'water barrier'],
    ['bicycle', 'bicyclist'],
    ['bus'],
    ['car'],
    ['crane'],
    ['motorcycle', 'motorcyclist'],
    ['pedestrian', 'adult', 'child'],
    ['cone'],
    ['trailer'],
    ['truck'],
    ['road'],
    ['traffic island', 'rail track', 'lake', 'river'],
    ['sidewalk'],
    ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
    ['building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'street sign', 'traffic light'],
    ['tree', 'bush'],
    ['sky', 'empty'],
)
        
# PROMPT attribute for SAM-based semantic segmentation
PROMPT = list(OCC3D_CATEGORIES)












def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument(
        "--output_dir",
        type=str,
        default='./demo_data/output',
        help="Directory where inference outputs and visualizations are written",
    )
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the RGB demo directory containing frame folders",
        default='./demo_data/input',
    )
    parser.add_argument('--frame_interval', type=int, default=5, help='Frame interval for video processing')
    parser.add_argument(
        '--model',
        type=str,
        default='occany_da3',
        choices=['occany_must3r', 'occany_da3'],
        help='Model to use',
    )
    parser.add_argument('--gen', action='store_true', default=False, help='Predict raymap')
    
    parser.add_argument(
        '--semantic', "-sem",
        type=str,
        choices=['pretrained@SAM2_small',
                 'distill@SAM2_small',
                 'distill@SAM2_base',
                 'distill@SAM2_large',
                 'pretrained@SAM2_base',
                 'pretrained@SAM2_large',
                 'pretrained@SAM3',
                 'distill@SAM3'],
        default=None,
        help='Semantic processing option. Choices: pretrained@SAM_small, distill@SAM_tiny, distill@SAM_small.'
    )
    parser.add_argument('--compute_segmentation_masks', action='store_true', default=False,
                        help='Compute segmentation masks')
    parser.add_argument('--sam3_conf_th', type=float, default=0.15,
                        help='Confidence threshold for SAM3 semantic inference')
    parser.add_argument('--sam3_resolution', type=int, default=1008,
                        help='Resolution for SAM3 model')
    parser.add_argument('--view_batch_size', type=int, default=4,
                        help='Number of views per SAM3 inference chunk (lower uses less GPU memory)')
    parser.add_argument('--key_to_get_pts3d', type=str, default='pts3d',
                        help='Key to get pts3d from the output')
    parser.add_argument('--views_per_interval', '-vpi', type=int, default=2,
                        help='Number of views per interval for inference')
    parser.add_argument('--gen_rotate_novel_poses_angle', '-rot', type=int, default=0,
                        help='Angle to rotate novel poses')
    parser.add_argument('--gen_forward_novel_poses_dist', '-fwd', type=int, default=1,
                        help='Distance to move forward for novel poses (in meters)')
    parser.add_argument('--num_seed_rotations', '-nseed', type=int, default=0,
                        help='Number of seed rotations to generate (e.g., 5 for [-10, -5, 0, 5, 10]). If 0, uses standard mode.')
    parser.add_argument('--seed_rotation_angle', '-seed_rot', type=int, default=None,
                        help='Angle in degrees between seed rotations. If None, defaults to 15.0 degrees.')
    parser.add_argument('--seed_translation_distance', '-seed_trans', type=int, default=None,
                        help='Distance in meters to translate seed poses laterally. Positive rotations translate right, negative translate left.')
    parser.add_argument('--batch_gen_view', '-bs_gen', type=int, default=4,
                        help='Number of generated views per batch')
    parser.add_argument('--no_semantic_from_rotated_views', "-nsr", action='store_true', default=False,
                        help='Disable using semantics from rotated views (only use semantics from straight/forward views)')
    parser.add_argument('--box_conf_thres', type=float, default=0.05,
                        help='Confidence threshold for bounding box filtering')
    parser.add_argument('--merge_masks', action='store_true', default=False,
                        help='Merge masks by label and binned confidence (0.01 bins)')
    parser.add_argument('--only_semantic_from_recon_view', "-osr", action='store_true', default=False,
                        help='Use only semantic information from the reconstruction view (exclude all generated views)')
    parser.add_argument(
        '--gen_semantic_from_distill_sam3',
        action='store_true',
        default=False,
        help='For pretrained@SAM3, infer generated-view semantics from distilled SAM3 features when available',
    )
    parser.add_argument('--apply_majority_pooling', action='store_true', default=False,
                        help='Apply majority pooling to voxel predictions (3x3x3 neighborhood)')
    
    parser.add_argument('--pose_from_depth_ray', action='store_true', default=False,
                        help='Use ray pose estimation (set to True for trained models that use ray pose)')
    parser.add_argument('--point_from_depth_and_pose', action='store_true', default=False,
                        help='Compute pointmap from depth, intrinsics and c2w')
    parser.add_argument('--recon_conf_thres', type=float, default=2.0,
                        help='Reconstruction confidence threshold.')
    parser.add_argument('--gen_conf_thres', type=float, default=6.0,
                        help='Generation confidence threshold.')
    
    # parser.add_argument('--class_prompt_mode', type=str, choices=['original', 'combined'], default='original',
    #                     help='Class and text prompt mode: original (20 classes) or combined (10 classes)')

    return parser










if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    def maybe_apply_pooling(voxel_pred_np: np.ndarray) -> np.ndarray:
        if not args.apply_majority_pooling:
            return voxel_pred_np

        return apply_majority_pooling(
            voxel_pred_np,
            n_classes=n_classes,
            other_class=other_class,
            empty_class=empty_class,
            is_geometry_only=False,
        )

    semantic_feat_src, semantic_model_type, semantic_family = parse_semantic_mode(args.semantic)
    model_family = "da3" if args.model == "occany_da3" else "must3r"

    if semantic_family is not None:
        if model_family == "must3r":
            assert semantic_family == "SAM2", f"must3r model requires SAM2 semantic family, but got {semantic_family}"
        if model_family == "da3":
            assert semantic_family == "SAM3", f"da3 model requires SAM3 semantic family, but got {semantic_family}"
    from occany.model.attention import toggle_memory_efficient_attention

    toggle_memory_efficient_attention(enabled=True)

    infer_sam2_feats = None
    infer_sam3_feats = None
    infer_semantic_from_boxes_and_sam2_feat_list = None
    infer_semantic_from_classname_and_sam3_inference_state = None
    build_sam3_inference_state = None
    select_sam_feature_views = None
    if semantic_feat_src is not None:
        from occany.semantic_inference import (
            build_sam3_inference_state,
            infer_sam2_feats,
            infer_sam3_feats,
            infer_semantic_from_boxes_and_sam2_feat_list,
            infer_semantic_from_classname_and_sam3_inference_state,
            select_sam_feature_views,
        )

    args.input_dir = os.path.abspath(args.input_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    print(f"RGB demo input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"RGB demo input directory not found: {args.input_dir}")
    frame_dirs = sorted(
        frame_dir for frame_dir in Path(args.input_dir).iterdir() if frame_dir.is_dir()
    )
    if not frame_dirs:
        raise FileNotFoundError(f"No frame directories found in {args.input_dir}")

    sample_image_paths, _ = extract_demo_rgb_images(str(frame_dirs[0]))
    output_resolution = get_output_resolution_from_image(
        sample_image_paths[0],
        model_family=model_family,
    )


    raymap_encoder = None
    gen_decoder = None  # Initialize gen_decoder for all model types
    use_raymap_only_conditioning = False
    checkpoint_args = None
    da3_model_gen = None
    da3_model_recon = None

    if args.model == "occany_must3r":
        from occany.model.model_must3r import Must3r, Dust3rEncoder, RaymapEncoderDiT, Must3rDecoder  # noqa: F401
        from occany.model.must3r_blocks.head import ActivationType  # noqa: F401
        from occany.must3r_inference import inference_occany_gen
        weights_path = REPO_ROOT / "checkpoints" / "occany_must3r.pth"
        if not weights_path.is_file():
            raise FileNotFoundError(
                f"OccAny Must3R checkpoint not found: {weights_path}. "
                "Expected the merged checkpoint at checkpoints/occany_must3r.pth."
            )
        encoder = Dust3rEncoder()
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        checkpoint_args = checkpoint['args']
        decoder = eval(checkpoint_args.decoder)
        
        use_raymap_only_conditioning = getattr(checkpoint_args, 'use_raymap_only_conditioning', False)
        if args.gen:
            print("use_raymap_only_conditioning:", use_raymap_only_conditioning)
            projection_features = getattr(checkpoint_args, 'projection_features', 'pts3d_local,pts3d,rgb,conf,sam')
            print("    Projection features:", projection_features)
            raymap_encoder = RaymapEncoderDiT(
                use_time_cond=False,    
                use_raymap_only_conditioning=use_raymap_only_conditioning,
                projection_features=projection_features,
            )
            raymap_encoder.load_state_dict(checkpoint['raymap_encoder'], strict=False)
        print("Loaded model from", weights_path)
        encoder.load_state_dict(checkpoint['encoder'], strict=False)
        decoder.load_state_dict(checkpoint['decoder'], strict=False)
       
        # Load gen_decoder if it exists in checkpoint (double decoder setup)
        if 'gen_decoder' in checkpoint:
            print("Loading gen_decoder from checkpoint")
            gen_decoder = eval(checkpoint['args'].decoder)
            gen_decoder.load_state_dict(checkpoint['gen_decoder'], strict=False)
            gen_decoder.pointmaps_activation = checkpoint['args'].pointmaps_activation
            gen_decoder.to(args.device)
            gen_decoder.eval()
        
        if args.gen and 'raymap_encoder' in checkpoint:
            raymap_encoder.load_state_dict(checkpoint['raymap_encoder'], strict=False)
        decoder.pointmaps_activation = checkpoint['args'].pointmaps_activation
        print("Set pointmaps_activation to", decoder.pointmaps_activation)
        del checkpoint

        encoder.to(args.device)
        decoder.to(args.device)
        if args.gen:
            raymap_encoder.to(args.device)
            raymap_encoder.eval()
        encoder.eval()
        decoder.eval()
    elif args.model == "occany_da3":
        from occany.da3_inference import inference_occany_da3, inference_occany_da3_gen

        
        gen_weights = REPO_ROOT / "checkpoints" / "occany_da3_gen.pth"
        recon_weights = REPO_ROOT / "checkpoints" / "occany_da3_recon.pth"
        print("[INFO] Preparing DA3 model(s)")
        da3_model_input_size = max(output_resolution)
        da3_model_gen, da3_model_recon, checkpoint_args = setup_da3_models(
            recon_model_path=recon_weights,
            gen_model_path=gen_weights,
            output_resolution=(da3_model_input_size, da3_model_input_size),
            semantic_feat_src=semantic_feat_src,
            semantic_family=semantic_family,
            device=args.device,
            use_generation=args.gen,
        )
    else:
        raise ValueError(f"Model {args.model} not supported")

    sam_model_for_inference = "SAM2"
    if semantic_family == "SAM3":
        sam_model_for_inference = "SAM3"
    elif checkpoint_args is not None:
        checkpoint_sam_model = getattr(checkpoint_args, "sam_model", None)
        if isinstance(checkpoint_sam_model, str) and checkpoint_sam_model.upper() in ["SAM2", "SAM3"]:
            sam_model_for_inference = checkpoint_sam_model.upper()
    

    if args.model == "occany_must3r":
        modules = [m for m in [encoder, decoder, raymap_encoder] if m is not None]
        if gen_decoder is not None:
            modules.append(gen_decoder)
        total_params = sum(p.numel() for m in modules for p in m.parameters())
        trainable_params = sum(
            p.numel() for m in modules for p in m.parameters() if p.requires_grad
        )
        extra = "+gen_decoder" if gen_decoder is not None else ""
        print(
            f"Model 'occany_must3r' (encoder+decoder+raymap_encoder{extra}) - "
            f"total parameters: {total_params:,}, trainable parameters: {trainable_params:,}"
        )
    elif args.model == "occany_da3":
        total_params = sum(p.numel() for p in da3_model_gen.parameters())
        trainable_params = sum(p.numel() for p in da3_model_gen.parameters() if p.requires_grad)
        primary_model_label = "occany_da3_gen" if args.gen else "occany_da3_recon"
        print(
            f"Model '{primary_model_label}' - total parameters: {total_params:,}, "
            f"trainable parameters: {trainable_params:,}"
        )
        if args.gen and da3_model_recon is not None and da3_model_recon is not da3_model_gen:
            recon_total_params = sum(p.numel() for p in da3_model_recon.parameters())
            recon_trainable_params = sum(
                p.numel() for p in da3_model_recon.parameters() if p.requires_grad
            )
            print(
                f"Model 'occany_da3_recon' - total parameters: {recon_total_params:,}, "
                f"trainable parameters: {recon_trainable_params:,}"
            )

  
  

    recon_conf_thres = args.recon_conf_thres
    gen_conf_thres = args.gen_conf_thres
    print(f"recon_conf_thres: {recon_conf_thres}")
    print(f"gen_conf_thres:   {gen_conf_thres}")

    voxel_size = 0.4
    occ_size = [200, 200, 24]
    voxel_origin = torch.tensor([-40.0, -40.0, -3.6], device=args.device, dtype=torch.float32)
    
    print("voxel_size:", voxel_size)
    print("occ_size:", occ_size)
    print("voxel_origin:", voxel_origin)
    print(f"Found {len(frame_dirs)} RGB demo frame directories in {args.input_dir}")
    print("output_resolution (first frame):", output_resolution)

    if not args.gen:
        raymap_encoder = None

    item_count = 0
    T_cam_to_voxel = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=args.device,
    )

    for frame_dir in tqdm(frame_dirs, desc=f"Processing RGB demo frames"):
        demo_image_paths, demo_frame_id = extract_demo_rgb_images(str(frame_dir))
        frame_output_resolution = get_output_resolution_from_image(
            demo_image_paths[0],
            model_family=model_family,
        )
        recon_views = build_demo_reconstruction_views(
            image_paths=demo_image_paths,
            output_resolution=frame_output_resolution,
            model_family=model_family,
            semantic_family=semantic_family,
            frame_interval=args.frame_interval,
            sam3_resolution=args.sam3_resolution,
            device=args.device,
        )
        print(f"Loaded RGB demo frame '{demo_frame_id}' with {len(recon_views)} views from {frame_dir}")
        data = {
            "frame_id": [demo_frame_id],
        }
        B = recon_views[0]["img"].shape[0]
        _, C, H, W = recon_views[0]["img"].shape

        if semantic_family == "SAM2" and args.compute_segmentation_masks:
            box_summary = populate_demo_sam2_box_dicts(
                recon_views=recon_views,
                class_names=CLASS_NAMES,
                device=args.device,
            )
            if box_summary["total_boxes"] == 0:
                print(
                    "[WARNING] GroundingDINO produced no SAM2 demo boxes. "
                    "Semantic outputs will remain empty unless detections are found."
                )
            else:
                print(
                    f"[INFO] Prepared {box_summary['total_boxes']} SAM2 demo boxes "
                    f"across {len(recon_views)} reconstruction views"
                )
            if box_summary["views_without_boxes"]:
                print(
                    "[WARNING] No SAM2 demo boxes detected for reconstruction views: "
                    f"{box_summary['views_without_boxes']}"
                )

        with torch.inference_mode():
            x_ray = None
            sam_feats = None
            sam_feats_raymap = None
            recon_2_gen_mapping = None
            generated_output_c2w = None

            if model_family == "da3":
                recon_model_to_use = da3_model_recon if da3_model_recon is not None else da3_model_gen
                force_pose_from_depth_ray_for_da3_gen = args.gen
                pose_from_depth_ray_for_da3 = args.pose_from_depth_ray or force_pose_from_depth_ray_for_da3_gen
                if force_pose_from_depth_ray_for_da3_gen and not args.pose_from_depth_ray and not getattr(args, "_logged_da3_pose_override", False):
                    print(
                        "[INFO] DA3 generation enabled: forcing pose_from_depth_ray=True "
                        "to generate novel views from predicted reconstruction poses."
                    )
                    args._logged_da3_pose_override = True
                recon_output = inference_occany_da3(
                    recon_views,
                    recon_model_to_use,
                    args.device,
                    dtype=torch.float32,
                    sam_model=sam_model_for_inference,
                    pose_from_depth_ray=pose_from_depth_ray_for_da3,
                    point_from_depth_and_pose=args.point_from_depth_and_pose,
                )
                recon_output.pop('aux_feats', None)
                recon_output.pop('aux_outputs', None)

                projection_features = getattr(da3_model_gen, "projection_features", "")
                needs_sam3_projection_for_gen = uses_sam3_projection_features(projection_features)
                if args.gen and needs_sam3_projection_for_gen and recon_output.get("sam_feats") is None:
                    raise RuntimeError(
                        "Generation checkpoint expects 'sam3' projection features, but reconstruction output "
                        "does not provide distilled SAM3 features. Ensure SAM3 head is initialized for the "
                        "reconstruction model."
                    )

                img_out = convert_da3_output_to_occany_format(recon_output)
                if args.key_to_get_pts3d not in img_out:
                    img_out[args.key_to_get_pts3d] = img_out['pts3d']
                sam_feats = recon_output.get('sam_feats')
                raymap_out = None

                if args.gen:
                    pred_recon_camera_poses = recon_output.get("c2w")
                    if pred_recon_camera_poses is None:
                        raise RuntimeError(
                            "DA3 generation requires predicted reconstruction poses, but recon_output['c2w'] is missing. "
                            "Ensure pose estimation from depth/ray is enabled."
                        )
                    if pred_recon_camera_poses.shape[-2:] == (3, 4):
                        bottom_row = torch.tensor(
                            [0.0, 0.0, 0.0, 1.0],
                            device=pred_recon_camera_poses.device,
                            dtype=pred_recon_camera_poses.dtype,
                        )
                        bottom_row = bottom_row.view(1, 1, 1, 4).expand(
                            pred_recon_camera_poses.shape[0],
                            pred_recon_camera_poses.shape[1],
                            1,
                            4,
                        )
                        pred_recon_camera_poses = torch.cat([pred_recon_camera_poses, bottom_row], dim=-2)
                   
                    gen_poses, recon_2_gen_mapping = generate_intermediate_poses(
                        pred_recon_camera_poses,
                        args.views_per_interval,
                        args.device,
                        forward=args.gen_forward_novel_poses_dist,
                        rotate_angle=args.gen_rotate_novel_poses_angle,
                        num_seed_rotations=args.num_seed_rotations,
                        seed_rotation_angle=args.seed_rotation_angle,
                        seed_translation_distance=args.seed_translation_distance,
                    )
                    gen_poses = gen_poses.float()
                    generated_output_c2w = gen_poses
                    gen_views = []
                    for gen_idx in range(gen_poses.shape[1]):
                        gen_views.append(
                            {
                                'camera_pose': gen_poses[:, gen_idx],
                                'true_shape': recon_views[0]['true_shape'],
                                'is_raymap': True,
                            }
                        )

                    keep_gen_sam_feats = (
                        args.semantic is not None
                        and semantic_family == "SAM3"
                        and (
                            is_distill_source(semantic_feat_src)
                            or (
                                semantic_feat_src == "pretrained"
                                and args.gen_semantic_from_distill_sam3
                            )
                        )
                    )
                    gen_output = inference_occany_da3_gen(
                        recon_output=recon_output,
                        img_views=recon_views,
                        gen_views=gen_views,
                        model=da3_model_gen,
                        device=args.device,
                        dtype=torch.float32,
                        pose_from_depth_ray=pose_from_depth_ray_for_da3,
                        point_from_depth_and_pose=args.point_from_depth_and_pose,
                        gen_batch_size=max(1, int(args.batch_gen_view)),
                        keep_aux_feats=False,
                        keep_sam_feats=keep_gen_sam_feats,
                    )
                    gen_output.pop('aux_feats', None)
                    gen_output.pop('aux_outputs', None)

                    raymap_out = convert_da3_output_to_occany_format(
                        gen_output,
                        fallback_focal=img_out['focal'],
                    )
                    if args.key_to_get_pts3d not in raymap_out:
                        raymap_out[args.key_to_get_pts3d] = raymap_out['pts3d']
                    sam_feats_raymap = gen_output.get('sam_feats')
            else:
                img_out, raymap_out, x_ray, sam_feats, sam_feats_raymap, recon_2_gen_mapping = inference_occany_gen(
                    recon_views,
                    None,
                    raymap_encoder,
                    encoder,
                    decoder,
                    gen_decoder,
                    decoder.pointmaps_activation,
                    args.device,
                    gen_rotate_novel_poses_angle=args.gen_rotate_novel_poses_angle,
                    gen_novel_poses=args.gen,
                    pred_raymap=args.gen,
                    views_per_interval=args.views_per_interval,
                    gen_forward_novel_poses_dist=args.gen_forward_novel_poses_dist,
                    num_seed_rotations=args.num_seed_rotations,
                    seed_rotation_angle=args.seed_rotation_angle,
                    seed_translation_distance=args.seed_translation_distance,
                    use_local_points_with_pose_as_pts3d=False,
                    use_raymap_only_conditioning=use_raymap_only_conditioning,
                    raymap_batch_size=args.batch_gen_view,
                    key_to_get_pts3d=args.key_to_get_pts3d,
                    dtype=torch.float32,
                    sam_model=sam_model_for_inference,
                )

            sam_feats_img_and_raymap = None
            sam3_recon_distill_feats = sam_feats[:3] if sam_feats is not None else None
            sam3_gen_distill_feats = sam_feats_raymap[:3] if sam_feats_raymap is not None else None
            if semantic_family == "SAM2":
                if sam_feats is not None and sam_feats_raymap is not None:
                    sam_feats_img_and_raymap = [
                        torch.cat([sam_feats[level_idx], sam_feats_raymap[level_idx]], dim=1)
                        for level_idx in range(min(len(sam_feats), len(sam_feats_raymap)))
                    ]
                elif sam_feats is not None:
                    sam_feats_img_and_raymap = sam_feats
            


        # res_0 = img_out_0
        res = img_out
        
        # gt_views = batch_result['gt_img']
        # imgs = batch_result['img_input']
        imgs = [v['img'] for v in recon_views]
        imgs = torch.stack(imgs, dim=1)
        if model_family == "da3":
            imgs = denormalize_da3_imgs_to_minus1_1(imgs)

        # pts3d_0 = res_0[args.key_to_get_pts3d]
        # pts3d_local_0 = res_0['pts3d_local']
        # conf_0 = res_0['conf']
        
        recon_semantic_2ds = None
        gen_semantic_2ds = None
        sam2_feats_batch = []
        if args.semantic is not None:
            feat_src = semantic_feat_src
            n_recon_views = len(recon_views)
            n_gen_views = 0 if raymap_out is None else raymap_out['pts3d'].shape[1]
            n_recon_and_gen_views = n_recon_views + n_gen_views

            semantic_fill_value = empty_class
            other_class =  other_class


            semantic_2ds = torch.full(
                (B, n_recon_and_gen_views, H, W),
                semantic_fill_value,
                dtype=torch.uint8,
            )

            if semantic_family == "SAM2":
                sam2_model_type = semantic_model_type
                sam2_imgs_recon = None
                if feat_src == 'pretrained':
                    if all('sam2_img' in view for view in recon_views):
                        sam2_imgs_recon = torch.stack([view['sam2_img'] for view in recon_views], dim=1)
                    else:
                        print("[WARNING] SAM2 pretrained mode requested but recon views do not contain sam2_img")

                if args.compute_segmentation_masks:
                    class_names = CLASS_NAMES
                    class2idx = {name: idx for idx, name in enumerate(class_names)}
                    ignore_ids = {empty_class, other_class, 255}

                    for batch_i in range(B):
                        if feat_src == 'pretrained':
                            if sam2_imgs_recon is None:
                                continue
                            sam2_feats = infer_sam2_feats(sam2_model_type, sam2_imgs_recon[batch_i], args.device)
                        elif is_distill_source(feat_src):
                            if sam_feats_img_and_raymap is None or len(sam_feats_img_and_raymap) < 3:
                                print(
                                    "[WARNING] SAM2 distill mode requested but distilled SAM features are unavailable"
                                )
                                continue
                            sam2_feats = {
                                "image_embed": sam_feats_img_and_raymap[0][batch_i],
                                "high_res_feats": [
                                    sam_feats_img_and_raymap[2][batch_i],
                                    sam_feats_img_and_raymap[1][batch_i],
                                ],
                            }
                        else:
                            raise ValueError(f"Unknown SAM2 feature source: {feat_src}")

                        sam2_feats_batch.append(sam2_feats)

                        for recon_view_i in range(n_recon_views):
                            box_dict = recon_views[recon_view_i]['box_dict'][batch_i]
                            boxes = box_dict['boxes']
                            confidences = box_dict['confidences']
                            labels = box_dict['labels']

                            valid_indices = [idx for idx, label in enumerate(labels) if label in class2idx]
                            if len(valid_indices) == 0:
                                continue

                            boxes_np = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
                            conf_np = (
                                confidences.detach().cpu().numpy()
                                if torch.is_tensor(confidences)
                                else np.asarray(confidences)
                            )
                            if boxes_np.size == 0:
                                continue
                            boxes_np = boxes_np.reshape(-1, 4)[valid_indices]
                            conf_np = conf_np.reshape(-1)[valid_indices]
                            label_ids = [class2idx[labels[idx]] for idx in valid_indices]

                            if args.gen:
                                if recon_2_gen_mapping is not None and recon_view_i in recon_2_gen_mapping:
                                    corresponding_gen_view_ids = [
                                        view_idx + n_recon_views
                                        for view_idx in recon_2_gen_mapping[recon_view_i]
                                    ]
                                    if args.no_semantic_from_rotated_views and args.gen_rotate_novel_poses_angle > 0:
                                        n_total_gen = len(corresponding_gen_view_ids)
                                        n_straight = n_total_gen // 3
                                        corresponding_gen_view_ids = corresponding_gen_view_ids[:n_straight]
                                else:
                                    corresponding_gen_view_ids = []
                                if args.only_semantic_from_recon_view:
                                    corresponding_gen_view_ids = []
                            else:
                                corresponding_gen_view_ids = []

                            for gen_view_i in range(
                                0,
                                max(1, len(corresponding_gen_view_ids)),
                                args.batch_gen_view,
                            ):
                                recon_and_gen_ids = [recon_view_i] + corresponding_gen_view_ids[
                                    gen_view_i:gen_view_i + args.batch_gen_view
                                ]

                                sam2_feat_list = []
                                for view_id in recon_and_gen_ids:
                                    sam2_feat_list.append(
                                        {
                                            "high_res_feats": [
                                                sam2_feats['high_res_feats'][0][view_id:view_id + 1],
                                                sam2_feats['high_res_feats'][1][view_id:view_id + 1],
                                            ],
                                            "image_embed": sam2_feats['image_embed'][view_id:view_id + 1],
                                        }
                                    )

                                sem2d = infer_semantic_from_boxes_and_sam2_feat_list(
                                    sam2_model_type,
                                    H,
                                    W,
                                    label_ids,
                                    ignore_ids,
                                    boxes_np,
                                    conf_np,
                                    other_class=other_class,
                                    empty_class=empty_class,
                                    use_sam_video=True,
                                    sam2_feats_list=sam2_feat_list,
                                    poses=None,
                                    focals=None,
                                    depth_maps=None,
                                    device=args.device,
                                    box_conf_thres=args.box_conf_thres,
                                    merge_masks=args.merge_masks,
                                )

                                for local_idx, view_i in enumerate(recon_and_gen_ids):
                                    semantic_2ds[batch_i, view_i] = torch.from_numpy(sem2d[local_idx])

                                del sam2_feat_list, sem2d
                                torch.cuda.empty_cache()

                recon_semantic_2ds = semantic_2ds[:, :n_recon_views]
                gen_semantic_2ds = semantic_2ds[:, n_recon_views:] if n_gen_views > 0 else None
                if args.compute_segmentation_masks and recon_semantic_2ds is not None:
                    if bool((recon_semantic_2ds == empty_class).all().item()):
                        print(
                            "[WARNING] SAM2 reconstruction semantics remained entirely empty "
                            f"(class {empty_class}) after inference"
                        )
                    if gen_semantic_2ds is not None and bool((gen_semantic_2ds == empty_class).all().item()):
                        print(
                            "[WARNING] SAM2 generated-view semantics remained entirely empty "
                            f"(class {empty_class}) after inference"
                        )
            elif semantic_family == "SAM3":
                recon_semantic_2ds = semantic_2ds[:, :n_recon_views]
                gen_semantic_2ds = semantic_2ds[:, n_recon_views:] if n_gen_views > 0 else None

                if not args.compute_segmentation_masks:
                    pass
                else:
                    prompts, prompt_to_class_mapping = build_fine_prompt_metadata(PROMPT)
                    ignore_ids = {empty_class, other_class, 255}
                    allowed_gen_view_ids = get_allowed_gen_view_ids(
                        n_gen_views=n_gen_views,
                        recon_2_gen_mapping=recon_2_gen_mapping,
                        only_semantic_from_recon_view=args.only_semantic_from_recon_view,
                        no_semantic_from_rotated_views=args.no_semantic_from_rotated_views,
                        gen_rotate_novel_poses_angle=args.gen_rotate_novel_poses_angle,
                    )

                    recon_distill_feats = sam3_recon_distill_feats
                    gen_distill_feats = sam3_gen_distill_feats

                    sam3_imgs_recon = None
                    if feat_src == 'pretrained':
                        if all('sam3_img' in view for view in recon_views):
                            sam3_imgs_recon = torch.stack([view['sam3_img'] for view in recon_views], dim=1)
                        else:
                            print("[WARNING] SAM3 pretrained mode requested but recon views do not contain sam3_img")
                    elif not is_distill_source(feat_src):
                        raise ValueError(f"Unknown SAM3 feature source: {feat_src}")

                    if is_distill_source(feat_src) and recon_distill_feats is None:
                        print("[WARNING] SAM3 distill mode requested but reconstruction distilled features are unavailable")

                    allow_pretrained_gen_sam3_from_distill = (
                        feat_src == 'pretrained' and args.gen_semantic_from_distill_sam3
                    )

                    if (
                        feat_src == 'pretrained'
                        and n_gen_views > 0
                        and len(allowed_gen_view_ids) > 0
                        and not args.only_semantic_from_recon_view
                        and allow_pretrained_gen_sam3_from_distill
                        and gen_distill_feats is None
                    ):
                        print(
                            "[WARNING] pretrained@SAM3 cannot infer generated-view semantics because "
                            "distilled generated SAM3 features are unavailable"
                        )

                    if n_gen_views > 0 and len(allowed_gen_view_ids) == 0 and not args.only_semantic_from_recon_view:
                        print("[WARNING] No generated views selected for SAM3 semantics after view filtering")

                    can_infer_gen_sam3 = (
                        gen_semantic_2ds is not None
                        and not args.only_semantic_from_recon_view
                        and len(allowed_gen_view_ids) > 0
                        and gen_distill_feats is not None
                        and (
                            is_distill_source(feat_src)
                            or allow_pretrained_gen_sam3_from_distill
                        )
                    )

                    selected_gen_view_ids = allowed_gen_view_ids
                    n_gen_views_for_sam3 = n_gen_views
                    if can_infer_gen_sam3 and len(selected_gen_view_ids) < n_gen_views:
                        selected_gen_distill_feats = select_sam_feature_views(
                            gen_distill_feats,
                            selected_gen_view_ids,
                            n_gen_views,
                            context="gen_sam3_distill_subset",
                        )
                        if selected_gen_distill_feats is None:
                            print("[WARNING] Failed to build selected generated-view SAM3 features")
                            can_infer_gen_sam3 = False
                        else:
                            gen_distill_feats = selected_gen_distill_feats
                            n_gen_views_for_sam3 = len(selected_gen_view_ids)
                    elif can_infer_gen_sam3:
                        n_gen_views_for_sam3 = len(selected_gen_view_ids)

                    from sam3.model.position_encoding import PositionEmbeddingSine

                    pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True)
                    for batch_i in range(B):
                        if feat_src == 'pretrained':
                            if sam3_imgs_recon is None:
                                continue
                            recon_state = infer_sam3_feats(
                                sam3_imgs_recon[batch_i],
                                H,
                                W,
                                args.device,
                                args.sam3_resolution,
                            )
                        else:
                            recon_state = build_sam3_inference_state(
                                recon_distill_feats,
                                batch_i,
                                n_recon_views,
                                H,
                                W,
                                pos_enc,
                                context="recon_sam3_distill",
                            )

                        if recon_state is not None:
                            recon_semantic_2ds[batch_i] = infer_semantic_from_classname_and_sam3_inference_state(
                                prompts,
                                prompt_to_class_mapping,
                                recon_state,
                                ignore_ids,
                                empty_class,
                                args.device,
                                args.sam3_conf_th,
                                args.sam3_resolution,
                                args.view_batch_size,
                            )

                        if not can_infer_gen_sam3:
                            continue

                        gen_state_context = "gen_sam3_distill"
                        if feat_src == 'pretrained':
                            gen_state_context = "gen_sam3_distill_for_pretrained"

                        gen_state = build_sam3_inference_state(
                            gen_distill_feats,
                            batch_i,
                            n_gen_views_for_sam3,
                            H,
                            W,
                            pos_enc,
                            context=gen_state_context,
                        )
                        if gen_state is None:
                            continue
                        gen_semantics = infer_semantic_from_classname_and_sam3_inference_state(
                            prompts,
                            prompt_to_class_mapping,
                            gen_state,
                            ignore_ids,
                            empty_class,
                            args.device,
                            args.sam3_conf_th,
                            args.sam3_resolution,
                            args.view_batch_size,
                        )
                        if gen_semantics.shape[0] != len(selected_gen_view_ids):
                            print(
                                "[WARNING] Generated SAM3 semantics shape mismatch: "
                                f"expected {len(selected_gen_view_ids)} views, got {gen_semantics.shape[0]}"
                            )
                            continue
                        for local_view_idx, view_idx in enumerate(selected_gen_view_ids):
                            gen_semantic_2ds[batch_i, view_idx] = gen_semantics[local_view_idx]
            else:
                raise ValueError(f"Unknown semantic family: {semantic_family}")
       
        
        outputs = {}
      

        pts3d_render = res[args.key_to_get_pts3d]
        pts3d_local_render = res['pts3d_local']
        conf_render = res['conf']
        outputs["render"] = {
            "pts3d": pts3d_render,
            "pts3d_local": pts3d_local_render,
            "conf": conf_render,
            "colors": imgs,
            "focal": res['focal'],
            "c2w": res['c2w'],
            "estimated_camera_poses": res['c2w_pose'] if 'c2w_pose' in res else res['c2w'],
            "semantic_2ds": recon_semantic_2ds,
            "is_recon": torch.ones(B, pts3d_render.shape[1], dtype=torch.bool, device=pts3d_render.device),
            # "c2w_pose": res['c2w_pose']
        }
            

        if args.gen and raymap_out is not None:
            pts3d_gen = raymap_out[args.key_to_get_pts3d]
            pts3d_local_gen = raymap_out['pts3d_local']
            conf_gen = raymap_out['conf']
            render_gen_c2w = generated_output_c2w
            if render_gen_c2w is None:
                render_gen_c2w = raymap_out.get('c2w_input')
            if render_gen_c2w is None:
                render_gen_c2w = raymap_out['c2w']

            outputs["render_gen"] = {
                "pts3d": pts3d_gen,
                "pts3d_local": pts3d_local_gen,
                "conf": conf_gen,
                "colors": torch.zeros(B, pts3d_gen.shape[1], 3, H, W, device=pts3d_gen.device),
                "focal": raymap_out['focal'],
                "c2w": render_gen_c2w,
                "semantic_2ds": gen_semantic_2ds,
                "is_recon": torch.zeros(B, pts3d_gen.shape[1], dtype=torch.bool, device=pts3d_gen.device),
                # "c2w_pose": gen_out['c2w_pose']
            }

            outputs['render_recon_gen'] = {
                "pts3d": torch.cat([outputs['render']['pts3d'], outputs['render_gen']['pts3d']], dim=1),
                "pts3d_local": torch.cat([outputs['render']['pts3d_local'], outputs['render_gen']['pts3d_local']], dim=1),
                "conf": torch.cat([outputs['render']['conf'], outputs['render_gen']['conf']], dim=1),
                "colors": torch.cat([outputs['render']['colors'], outputs['render_gen']['colors']], dim=1),
                "focal": torch.cat([outputs['render']['focal'], outputs['render_gen']['focal']], dim=1),
                "c2w": torch.cat([outputs['render']['c2w'], outputs['render_gen']['c2w']], dim=1),
                "semantic_2ds": (
                    torch.cat([outputs['render']['semantic_2ds'], outputs['render_gen']['semantic_2ds']], dim=1)
                    if outputs['render']['semantic_2ds'] is not None and outputs['render_gen']['semantic_2ds'] is not None
                    else None
                ),
                "is_recon": torch.cat([outputs['render']['is_recon'], outputs['render_gen']['is_recon']], dim=1),
                # "c2w_pose": torch.cat([outputs['render']['c2w_pose'], outputs['render_gen']['c2w_pose']], dim=1)
            }
        elif args.gen:
            print("[WARNING] Generation was requested but no generated views were produced")
            
            
        
        
        for j in tqdm(range(B), leave=False):

            frame_id = data['frame_id'][j]
            voxel_pred_save_dir = os.path.join(args.output_dir, f"{frame_id}_{args.model}")
            os.makedirs(voxel_pred_save_dir, exist_ok=True)

            for name, output in outputs.items():
                has_semantic_output = args.semantic is not None and output.get("semantic_2ds") is not None
                colors_hwc = output['colors'][j].permute(0, 2, 3, 1).cpu().numpy()
                save_dict = {
                    "pts3d": output['pts3d'][j].cpu().numpy(),
                    "pts3d_local": output['pts3d_local'][j].cpu().numpy(),
                    "colors": colors_hwc,
                    "conf": output['conf'][j].cpu().numpy(),
                    "focal": output['focal'][j].cpu().numpy(),
                    "c2w": output['c2w'][j].cpu().numpy(),
                }
                if has_semantic_output:
                    save_dict["semantic_2ds"] = output["semantic_2ds"][j].cpu().numpy()

                save_path = os.path.join(voxel_pred_save_dir, f"pts3d_{name}.npy")
                np.save(save_path, save_dict)

            grid_size = tuple(occ_size)
            voxel_predictions_dict = {
                "estimated_input_camera_poses": outputs['render']['estimated_camera_poses'][j].cpu().numpy(),
                "estimated_input_intrinsics": build_intrinsics_from_focal(
                    outputs['render']['focal'][j],
                    H,
                    W,
                ).cpu().numpy(),
                "estimated_input_images": convert_images_to_uint8_hwc(outputs['render']['colors'][j]),
                "voxel_size": voxel_size,
                "voxel_origin": voxel_origin.cpu().numpy(),
            }

            recon_output = outputs['render']

            # Process render (reconstruction) output
            render_conf_mask = recon_output['conf'][j] > recon_conf_thres
            render_pts3d_th = recon_output['pts3d'][j][render_conf_mask]
            render_conf_th = recon_output['conf'][j][render_conf_mask]

            if args.semantic is not None:
                render_semantic_2ds_th = recon_output.get('semantic_2ds', [None])[j]
                if render_semantic_2ds_th is not None:
                    render_semantic_2ds_th = render_semantic_2ds_th.to(render_conf_mask.device)
                    render_semantic_2ds_th = render_semantic_2ds_th[render_conf_mask]
                    render_has_semantic = True
                else:
                    render_has_semantic = False
                    render_semantic_2ds_th = None
            else:
                render_has_semantic = False
                render_semantic_2ds_th = None

            # Create and save render voxel prediction
            render_pts3d_in_velo = transform_points_torch(T=T_cam_to_voxel.float(), points=render_pts3d_th)

            render_voxel_pred = create_voxel_prediction(
                render_pts3d_in_velo, render_has_semantic, render_semantic_2ds_th, render_conf_th,
                grid_size, voxel_origin, voxel_size,
                n_classes, other_class, empty_class
            )
            render_voxel_pred_np = render_voxel_pred.cpu().numpy().astype(np.uint8)
            
            voxel_predictions_dict[f"render_th{recon_conf_thres}"] = render_voxel_pred_np
            
            # If render_gen exists, also create and save combined output
            if 'render_gen' in outputs:
                gen_output = outputs['render_gen']

                # Filter generation output
                gen_conf_mask = gen_output['conf'][j] > gen_conf_thres
                gen_pts3d_th = gen_output['pts3d'][j][gen_conf_mask]
                gen_conf_th = gen_output['conf'][j][gen_conf_mask]

                if args.semantic is not None:
                    gen_semantic_2ds_th = gen_output.get('semantic_2ds', [None])[j]
                    if gen_semantic_2ds_th is not None:
                        gen_semantic_2ds_th = gen_semantic_2ds_th.to(gen_conf_mask.device)
                        gen_semantic_2ds_th = gen_semantic_2ds_th[gen_conf_mask]
                        gen_has_semantic = True
                    else:
                        gen_has_semantic = False
                        gen_semantic_2ds_th = None
                else:
                    gen_has_semantic = False
                    gen_semantic_2ds_th = None

                if gen_pts3d_th.numel() == 0:
                    gen_voxel_pred = torch.full(
                        grid_size,
                        empty_class,
                        dtype=torch.long,
                        device=render_voxel_pred.device,
                    )
                else:
                    gen_pts3d_in_velo = transform_points_torch(
                        T=T_cam_to_voxel.float(),
                        points=gen_pts3d_th,
                    )
                    gen_voxel_pred = create_voxel_prediction(
                        gen_pts3d_in_velo,
                        gen_has_semantic,
                        gen_semantic_2ds_th,
                        gen_conf_th,
                        grid_size,
                        voxel_origin,
                        voxel_size,
                        n_classes,
                        other_class,
                        empty_class,
                    )

                voxel_pred = gen_voxel_pred.clone()
                non_empty_mask = render_voxel_pred != empty_class
                voxel_pred[non_empty_mask] = render_voxel_pred[non_empty_mask]

                voxel_pred_np = voxel_pred.cpu().numpy().astype(np.uint8)
                voxel_pred_np = maybe_apply_pooling(voxel_pred_np)
                print("Number of occupied voxels:", np.sum(voxel_pred_np != empty_class))
                key = f"render_recon_gen_recon{recon_conf_thres}_gen{gen_conf_thres}"
                voxel_predictions_dict[key] = voxel_pred_np
                print(f"Added combined voxel prediction: {key}")

            save_path = os.path.join(voxel_pred_save_dir, "voxel_predictions.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(voxel_predictions_dict, f)
            print(f"Saved voxel predictions dictionary: {save_path}")
        
            item_count += 1  # Increment item counter after processing each item in batch
            
            # Clean up tensors from current iteration
            del outputs, data, imgs
            if 'sam2_imgs' in locals() and sam2_imgs is not None:
                del sam2_imgs
            if 'sam3_imgs' in locals() and sam3_imgs is not None:
                del sam3_imgs
            if 'recon_semantic_2ds' in locals() and recon_semantic_2ds is not None:
                del recon_semantic_2ds
            if 'gen_semantic_2ds' in locals() and gen_semantic_2ds is not None:
                del gen_semantic_2ds
            if 'semantic_2ds' in locals():
                del semantic_2ds
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    print("=" * 50)
    print(f"Total items processed: {item_count}")
