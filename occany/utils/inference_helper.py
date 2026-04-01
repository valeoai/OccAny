import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor

from occany.utils.image_util import (
    GroundingDinoImgNorm,
    ImgNorm,
    crop_resize_if_necessary,
    get_SAM2_transforms,
    get_SAM3_transforms,
)

from occany.semantic_inference import infer_sam2_boxes, ModelManager
from depth_anything_3.utils.geometry import affine_inverse
from dust3r.utils.geometry import geotrf        

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")



def count_module_parameters(module: Optional[torch.nn.Module]) -> Tuple[int, int]:
    """Count total and trainable parameters for one module."""
    if module is None:
        return 0, 0
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total_params, trainable_params


def count_unique_parameters(modules: List[Optional[torch.nn.Module]]) -> Tuple[int, int]:
    """Count unique parameters across multiple modules, skipping shared tensors."""
    total_params = 0
    trainable_params = 0
    seen_param_ids = set()
    for module in modules:
        if module is None:
            continue
        for param in module.parameters():
            param_id = id(param)
            if param_id in seen_param_ids:
                continue
            seen_param_ids.add(param_id)
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
    return total_params, trainable_params


def get_pretrained_semantic_encoder_for_count(
    semantic_feat_src: Optional[str],
    semantic_family: Optional[str],
    semantic_model_type: Optional[str],
    device: str,
    image_size: int,
    sam3_resolution: int,
    sam3_conf_th: float,
) -> Tuple[Optional[torch.nn.Module], Optional[str]]:
    """Load pretrained semantic encoder module used for inference parameter reporting."""
    if semantic_feat_src != "pretrained" or semantic_family is None:
        return None, None

    try:
        if semantic_family == "SAM2":
            if semantic_model_type is None:
                return None, None

            model_manager = ModelManager(device)
            sam2_model = model_manager.get_sam2(
                semantic_model_type,
                load_video_model=False,
                image_size=image_size,
            )

            sam2_model_core = getattr(sam2_model, "model", None)
            sam2_encoder = getattr(sam2_model_core, "image_encoder", None)
            if isinstance(sam2_encoder, torch.nn.Module):
                return sam2_encoder, semantic_model_type
            print("[WARNING] Unable to locate SAM2 encoder module for parameter counting")
            return None, semantic_model_type

        if semantic_family == "SAM3":
            sam3_manager = Sam3ModelManager(
                resolution=sam3_resolution,
                confidence_threshold=sam3_conf_th,
            )
            sam3_processor = sam3_manager.get_sam3(device)

            sam3_model = getattr(sam3_processor, "model", None)
            sam3_vl_backbone = getattr(sam3_model, "backbone", None)
            sam3_vision_encoder = getattr(sam3_vl_backbone, "vision_backbone", None)
            if isinstance(sam3_vision_encoder, torch.nn.Module):
                return sam3_vision_encoder, "SAM3_vision_backbone"

            if isinstance(sam3_vl_backbone, torch.nn.Module):
                print(
                    "[WARNING] SAM3 vision_backbone not found; "
                    "falling back to full SAM3 visual-language backbone for counting"
                )
                return sam3_vl_backbone, "SAM3_vl_backbone"

            if isinstance(sam3_model, torch.nn.Module):
                print(
                    "[WARNING] SAM3 backbone not found; "
                    "falling back to full SAM3 model for counting"
                )
                return sam3_model, "SAM3"

            print("[WARNING] Unable to locate SAM3 encoder module for parameter counting")
            return None, "SAM3"

    except Exception as exc:
        print(
            f"[WARNING] Failed to load pretrained {semantic_family} model for parameter counting: {exc}"
        )

    return None, semantic_model_type
    
def parse_semantic_mode(semantic: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse semantic mode into feature source and SAM family."""
    if semantic is None:
        return None, None, None
    if "@" not in semantic:
        raise ValueError(f"Invalid semantic mode '{semantic}'. Expected format '<src>@<model>'")
    feat_src, semantic_model_type = semantic.split("@", 1)
    semantic_family = "SAM3" if "SAM3" in semantic_model_type.upper() else "SAM2"
    return feat_src, semantic_model_type, semantic_family


def is_distill_source(feat_src: Optional[str]) -> bool:
    """Return True if semantic source uses distilled features."""
    return feat_src is not None and feat_src.startswith("distill")


def uses_sam3_projection_features(projection_features: Optional[str]) -> bool:
    """Return True when projection features include SAM3 conditioning."""
    if projection_features is None:
        return False
    feature_set = {
        feature.strip().lower()
        for feature in projection_features.split(",")
        if feature.strip()
    }
    return "sam3" in feature_set


def denormalize_da3_imgs_to_minus1_1(imgs: torch.Tensor) -> torch.Tensor:
    """Convert ImageNet-normalized DA3 inputs back to [-1, 1] RGB."""
    mean = torch.tensor(IMAGENET_MEAN, dtype=imgs.dtype, device=imgs.device).view(1, 1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=imgs.dtype, device=imgs.device).view(1, 1, 3, 1, 1)
    imgs = (imgs * std) + mean
    imgs = imgs.clamp(0.0, 1.0).mul(2.0).sub(1.0)
    return imgs




def get_pts3d_from_voxel(voxel_grid):
    """
    Extract 3D point coordinates from a binary voxel grid where occupied voxels have value 1.

    Parameters
    ----------
    voxel_grid : np.ndarray of shape (H, W, D)
        Binary array with 0 for empty and 1 for occupied voxels.

    Returns
    -------
    pts : np.ndarray of shape (N, 3)
        Array of 3D integer coordinates (x, y, z) for occupied voxels.
    """
    if not isinstance(voxel_grid, np.ndarray):
        raise TypeError("voxel_grid must be a numpy ndarray")
    if voxel_grid.ndim != 3:
        raise ValueError("voxel_grid must be 3D (H, W, D)")
    # Treat any nonzero as occupied
    occupied = (voxel_grid > 0) & (voxel_grid < 255)
    # Indices of occupied cells; returns rows as (i, j, k)
    ijk = np.argwhere(occupied)
    # Return as (x, y, z) == (i, j, k). If a different convention is desired,
    # e.g., (x, y, z) = (j, i, k), reorder columns accordingly.
    return ijk.astype(np.int64)


def build_intrinsics_from_focal(
    focal: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build per-view 3x3 intrinsics matrices from estimated focal lengths."""
    if focal.dim() == 1:
        fx = focal
        fy = focal
    elif focal.dim() == 2:
        if focal.shape[-1] == 1:
            fx = focal[..., 0]
            fy = focal[..., 0]
        elif focal.shape[-1] >= 2:
            fx = focal[..., 0]
            fy = focal[..., 1]
        else:
            raise ValueError(f"Unsupported focal shape: {tuple(focal.shape)}")
    else:
        raise ValueError(f"Unsupported focal rank: {focal.dim()}")

    intrinsics = torch.zeros((focal.shape[0], 3, 3), dtype=focal.dtype, device=focal.device)
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = width / 2.0
    intrinsics[:, 1, 2] = height / 2.0
    intrinsics[:, 2, 2] = 1.0
    return intrinsics




def derive_demo_frame_id(image_paths: List[str], scene_dir: str) -> str:
    prefixes = {
        Path(image_path).stem.rsplit("_", 1)[0]
        for image_path in image_paths
        if "_" in Path(image_path).stem
    }
    if len(prefixes) == 1:
        return prefixes.pop()
    scene_name = os.path.basename(os.path.normpath(scene_dir))
    return scene_name if scene_name else "demo"


def extract_demo_rgb_images(scene_dir: str) -> Tuple[List[str], str]:
    scene_dir = os.path.abspath(scene_dir)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Demo scene directory not found: {scene_dir}")

    image_paths = sorted(
        os.path.join(scene_dir, name)
        for name in os.listdir(scene_dir)
        if name.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(os.path.join(scene_dir, name))
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No demo image files found in {scene_dir}. Expected files like 000000.png."
        )

    frame_id = derive_demo_frame_id(image_paths, scene_dir)
    return image_paths, frame_id

def normalize_demo_rgb_image(image: Image.Image, model_family: str) -> torch.Tensor:
    if model_family == "da3":
        image_tensor = to_tensor(image)
        mean = torch.tensor(IMAGENET_MEAN, dtype=image_tensor.dtype).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=image_tensor.dtype).view(3, 1, 1)
        return (image_tensor - mean) / std
    return ImgNorm(image)



def build_demo_reconstruction_views(
    image_paths: List[str],
    output_resolution: Tuple[int, int],
    model_family: str,
    semantic_family: Optional[str],
    frame_interval: int,
    sam3_resolution: int,
    device: str,
) -> List[Dict[str, Any]]:
    if len(image_paths) == 0:
        raise ValueError("No RGB images were found for the demo scene")

    semantic_transform = None
    if semantic_family == "SAM2":
        semantic_transform = get_SAM2_transforms(resolution=min(1024, max(output_resolution)))
    elif semantic_family == "SAM3":
        semantic_transform = get_SAM3_transforms(resolution=sam3_resolution)

    empty_box_dict = {"boxes": [], "confidences": [], "labels": []}
    recon_views: List[Dict[str, Any]] = []
    output_width, output_height = output_resolution
    true_shape = np.array([[output_height, output_width]], dtype=np.int32)

    for view_idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        resized_image = crop_resize_if_necessary(
            image,
            resolution=output_resolution,
            rng=np.random.default_rng(0),
            info=image_path,
        )
        img_tensor = normalize_demo_rgb_image(resized_image, model_family=model_family).unsqueeze(0).to(device)
        view: Dict[str, Any] = {
            "img": img_tensor,
            "timestep": torch.tensor(
                [float(view_idx * frame_interval)],
                dtype=torch.float32,
                device=device,
            ),
            "true_shape": true_shape.copy(),
            "box_dict": [copy.deepcopy(empty_box_dict)],
        }
        if semantic_family == "SAM2" and semantic_transform is not None:
            view["sam2_img"] = semantic_transform(resized_image).unsqueeze(0).to(device)
            gdino_img, _ = GroundingDinoImgNorm(resized_image, None)
            view["gdino_img"] = gdino_img.unsqueeze(0)
        if semantic_family == "SAM3" and semantic_transform is not None:
            view["sam3_img"] = semantic_transform(resized_image).unsqueeze(0).to(device)
        recon_views.append(view)

    return recon_views


def populate_demo_sam2_box_dicts(
    recon_views: List[Dict[str, Any]],
    class_names: List[str],
    device: str,
    box_threshold: float = 0.15,
    text_threshold: float = 0.0,
) -> Dict[str, Any]:
    if len(recon_views) == 0:
        return {"total_boxes": 0, "views_without_boxes": []}


    total_boxes = 0
    views_without_boxes: List[int] = []

    for view_idx, view in enumerate(recon_views):
        gdino_imgs = view.get("gdino_img")
        if gdino_imgs is None:
            raise ValueError(
                "SAM2 demo box generation requires 'gdino_img' in reconstruction views."
            )

        batch_box_dicts: List[Dict[str, Any]] = []
        for batch_idx in range(gdino_imgs.shape[0]):
            boxes, confidences, labels = infer_sam2_boxes(
                gdino_imgs=gdino_imgs[batch_idx:batch_idx + 1],
                class_names=class_names,
                device=device,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            if len(labels) == 0:
                views_without_boxes.append(view_idx)
            total_boxes += len(labels)
            batch_box_dicts.append(
                {
                    "boxes": boxes,
                    "confidences": confidences,
                    "labels": labels,
                }
            )

        view["box_dict"] = batch_box_dicts

    return {
        "total_boxes": total_boxes,
        "views_without_boxes": sorted(set(views_without_boxes)),
    }


def get_allowed_gen_view_ids(
    n_gen_views: int,
    recon_2_gen_mapping: Optional[Dict[int, List[int]]],
    only_semantic_from_recon_view: bool,
    no_semantic_from_rotated_views: bool,
    gen_rotate_novel_poses_angle: int,
) -> List[int]:
    """Return generated-view ids that should receive semantic labels."""
    if n_gen_views <= 0 or only_semantic_from_recon_view:
        return []

    if (
        not no_semantic_from_rotated_views
        or gen_rotate_novel_poses_angle <= 0
        or recon_2_gen_mapping is None
    ):
        return list(range(n_gen_views))

    allowed_ids: List[int] = []
    for gen_ids in recon_2_gen_mapping.values():
        if len(gen_ids) == 0:
            continue
        n_straight = len(gen_ids) // 3
        if n_straight == 0:
            n_straight = len(gen_ids)
        allowed_ids.extend(gen_ids[:n_straight])

    if len(allowed_ids) == 0:
        return list(range(n_gen_views))
    return sorted(set(idx for idx in allowed_ids if 0 <= idx < n_gen_views))



def convert_da3_output_to_occany_format(
    da3_output: Dict[str, torch.Tensor],
    fallback_focal: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Convert DA3 output dict to the schema used in this extraction script."""


    pts3d = da3_output["pointmap"]
    conf = da3_output["depth_conf"]
    B, T, H, W, _ = pts3d.shape

    c2w = da3_output.get("c2w")
    if c2w is None:
        c2w = torch.eye(4, device=pts3d.device, dtype=pts3d.dtype).view(1, 1, 4, 4).expand(B, T, 4, 4).clone()
    elif c2w.shape[-2:] == (3, 4):
        bottom_row = torch.tensor([0.0, 0.0, 0.0, 1.0], device=c2w.device, dtype=c2w.dtype)
        bottom_row = bottom_row.view(1, 1, 1, 4).expand(c2w.shape[0], c2w.shape[1], 1, 4)
        c2w = torch.cat([c2w, bottom_row], dim=-2)

    # Compute pts3d_local: transform world-space points into each camera's frame
    w2c = affine_inverse(c2w)
    pts3d_local = geotrf(w2c, pts3d.reshape(B, T, -1, 3))
    pts3d_local = pts3d_local.reshape(B, T, H, W, 3)

    intrinsics = da3_output.get("intrinsics")
    if intrinsics is not None:
        focal = torch.stack([intrinsics[:, :, 0, 0], intrinsics[:, :, 1, 1]], dim=-1)
    elif fallback_focal is not None:
        if fallback_focal.dim() == 3:
            base_focal = fallback_focal[:, 0]
        else:
            base_focal = fallback_focal
        focal = base_focal.unsqueeze(1).expand(B, T, 2)
    else:
        focal = torch.ones((B, T, 2), device=pts3d.device, dtype=pts3d.dtype)
    
    return {
        "pts3d": pts3d,
        "pts3d_local": pts3d_local,
        "conf": conf,
        "focal": focal,
        "c2w": c2w,
        "c2w_input": c2w,
        "depth": da3_output["depth"],
    }
