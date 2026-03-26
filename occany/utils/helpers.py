import numpy as np
try:
    import occany.utils.fusion as fusion
    _FUSION_AVAILABLE = True
except Exception as _e:
    fusion = None
    _FUSION_AVAILABLE = False
import torch
import cv2
from occany.utils.image_util import colorize_depth_maps, chw2hwc
from occany.utils import cropping
from PIL import Image
import os
from einops import rearrange
from scipy.spatial.transform import Rotation as R, Slerp
import torch.nn.functional as F
from typing import Optional
from depth_anything_3.utils.geometry import (
    as_homogeneous,
    homogenize_vectors,
    transform_cam2world,
    unproject
)

def get_world_rays(
    coordinates: torch.Tensor,
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute world-space ray origins and directions from pixel coordinates.
    
    Args:
        coordinates: Pixel coordinates (*#batch, 2)
        extrinsics: Camera-to-world matrices (*#batch, 4, 4)
        intrinsics: Camera intrinsics (*#batch, 3, 3)
        normalize: If True, normalize directions to unit length. 
                   If False, directions have z=1 in camera space (suitable for z-depth).
    
    Returns:
        origins: Ray origins in world space (*batch, 3)
        directions: Ray directions in world space (*batch, 3)
    """
    # Get camera-space ray directions (with z=1)
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    if normalize:
        directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Get ray origins from extrinsics translation
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions




def intrinsics_c2w_to_raymap(
    intrinsics: torch.Tensor,  # (*batch, 3, 3)
    c2w: torch.Tensor,         # (*batch, 4, 4) or (*batch, 3, 4) - camera-to-world (NOT world-to-camera!)
    height: int,
    width: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:  # (*batch, H, W, 6)
    """
    Convert camera intrinsics and camera-to-world (c2w) extrinsics to a raymap.
    
    IMPORTANT: The extrinsics must be camera-to-world (c2w), NOT world-to-camera (w2c).
    This is because `get_world_rays` expects c2w matrices:
    - The camera origin in world space is extracted from c2w's translation column
    - Ray directions are transformed from camera space to world space using c2w
    
    If you have w2c matrices, you need to invert them first: c2w = w2c.inverse()
    
    Args:
        intrinsics: Camera intrinsics matrix of shape (*batch, 3, 3).
        c2w: Camera-to-world extrinsics matrix of shape (*batch, 4, 4) or (*batch, 3, 4).
        height: Image height in pixels.
        width: Image width in pixels.
        device: Device for the output tensor. If None, uses intrinsics.device.
        dtype: Dtype for the output tensor. If None, uses intrinsics.dtype.
        
    Returns:
        raymap: Tensor of shape (*batch, H, W, 6) where:
            - raymap[..., :3] = ray directions (normalized, unit length)
            - raymap[..., 3:] = ray origins (camera position in world space)
            
    Example:
        >>> intrinsics = torch.eye(3)[None]  # (1, 3, 3)
        >>> c2w = torch.eye(4)[None]  # (1, 4, 4) 
        >>> raymap = intrinsics_c2w_to_raymap(intrinsics, c2w, height=480, width=640)
        >>> # Output shape: (1, 480, 640, 6)
    """
    if device is None:
        device = intrinsics.device
    if dtype is None:
        dtype = intrinsics.dtype
    
    # Get batch dimensions from intrinsics (excluding last two dims which are 3x3)
    batch_shape = intrinsics.shape[:-2]
    
    # Generate pixel coordinates grid (no offset to match dust3r convention)
    # Using indexing='ij' for grid and then stacking to [x, y]
    rows, cols = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([cols, rows], dim=-1)  # (H, W, 2)
    
    # Expand coordinates to match batch dimensions
    # pixel_coords: (H, W, 2) -> (*batch, H, W, 2)
    for _ in range(len(batch_shape)):
        pixel_coords = pixel_coords.unsqueeze(0)
    expand_shape = list(batch_shape) + [height, width, 2]
    pixel_coords = pixel_coords.expand(*expand_shape)
    
    # Ensure extrinsics are 4x4 homogeneous matrices
    c2w_4x4 = as_homogeneous(c2w)  # (*batch, 4, 4)
    
    # Expand extrinsics and intrinsics for spatial dimensions
    # get_world_rays expects: coordinates (*#batch, 2), extrinsics (*#batch, 4, 4), intrinsics (*#batch, 3, 3)
    c2w_expanded = c2w_4x4
    intrinsics_expanded = intrinsics
    for _ in range(2):  # Add H, W dimensions
        c2w_expanded = c2w_expanded.unsqueeze(-3)
        intrinsics_expanded = intrinsics_expanded.unsqueeze(-3)
    
    # Expand to (*batch, H, W, 4, 4) and (*batch, H, W, 3, 3)
    c2w_expanded = c2w_expanded.expand(*batch_shape, height, width, 4, 4)
    intrinsics_expanded = intrinsics_expanded.expand(*batch_shape, height, width, 3, 3)
    
    # Compute world rays using get_world_rays
    # Note: get_world_rays expects c2w (camera-to-world), NOT w2c
    # Use normalize=False to get unnormalized directions (z=1 in camera space)
    # This is correct for z-depth computation: point = origin + direction * z_depth
    origins, directions = get_world_rays(
        pixel_coords,
        c2w_expanded,
        intrinsics_expanded,
        normalize=False,
    )  # Both (*batch, H, W, 3)
    
    # Concatenate to form raymap: [directions, origins]
    # Following the DA3 convention where raymap[..., :3] = directions, raymap[..., 3:] = origins
    raymap = torch.cat([directions, origins], dim=-1)  # (*batch, H, W, 6)
    
    return raymap



def intrinsics_c2w_to_raymap_np(
    intrinsics: np.ndarray,  # (3, 3)
    c2w: np.ndarray,         # (4, 4) or (3, 4) - camera-to-world (NOT world-to-camera!)
    height: int,
    width: int,
) -> np.ndarray:  # (H, W, 6)
    """
    Generate GT raymap from camera intrinsics and camera-to-world (c2w) extrinsics.
    
    This function generates a raymap that is consistent with depthmap_to_absolute_camera_coordinates.
    The relationship is: pts3d = origin + direction * z_depth
    where direction has z=1 in camera space (unnormalized).
    
    Args:
        intrinsics: Camera intrinsics matrix of shape (3, 3).
        c2w: Camera-to-world extrinsics matrix of shape (4, 4) or (3, 4).
        height: Image height in pixels.
        width: Image width in pixels.
        
    Returns:
        raymap: Array of shape (H, W, 6) where:
            - raymap[..., :3] = ray directions (unnormalized, z=1 in camera space)
            - raymap[..., 3:] = ray origins (camera position in world space)
    """
    # Extract intrinsic parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    # Create pixel grid using pixel centers (0.5 offset)
    # Pixel centers are at (0.5, 1.5, 2.5, ...) for a W-pixel wide image
    # u, v = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    u, v = np.meshgrid(np.arange(width), np.arange(height))

    
    # Compute camera-space ray directions with z=1 (unnormalized)
    # x_cam = (u - cx) / fx
    # y_cam = (v - cy) / fy
    # z_cam = 1
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    z_cam = np.ones_like(u)
    directions_cam = np.stack([x_cam, y_cam, z_cam], axis=-1).astype(np.float32)  # (H, W, 3)
    
    # Ensure c2w is 4x4
    if c2w.shape == (3, 4):
        c2w_4x4 = np.eye(4, dtype=np.float32)
        c2w_4x4[:3, :] = c2w
    else:
        c2w_4x4 = c2w
    
    # Extract rotation and translation from c2w
    R_cam2world = c2w_4x4[:3, :3]  # (3, 3)
    t_cam2world = c2w_4x4[:3, 3]   # (3,)
    
    # Transform directions to world space (same as depthmap_to_absolute_camera_coordinates)
    # X_world = R @ X_cam + t
    # For directions: directions_world = R @ directions_cam (no translation for directions)
    directions_world = np.einsum("ik, hwk -> hwi", R_cam2world, directions_cam)  # (H, W, 3)
    
    # Ray origins are the camera position in world space (same for all pixels)
    origins = np.broadcast_to(t_cam2world[None, None, :], (height, width, 3)).astype(np.float32)
    
    # Concatenate to form raymap: [directions, origins]
    # Following the DA3 convention where raymap[..., :3] = directions, raymap[..., 3:] = origins
    raymap = np.concatenate([directions_world, origins], axis=-1).astype(np.float32)  # (H, W, 6)
    
    return raymap


def convert_depth_to_point_cloud(depth, intrinsics, c2w):
    """
    Convert depth maps to point clouds in world coordinates.

    Args:
        depth: (B, T, H, W)
        intrinsics: (B, T, 3, 3)
        c2w: (B, T, 3, 4) or (B, T, 4, 4) - camera-to-world transformation

    Returns:
        point_map: (B, T, H, W, 3) World space coordinates
    """
    b, t, h, w = depth.shape
    device = depth.device

    # 1. Create pixel grid coordinates
    # Create meshgrid of pixel coordinates (y, x)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing='ij'
    )
    # Stack to get (H, W, 2) with [x, y] coordinates
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1)  # (H, W, 2)

    # Broadcast to (B, T, H, W, 2)
    pixel_coords = pixel_coords[None, None, :, :, :].expand(b, t, -1, -1, -1)

    # 2. Convert pixel coordinates to camera space using intrinsics
    # Extract intrinsic parameters: fx, fy, cx, cy
    fx = intrinsics[:, :, 0, 0]  # (B, T)
    fy = intrinsics[:, :, 1, 1]  # (B, T)
    cx = intrinsics[:, :, 0, 2]  # (B, T)
    cy = intrinsics[:, :, 1, 2]  # (B, T)

    # Reshape for broadcasting: (B, T, 1, 1)
    fx = fx[:, :, None, None]
    fy = fy[:, :, None, None]
    cx = cx[:, :, None, None]
    cy = cy[:, :, None, None]

    # Compute normalized camera coordinates
    x_cam = (pixel_coords[..., 0] - cx) / fx  # (B, T, H, W)
    y_cam = (pixel_coords[..., 1] - cy) / fy  # (B, T, H, W)

    # 3. Compute 3D points in camera space: [X, Y, Z] = [x_cam * depth, y_cam * depth, depth]
    points_cam = torch.stack([
        x_cam * depth,
        y_cam * depth,
        depth
    ], dim=-1)  # (B, T, H, W, 3)

    # 4. Transform to world coordinates using extrinsics (camera to world)
    # extrinsics is (B, T, 3, 4) = [R | t] where R is 3x3 rotation, t is 3x1 translation
    R = c2w[:, :, :3, :3]  # (B, T, 3, 3)
    T = c2w[:, :, :3, 3]   # (B, T, 3)
    
    # Reshape for batch matrix multiplication
    # points_cam: (B, T, H, W, 3) -> (B, T, H*W, 3)
    points_cam_flat = points_cam.reshape(b, t, h * w, 3)

    # Apply rotation: R @ points_cam^T -> (B, T, 3, 3) @ (B, T, 3, H*W) = (B, T, 3, H*W)
    # We need to transpose points for matmul: (B, T, H*W, 3) -> (B, T, 3, H*W)
    points_cam_T = points_cam_flat.transpose(-2, -1)  # (B, T, 3, H*W)
    points_world_T = torch.matmul(R, points_cam_T)  # (B, T, 3, H*W)
    
    # Transpose back and add translation
    points_world_flat = points_world_T.transpose(-2, -1)  # (B, T, H*W, 3)
    points_world_flat = points_world_flat + T[:, :, None, :]  # (B, T, H*W, 3)
    
    # Reshape back to spatial dimensions
    point_map = points_world_flat.reshape(b, t, h, w, 3)  # (B, T, H, W, 3)

    return point_map


def save_gt_render_data(
    img_views,
    output_folder,
    batch_idx=0,
    imagenet_normalize=True,
):
    """
    Save ground-truth rendering data to a .npy file.
    
    Args:
        img_views: List of view dictionaries containing 'pts3d', 'depthmap', 'img', 
                   'camera_pose', 'camera_intrinsics' keys.
        output_folder: Directory to save the output .npy file.
        batch_idx: Batch index to extract (default: 0).
        imagenet_normalize: If True, undo ImageNet normalization for colors.
                           If False, assume images are in [-1, 1] range.
    
    Returns:
        save_path: Path to the saved .npy file.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    device = 'cpu'
    
    # Stack data from views
    # pts3d: (B, T, H, W, 3)
    pts3d = torch.stack([v['pts3d'] for v in img_views], dim=1)
    pts3d_render = pts3d[batch_idx].detach().cpu()  # (T, H, W, 3)
    
    # Depth: (B, T, H, W)
    if 'depthmap' in img_views[0]:
        gt_depths = torch.stack([v['depthmap'] for v in img_views], dim=1)
        gt_depths = gt_depths[batch_idx].detach().cpu()  # (T, H, W)
    else:
        gt_depths = pts3d_render[..., 2]  # Use Z coordinate as depth
    
    # Confidence: use ones as placeholder if not available  
    t, h, w = gt_depths.shape
    conf_render = torch.ones(t, h, w, dtype=torch.float32)
    
    # Colors from images: (B, T, C, H, W)
    _img = torch.stack([v['img'] for v in img_views], dim=1)
    _img = _img[batch_idx].detach().cpu()  # (T, 3, H, W)
    
    if imagenet_normalize:
        # Undo ImageNet normalization
        _mean = torch.tensor([0.485, 0.456, 0.406], dtype=_img.dtype, device=_img.device).view(1, 3, 1, 1)
        _std = torch.tensor([0.229, 0.224, 0.225], dtype=_img.dtype, device=_img.device).view(1, 3, 1, 1)
        _img = (_img * _std) + _mean
        colors = _img.clamp(0.0, 1.0).mul(2.0).sub(1.0).permute(0, 2, 3, 1)  # (T, H, W, 3)
    else:
        # Assume images are in [-1, 1] range
        colors = _img.clamp(-1.0, 1.0).permute(0, 2, 3, 1)  # (T, H, W, 3)
    
    # Camera intrinsics: (B, T, 3, 3)
    intrinsics = torch.stack([v['camera_intrinsics'] for v in img_views], dim=1)
    intrinsics = intrinsics[batch_idx].detach().cpu()  # (T, 3, 3)
    focal = torch.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], dim=-1)  # (T, 2)
    
    # Camera poses (c2w): (B, T, 4, 4)
    c2w = torch.stack([v['camera_pose'] for v in img_views], dim=1)
    c2w_save = c2w[batch_idx].detach().cpu()  # (T, 4, 4)
    # Extract 3x4 if needed
    if c2w_save.shape[-2:] == (4, 4):
        c2w_save = c2w_save[:, :3, :]  # (T, 3, 4)
    
    # Placeholders for semantic and is_recon
    recon_semantic_2ds = torch.zeros(t, h, w, dtype=torch.float32)  # (T, H, W)
    is_recon = torch.ones(t, h, w, dtype=torch.bool)  # (T, H, W)
    
    # c2w_save[0, :3, :3] = torch.eye(3, dtype=torch.float32)
    # Create save dictionary
    save_dict = {
        "pts3d": pts3d_render.numpy(),
        "conf": conf_render.numpy(),
        "colors": colors.numpy(),
        "gt_depths": gt_depths.numpy(),
        "focal": focal[:, 0].numpy(),
        "c2w": c2w_save.numpy(),
        "semantic_2ds": recon_semantic_2ds.numpy(),
        "is_recon": is_recon.numpy(),
    }
    
    # Save to .npy file
    save_path = os.path.join(output_folder, "pts3d_render.npy")
    np.save(save_path, save_dict)
    print(f"Saved pts3d_render.npy to {save_path}")
    
    return save_path


def save_gt_render_data_single(
    views,
    output_folder,
    imagenet_normalize=False,
):
    """
    Save ground-truth rendering data to a .npy file for a single (unbatched) sample.
    This is designed to be called from the dataset's __getitem__ method.
    
    Args:
        views: List of view dictionaries containing 'pts3d', 'depthmap', 'img', 
               'camera_pose', 'camera_intrinsics' keys. Each tensor has no batch dimension.
        output_folder: Directory to save the output .npy file.
        imagenet_normalize: If True, undo ImageNet normalization for colors.
                           If False, assume images are in [0, 1] range or PIL images.
    
    Returns:
        save_path: Path to the saved .npy file.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    t = len(views)
    
    # Stack data from views - no batch dimension
    # pts3d: (H, W, 3) per view -> (T, H, W, 3)
    pts3d_render = np.stack([v['pts3d'] for v in views], axis=0)  # (T, H, W, 3)
    
    # Depth: (H, W) per view -> (T, H, W)
    if 'depthmap' in views[0]:
        gt_depths = np.stack([v['depthmap'] for v in views], axis=0)  # (T, H, W)
    else:
        gt_depths = pts3d_render[..., 2]  # Use Z coordinate as depth
    
    # Confidence: use ones as placeholder
    h, w = gt_depths.shape[1], gt_depths.shape[2]
    conf_render = np.ones((t, h, w), dtype=np.float32)
    
    # Colors from images
    colors_list = []
    for v in views:
        img = v['img']
        if hasattr(img, 'size'):  # PIL Image
            img_np = np.array(img).astype(np.float32) / 255.0  # (H, W, 3) in [0, 1]
            # Convert to [-1, 1] range
            img_np = img_np * 2.0 - 1.0
        elif isinstance(img, torch.Tensor):
            img_tensor = img  # (C, H, W)
            if imagenet_normalize:
                # Undo ImageNet normalization
                _mean = torch.tensor([0.485, 0.456, 0.406], dtype=img_tensor.dtype).view(3, 1, 1)
                _std = torch.tensor([0.229, 0.224, 0.225], dtype=img_tensor.dtype).view(3, 1, 1)
                img_tensor = (img_tensor * _std) + _mean
                img_np = img_tensor.clamp(0.0, 1.0).mul(2.0).sub(1.0).permute(1, 2, 0).numpy()  # (H, W, 3)
            else:
                # Assume already in [0, 1] range from DA3 normalization
                img_np = img_tensor.clamp(0.0, 1.0).mul(2.0).sub(1.0).permute(1, 2, 0).numpy()  # (H, W, 3)
        else:
            img_np = img  # Assume already numpy (H, W, 3)
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            img_np = img_np * 2.0 - 1.0
        colors_list.append(img_np)
    colors = np.stack(colors_list, axis=0)  # (T, H, W, 3)
    
    # Camera intrinsics: (3, 3) per view -> (T, 3, 3)
    intrinsics = np.stack([v['camera_intrinsics'] for v in views], axis=0)  # (T, 3, 3)
    focal = np.stack([intrinsics[:, 0, 0], intrinsics[:, 1, 1]], axis=-1)  # (T, 2)
    
    # Camera poses (c2w): (4, 4) per view -> (T, 4, 4)
    c2w = np.stack([v['camera_pose'] for v in views], axis=0)  # (T, 4, 4)
    # Extract 3x4 if needed
    c2w_save = c2w[:, :3, :]  # (T, 3, 4)
    
    # Placeholders for semantic and is_recon
    recon_semantic_2ds = np.zeros((t, h, w), dtype=np.float32)
    is_recon = np.ones((t, h, w), dtype=bool)
    
    # Create save dictionary
    save_dict = {
        "pts3d": pts3d_render,
        "conf": conf_render,
        "colors": colors,
        "gt_depths": gt_depths,
        "focal": focal[:, 0],
        "c2w": c2w_save,
        "semantic_2ds": recon_semantic_2ds,
        "is_recon": is_recon,
    }
    
    # Save to .npy file
    save_path = os.path.join(output_folder, "pts3d_render.npy")
    np.save(save_path, save_dict)
    print(f"Saved pts3d_render.npy to {save_path}")
    
    return save_path
    

def build_fine_prompt_metadata(text_prompt_list):
    """Return flattened prompt strings and the mapping back to KITTI class indices."""
    flattened_prompts = []
    prompt_to_class_index = []
    for class_idx, synonym_list in enumerate(text_prompt_list):
        flattened_prompts.extend(synonym_list)
        prompt_to_class_index.extend([class_idx] * len(synonym_list))
    return flattened_prompts, prompt_to_class_index


def apply_majority_pooling(
    voxel_pred_np,
    n_classes,
    other_class,
    empty_class,
    is_geometry_only=False,
    use_dilation=True,
    update_other_only=False,
):  # like in scenedino, s4c
    """Apply majority pooling to voxel predictions.
    
    Args:
        voxel_pred_np: numpy array of voxel predictions (H, W, D)
        n_classes: number of classes
        other_class: index of "other" class to ignore
        empty_class: index of "empty" class to ignore
        is_geometry_only: if True, treat as binary occupancy (no semantic classes)
        use_dilation: if True, use max pooling (dilation) instead of majority voting for geometry_only mode
        update_other_only: if True in semantic mode, only relabel voxels that are currently
            `other_class`; otherwise relabel all occupied voxels with valid semantic votes.
    
    Returns:
        numpy array of pooled predictions with same shape as input
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_geometry_only:
        # Create binary mask: 1 for occupied (not empty_class), 0 for empty
        voxel_pred_tensor = torch.from_numpy(voxel_pred_np).to(device)
        binary_mask = (voxel_pred_tensor != empty_class).float()
        
        if use_dilation:
            # Mode 1: Binary max pooling (dilation) - occupied if ANY neighbor is occupied
            binary_mask_pooled = F.max_pool3d(
                binary_mask.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(0).squeeze(0)
            is_occupied = binary_mask_pooled > 0.0
        else:
            # Mode 2: Majority voting - occupied if MORE THAN HALF of neighbors are occupied
            # Only update empty voxels to occupied, never update occupied to empty
            # Apply average pooling to count occupied neighbors in 3x3x3 neighborhood
            # avg_pool3d computes the mean, so multiply by 27 to get the count
            neighbor_count = F.avg_pool3d(
                binary_mask.unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(0).squeeze(0) * 27.0
            
            # Majority vote: occupied if more than half of 27 neighbors are occupied (>13.5)
            should_be_occupied = neighbor_count > 13.5
            
            # Only update empty voxels to occupied, keep already occupied voxels as is
            is_occupied = binary_mask.bool() | should_be_occupied
        
        # Convert back: occupied voxels get a non-empty value, empty voxels get empty_class
        # Use 0 for occupied if empty_class != 0, otherwise use 1
        occupied_value = 0 if empty_class != 0 else 1
        result = torch.where(
            is_occupied,
            torch.full_like(voxel_pred_tensor, occupied_value),
            torch.full_like(voxel_pred_tensor, empty_class)
        )
        return result.cpu().numpy().astype(np.uint8)
    else:
        # For semantic predictions, apply majority pooling only to "other" class voxels
        # When computing majority, only count voxels that are not empty and not other
        voxel_pred_tensor = torch.from_numpy(voxel_pred_np).to(device)
        
        # Select which occupied voxels are eligible for semantic relabeling.
        if update_other_only:
            to_update_mask = voxel_pred_tensor == other_class
        else:
            to_update_mask = voxel_pred_tensor != empty_class
        
        
        # Create one-hot encoding on GPU, excluding empty and other classes
        voxel_pred_onehot = torch.zeros((n_classes,) + voxel_pred_tensor.shape, 
                                         dtype=torch.float32, device=device)
        for c in range(n_classes):
            if c == other_class or c == empty_class:
                continue  # Skip empty and other classes - they won't be counted
            voxel_pred_onehot[c] = (voxel_pred_tensor == c).float()
        
        # Apply average pooling (equivalent to counting votes in each neighborhood)
        # Only non-empty and non-other voxels contribute to the vote
        voxel_pred_onehot_pooled = F.avg_pool3d(
            voxel_pred_onehot.unsqueeze(0), 
            kernel_size=3, 
            stride=1, 
            padding=1
        ).squeeze(0)
        
        # Check if there are any valid votes (non-empty, non-other neighbors)
        has_valid_votes = voxel_pred_onehot_pooled.sum(dim=0) > 0
        
        # Convert back to class labels (argmax gives majority class)
        pooled_labels = torch.argmax(voxel_pred_onehot_pooled, dim=0)
        
        # Only update eligible voxels that have valid semantic neighbors.
        result = voxel_pred_tensor.clone()
        update_mask = to_update_mask & has_valid_votes
        result[update_mask] = pooled_labels[update_mask].to(result.dtype)
        # print(f"Number of voxels updated: {update_mask.sum()}")
        # Voxels with only empty/other neighbors remain as "other" class
        
        return result.cpu().numpy().astype(np.uint8)


def apply_unified_majority_pooling(
    voxel_pred_np: np.ndarray,
    n_classes: int,
    other_class: int,
    empty_class: int,
    use_dilation: bool = True,
) -> np.ndarray:
    """Apply unified majority pooling: geometry expansion followed by semantic refinement.
    
    This function performs a two-step process:
    1. Geometry pooling: Expand occupancy using dilation or majority voting.
       Newly occupied voxels (previously empty) are set to `other_class`.
    2. Semantic majority pooling: Refine all occupied voxels by assigning the
       majority class from neighbors. Only real semantic classes vote (excluding
       `other_class` and `empty_class`). Voxels are only updated if valid votes exist.
    
    Args:
        voxel_pred_np: numpy array of voxel predictions (H, W, D)
        n_classes: number of classes
        other_class: index of "other" class (used for newly filled voxels)
        empty_class: index of "empty" class
        use_dilation: if True, use max pooling (dilation) for geometry expansion;
                      if False, use majority voting (>50% neighbors occupied)
    
    Returns:
        numpy array of pooled predictions with same shape as input
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    voxel_pred_tensor = torch.from_numpy(voxel_pred_np).to(device)
    
    # =========================================================================
    # Step 1: Geometry pooling - expand occupancy
    # =========================================================================
    # Create binary mask: 1 for occupied (not empty_class), 0 for empty
    original_occupied = (voxel_pred_tensor != empty_class)
    binary_mask = original_occupied.float()
    
    if use_dilation:
        # Mode 1: Binary max pooling (dilation) - occupied if ANY neighbor is occupied
        binary_mask_pooled = F.max_pool3d(
            binary_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze(0).squeeze(0)
        is_occupied = binary_mask_pooled > 0.0
    else:
        # Mode 2: Majority voting - occupied if MORE THAN HALF of neighbors are occupied
        neighbor_count = F.avg_pool3d(
            binary_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            stride=1,
            padding=1
        ).squeeze(0).squeeze(0) * 27.0
        
        # Majority vote: occupied if more than half of 27 neighbors are occupied (>13.5)
        should_be_occupied = neighbor_count > 13.5
        
        # Only update empty voxels to occupied, keep already occupied voxels as is
        is_occupied = original_occupied | should_be_occupied
    
    # Identify newly occupied voxels (were empty, now occupied)
    newly_occupied = is_occupied & ~original_occupied
    
    # Set newly occupied voxels to other_class
    result = voxel_pred_tensor.clone()
    result[newly_occupied] = other_class
    
    # =========================================================================
    # Step 2: Semantic majority pooling - refine all occupied voxels
    # =========================================================================
    # Create one-hot encoding excluding empty and other classes
    # Only real semantic classes contribute to voting
    voxel_pred_onehot = torch.zeros(
        (n_classes,) + result.shape, dtype=torch.float32, device=device
    )
    for c in range(n_classes):
        if c == other_class or c == empty_class:
            continue  # Skip empty and other classes - they won't vote
        voxel_pred_onehot[c] = (result == c).float()
    
    # Apply average pooling (equivalent to counting votes in each neighborhood)
    voxel_pred_onehot_pooled = F.avg_pool3d(
        voxel_pred_onehot.unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1
    ).squeeze(0)
    
    # Check if there are any valid votes (non-empty, non-other neighbors)
    has_valid_votes = voxel_pred_onehot_pooled.sum(dim=0) > 0
    
    # Convert back to class labels (argmax gives majority class)
    pooled_labels = torch.argmax(voxel_pred_onehot_pooled, dim=0)
    
    # Update all occupied voxels that have valid neighbors with real semantic classes
    # This includes both original occupied voxels and newly occupied ones
    update_mask = is_occupied & has_valid_votes
    result[update_mask] = pooled_labels[update_mask].to(result.dtype)
    
    # Voxels without valid semantic neighbors keep their current class:
    # - Original semantic voxels remain unchanged if no valid neighbors
    # - Newly occupied voxels remain as other_class if no valid neighbors
    
    return result.cpu().numpy().astype(np.uint8)


def voxel_to_pts3d(voxel_grid, voxel_origin, voxel_size, keep_all=False,empty_class=0):
    """Convert voxel grid to point cloud.
    
    Args:
        voxel_grid: (H, W, D) array with semantic labels
        voxel_origin: (3,) array with origin coordinates
        voxel_size: float, size of each voxel
    
    Returns:
        pts3d: (N, 3) array of point coordinates
        labels: (N,) array of semantic labels
    """
    # Get occupied voxel indices (exclude 0=empty and 255=invalid)
    if keep_all:
        occupied_mask = np.ones_like(voxel_grid, dtype=bool)
    else:
        occupied_mask = (voxel_grid != empty_class) & (voxel_grid != 255)
    indices = np.argwhere(occupied_mask)  # (N, 3)
    
    # Convert indices to world coordinates
    # voxel_grid is typically (X, Y, Z) = (256, 256, 32)
    pts3d = indices.astype(np.float32) * voxel_size + voxel_origin
    labels = voxel_grid[occupied_mask]
    
    return pts3d, labels


def process_voxels(
    voxel_grid: torch.Tensor, 
    num_classes: int, 
    empty_class_label: int, 
    kernel_size: tuple = (2, 2, 2), 
    stride: tuple = (2, 2, 2),
    other_class_label: int = 20
) -> torch.Tensor:
    """
    Performs one-hot encoding, 3D average pooling, and a masked argmax 
    (excluding the empty class) on a voxel grid. If the winning class is 
    other_class_label, replace it with the second most confident class.

    Args:
        voxel_grid: Input tensor of integer class labels.
                      Expected shape: (B, D, H, W).
        num_classes: The total number of classes (e.g., if labels are
                       0, 1, 2, num_classes=3).
        empty_class_label: The integer label of the class to exclude
                           from the final argmax (e.g., 0).
        kernel_size: The size of the pooling window.
        stride: The stride of the pooling window.
        other_class_label: The integer label of "other" class that should
                           be filled with the second most confident class.

    Returns:
        A downsampled tensor of class labels, with the empty class
        excluded from the "winning" class calculation and other_class
        filled with the second best class.
        Shape: (B, D_out, H_out, W_out).
    """
    
    if voxel_grid.dim() == 3:
        voxel_grid = voxel_grid.unsqueeze(0)
    
    # 4. Fill other_class_label with second most confident class
    # Find locations where the winner is other_class_label
    other_mask = (voxel_grid == other_class_label)
    if other_mask.sum() == 0:
        return voxel_grid
    

    # 1. Convert to one-hot
    # Input shape: (B, D, H, W)
    # F.one_hot output: (B, D, H, W, C)
    one_hot = F.one_hot(voxel_grid.long(), num_classes=num_classes+1)
    
    # Permute to the channel-first format required by 3D ops
    # Shape: (B, C, D, H, W)
    # Convert to float for pooling calculation
    one_hot = one_hot.permute(0, 4, 1, 2, 3).float()

    # 2. Do average pooling
    # This calculates the average presence of each class in the kernel window
    # Shape: (B, C, D_out, H_out, W_out)
    pooled = F.avg_pool3d(one_hot, kernel_size=kernel_size, stride=stride, padding=1)
    # 3. Argmax while excluding the empty voxel
    
    # To exclude the empty class, we set its score to negative infinity
    # so it can never win the argmax.
    pooled_masked = pooled.clone()
    pooled_masked[:, empty_class_label, :, :, :] = -torch.inf
    pooled_masked[:, other_class_label, :, :, :] = -torch.inf
    
    # Find the index (class) with the highest average score.
    # dim=1 is the channel (class) dimension.
    # Shape: (B, D_out, H_out, W_out)
    final_labels = torch.argmax(pooled_masked, dim=1)

    final_labels = final_labels.to(voxel_grid.dtype)
    
    nonempty_mask = voxel_grid != empty_class_label
    voxel_grid[nonempty_mask] = final_labels[nonempty_mask]
    
    return voxel_grid.squeeze()
    # # Convert final_labels to match voxel_grid dtype
    # final_labels = final_labels.to(voxel_grid.dtype)
    
    # if other_mask.any():
    #     voxel_grid[other_mask] = final_labels[other_mask]
    
    # return voxel_grid.squeeze()


def compute_vox_visible_mask(voxel_label, camera_poses, K, T_velo_2_cam, voxel_origin, voxel_size=0.2, save_path=None):
    """
    This function compute the visible mask of voxels that are projected on the image plane using 
    camera_poses and K.
    Parameters:
        voxel_label: B, 256 , 256, 32
        camera_poses: B, N, 4, 4
        K: B, 3, 3
        T_velo_2_cam: B, 4, 4
        voxel_origin: 3
        voxel_size: float
        save_path: str or None, if provided, save visualization point clouds to this directory

    Returns:
        visible mask: binary mask of size B, 256, 256, 32
    """
    device = voxel_label.device
    B, X, Y, Z = voxel_label.shape
    N = camera_poses.shape[1]
    
    # Ensure all inputs are on the correct device and dtype (float32)
    if isinstance(voxel_origin, torch.Tensor):
        voxel_origin = voxel_origin.to(device=device, dtype=torch.float32)
    else:
        voxel_origin = torch.tensor(voxel_origin, device=device, dtype=torch.float32)
    
    camera_poses = camera_poses.to(device=device, dtype=torch.float32)
    K = K.to(device=device, dtype=torch.float32)
    T_velo_2_cam = T_velo_2_cam.to(device=device, dtype=torch.float32)
    
    # Generate voxel center coordinates in velodyne frame
    # Create grid indices
    x_indices = torch.arange(X, device=device, dtype=torch.float32)
    y_indices = torch.arange(Y, device=device, dtype=torch.float32)
    z_indices = torch.arange(Z, device=device, dtype=torch.float32)
    
    # Create meshgrid
    xx, yy, zz = torch.meshgrid(x_indices, y_indices, z_indices, indexing='ij')
    
    # Compute voxel centers in velodyne frame
    # voxel_centers shape: (X*Y*Z, 3)
    voxel_centers = torch.stack([
        xx.flatten() * voxel_size + voxel_origin[0] + voxel_size / 2,
        yy.flatten() * voxel_size + voxel_origin[1] + voxel_size / 2,
        zz.flatten() * voxel_size + voxel_origin[2] + voxel_size / 2
    ], dim=1)
    
    # Convert to homogeneous coordinates (X*Y*Z, 4)
    voxel_centers_homo = torch.cat([
        voxel_centers,
        torch.ones(voxel_centers.shape[0], 1, device=device)
    ], dim=1)
    
    # Initialize visibility mask
    visible_mask = torch.zeros(B, X * Y * Z, device=device, dtype=torch.bool)
    
    # Process each batch
    for b in range(B):
        # Transform voxel centers from velodyne to cam0
        # T_velo_2_cam shape: (4, 4)
        voxels_in_cam0 = (T_velo_2_cam[b] @ voxel_centers_homo.T).T  # (X*Y*Z, 4)
        
        # Get image dimensions from K (assuming standard image size)
        # Typical KITTI image sizes: (512, 160) or (224, 80) [W, H]
        # We'll infer from K's principal point
        cx = K[b, 0, 2]
        cy = K[b, 1, 2]
        img_w = int(cx * 2)
        img_h = int(cy * 2)
        
        # Process each camera view
        for n in range(N):
            # Transform from cam0 to camN
            # camera_poses[b, n] is camN-to-cam0, so we need cam0-to-camN
            cam0_to_camN = torch.inverse(camera_poses[b, n])
            voxels_in_camN = (cam0_to_camN @ voxels_in_cam0.T).T  # (X*Y*Z, 4)
            
            # Extract 3D points
            points3d_camera = voxels_in_camN[:, :3]  # (X*Y*Z, 3)
            
            # Filter points behind camera
            depth = points3d_camera[:, 2]
            in_front = depth > 0
            
            # Project to image plane
            points2d = K[b] @ points3d_camera.T  # (3, X*Y*Z)
            points2d = points2d[:2, :] / (points2d[2:3, :] + 1e-8)  # (2, X*Y*Z)
            points2d = points2d.T  # (X*Y*Z, 2)
            
            # Check if within image bounds
            in_image = (
                (points2d[:, 0] >= 0) & (points2d[:, 0] < img_w) &
                (points2d[:, 1] >= 0) & (points2d[:, 1] < img_h)
            )
            
            # Combine conditions
            is_visible = in_front & in_image
            
            # Update visibility mask (OR operation across views)
            visible_mask[b] = visible_mask[b] | is_visible
    
    # Reshape back to voxel grid shape
    visible_mask = visible_mask.view(B, X, Y, Z)
    
    # Save visualization if requested
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        for b in range(B):
            # Convert tensors to numpy
            voxel_label_np = voxel_label[b].cpu().numpy()
            visible_mask_np = visible_mask[b].cpu().numpy()
            voxel_origin_np = voxel_origin.cpu().numpy() if isinstance(voxel_origin, torch.Tensor) else voxel_origin
            
            # Save voxel_label as point cloud (occupied voxels only)
            occupied_mask = (voxel_label_np != 255) & (voxel_label_np != 0)
            if occupied_mask.any():
                coords = np.argwhere(occupied_mask)
                points = coords * voxel_size + voxel_origin_np + voxel_size / 2
                labels = voxel_label_np[occupied_mask]
                # Save with label as feature
                save_pcd_as_txt(
                    os.path.join(save_path, f'voxel_label_b{b}.txt'),
                    points,
                    labels.reshape(-1, 1)
                )
            
            # Save visible mask as point cloud (visible voxels only)
            if visible_mask_np.any():
                coords_visible = np.argwhere(visible_mask_np)
                points_visible = coords_visible * voxel_size + voxel_origin_np + voxel_size / 2
                # Add visibility flag (1 for visible)
                visibility_flag = np.ones((len(points_visible), 1))
                save_pcd_as_txt(
                    os.path.join(save_path, f'visible_mask_b{b}.txt'),
                    points_visible,
                    visibility_flag
                )
            
            # Save combined visualization: voxel_label with visibility info
            if occupied_mask.any():
                coords_occupied = np.argwhere(occupied_mask)
                points_occupied = coords_occupied * voxel_size + voxel_origin_np + voxel_size / 2
                labels_occupied = voxel_label_np[occupied_mask]
                visibility_occupied = visible_mask_np[occupied_mask].astype(np.float32)
                # Save with both label and visibility as features
                combined_features = np.column_stack([labels_occupied, visibility_occupied])
                save_pcd_as_txt(
                    os.path.join(save_path, f'voxel_with_visibility_b{b}.txt'),
                    points_occupied,
                    combined_features
                )
    
    return visible_mask

def interpolate_se3_slerp(A, B, n):
    """
    Interpolate between SE(3) transforms using SLERP for rotation and linear
    interpolation for translation.

    Parameters
    ----------
    A : array-like, shape (..., 4, 4)
        Start transforms. Can be a single transform `[4, 4]` or a batch
        `[bs, 4, 4]`.
    B : array-like, shape (..., 4, 4)
        End transforms matching the shape of `A`.
    n : int
        Number of interpolation steps to generate (including endpoints).

    Returns
    -------
    If input is unbatched, returns a list of `n` numpy arrays with shape `(4, 4)`.
    If input is batched, returns a numpy array with shape `(bs, n, 4, 4)`.
    """
    n = n + 2
    if isinstance(A, torch.Tensor):
        A_np = A.detach().cpu().numpy()
    else:
        A_np = np.asarray(A)

    if isinstance(B, torch.Tensor):
        B_np = B.detach().cpu().numpy()
    else:
        B_np = np.asarray(B)

    if A_np.shape != B_np.shape:
        raise ValueError("A and B must have matching shapes")

    if A_np.ndim == 2:
        A_np = A_np[None, ...]
        B_np = B_np[None, ...]
        batched = False
    elif A_np.ndim == 3:
        batched = True
    else:
        raise ValueError("A and B must have shape (4, 4) or (bs, 4, 4)")

    bs = A_np.shape[0]
    key_times = np.array([0.0, 1.0])
    taus = np.linspace(0.0, 1.0, n)


    Ts = np.zeros((bs, n, 4, 4), dtype=np.float64)
    Ts[:, :, 3, 3] = 1.0

    for b in range(bs):
        RA = A_np[b, :3, :3]
        RB = B_np[b, :3, :3]
        tA = A_np[b, :3, 3]
        tB = B_np[b, :3, 3]

        key_rots = R.from_matrix(np.stack([RA, RB], axis=0))
        slerp = Slerp(key_times, key_rots)

        R_tau = slerp(taus).as_matrix()
        t_tau = (1.0 - taus)[:, None] * tA[None, :] + taus[:, None] * tB[None, :]

        Ts[b, :, :3, :3] = R_tau
        Ts[b, :, :3, 3] = t_tau

    if batched:
        return Ts[:, 1:-1]

    # Unbatched input: return list of 4x4 matrices for backward compatibility
    return [Ts[0, i] for i in range(1, n-1)]



def generate_novel_straight_rotated_poses(camera_poses, views_per_interval, 
                                          device, 
                                          forward=1.0, # m
                                          rotate_angle=0,
                                          lateral_translation=0.0):
    """
    For each camera pose, generate novel poses by moving straight forward and optionally rotating.
    
    Unlike interpolation between poses, this generates new viewpoints by:
    1. Moving the camera forward along its viewing direction (z-axis in camera frame)
    2. Optionally translating laterally (x-axis in camera frame)
    3. Optionally rotating left/right around y-axis
    
    Args:
        camera_poses: Camera poses tensor of shape (B, N, 4, 4)
        views_per_interval: Number of forward steps to generate from each pose
        device: Device to place tensors on
        forward: Distance in meters to move forward for each step (default: 0.5m)
        rotate_angle: Angle in degrees to rotate poses around y-axis. If > 0, generates
                     left and right rotated versions in addition to straight-forward poses.
        lateral_translation: Distance in meters to translate laterally. If rotate_angle > 0,
                     left rotation translates left (negative x), right rotation translates right (positive x).
    
    Returns:
        gen_poses: Generated poses tensor of shape (B, N*views_per_interval, 4, 4) if rotate_angle=0
        or (B, N*views_per_interval*3, 4, 4) if rotate_angle > 0 (includes straight, left, right)
    """
    if views_per_interval <= 0:
        batch_size = camera_poses.shape[0]
        return torch.empty(batch_size, 0, 4, 4, device=device, dtype=torch.float32), {}
    
    batch_size, num_poses = camera_poses.shape[0], camera_poses.shape[1]
    
    # Generate straight-forward poses
    gen_poses_list = []
    gen_2_recon_mapping = []
    for pose_idx in range(num_poses):
        cam_pose = camera_poses[:, pose_idx]  # (B, 4, 4)
        
        # Extract camera forward direction (positive z-axis in camera frame)
        # Camera looks down Z axis, so forward direction is the Z column of rotation matrix
        forward_dir = cam_pose[:, :3, 2]  # (B, 3) - camera's Z axis in world frame
        
        # Generate multiple steps forward
        for step in range(1, views_per_interval + 1):
            # Translation: move forward by (step * forward) meters
            translation_offset = forward_dir * (step * forward)  # (B, 3)
            
            # Create new pose: same rotation, translated position
            new_pose = cam_pose.clone()
            new_pose[:, :3, 3] = cam_pose[:, :3, 3] + translation_offset
            
            gen_poses_list.append(new_pose.unsqueeze(1))  # (B, 1, 4, 4)
            gen_2_recon_mapping.append(pose_idx)  # Track which recon view this gen view comes from
    
    # Concatenate all straight-forward poses
    gen_poses = torch.cat(gen_poses_list, dim=1)  # (B, N*views_per_interval, 4, 4)
    
    # Add rotated versions if rotate_angle is specified
    if rotate_angle > 0:
        # Step 1: Apply lateral translation first (if specified)
        gen_poses_left = gen_poses
        gen_poses_right = gen_poses
        
        if lateral_translation != 0.0:
            # Translate left (negative x in camera frame)
            trans_left = torch.tensor([
                [1, 0, 0, lateral_translation],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=gen_poses.dtype, device=gen_poses.device)
            gen_poses_left = torch.einsum('bnik,kj->bnij', gen_poses, trans_left)
            
            # Translate right (positive x in camera frame)
            trans_right = torch.tensor([
                [1, 0, 0, -lateral_translation],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=gen_poses.dtype, device=gen_poses.device)
            gen_poses_right = torch.einsum('bnik,kj->bnij', gen_poses, trans_right)
        
        # Step 2: Apply rotation around y-axis
        # Create rotation matrices for left and right rotation around y-axis
        angle_left = np.radians(rotate_angle)
        rot_left = torch.tensor([
            [np.cos(angle_left), 0, np.sin(angle_left), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_left), 0, np.cos(angle_left), 0],
            [0, 0, 0, 1]
        ], dtype=gen_poses.dtype, device=gen_poses.device)
        
        angle_right = np.radians(-rotate_angle)
        rot_right = torch.tensor([
            [np.cos(angle_right), 0, np.sin(angle_right), 0],
            [0, 1, 0, 0],
            [-np.sin(angle_right), 0, np.cos(angle_right), 0],
            [0, 0, 0, 1]
        ], dtype=gen_poses.dtype, device=gen_poses.device)
        
        # Apply rotations in camera's local frame: pose @ rot_matrix
        gen_poses_left = torch.einsum('bnik,kj->bnij', gen_poses_left, rot_left)
        gen_poses_right = torch.einsum('bnik,kj->bnij', gen_poses_right, rot_right)
        
        # Concatenate: [straight, left, right]
        gen_poses = torch.cat([gen_poses, gen_poses_left, gen_poses_right], dim=1)
        
        # Extend mapping for rotated versions (they map to same recon views as straight)
        n_straight = len(gen_2_recon_mapping)
        gen_2_recon_mapping = gen_2_recon_mapping + gen_2_recon_mapping + gen_2_recon_mapping  # [straight, left, right]
    
    
    # Convert to recon_2_gen_mapping: dict mapping recon_idx -> list of gen_indices
    recon_2_gen_mapping = {}
    for gen_idx, recon_idx in enumerate(gen_2_recon_mapping):
        if recon_idx not in recon_2_gen_mapping:
            recon_2_gen_mapping[recon_idx] = []
        recon_2_gen_mapping[recon_idx].append(gen_idx)
    
    return gen_poses, recon_2_gen_mapping




def generate_intermediate_poses(camera_poses, views_per_interval, device, 
                                interpolate=False,
                                forward=1.0,
                                rotate_angle=0,
                                num_seed_rotations=0,
                                seed_rotation_angle=None,
                                seed_translation_distance=None):
    """
    Generate intermediate camera poses between consecutive reconstructed views.
    
    Args:
        camera_poses: Camera poses tensor of shape (B, N, 4, 4)
        views_per_interval: Number of interpolated views between consecutive poses
        device: Device to place tensors on
        rotate_angle: Angle in degrees to rotate poses around y-axis. If > 0, generates
                     left and right rotated versions in addition to interpolated poses.
        num_seed_rotations: If > 0, first generate this many seed poses by rotating around
                           y-axis, then generate poses for each seed. Rotations are centered at 0°.
                           E.g., num_seed_rotations=5, seed_rotation_angle=5 generates [-10, -5, 0, 5, 10]
        seed_rotation_angle: Angle in degrees between seed rotations. Can be:
                           - A scalar: step size between rotations (e.g., 5 for [-10, -5, 0, 5, 10])
                           - A list: explicit angles to use (e.g., [-15, -10, 0, 10, 15])
                           - None: defaults to evenly spaced angles
        seed_translation_distance: Distance in meters to translate the seed poses laterally.
                           - If provided, positive rotation angles translate right (positive x in camera frame)
                           - Negative rotation angles translate left (negative x in camera frame)
                           - Can be a scalar or list matching rotation_angles
    
    Returns:
        Generated poses tensor. Shape depends on num_seed_rotations:
        - If num_seed_rotations=0: normal behavior
        - If num_seed_rotations>0: concatenates poses from all seed rotations
    """

    if views_per_interval <= 0:
        batch_size = camera_poses.shape[0]
        return torch.empty(batch_size, 0, 4, 4, device=device, dtype=torch.float32)

    gen_pose_segments = []
    if interpolate:
        raise NotImplementedError("Not correct anymore with the new recon_2_gen_mapping")
        for i in range(camera_poses.shape[1] - 1):
            cam_pose_start = camera_poses[:, i]
            cam_pose_end = camera_poses[:, i + 1]
            
            interp = interpolate_se3_slerp(
                cam_pose_start,
                cam_pose_end,
                views_per_interval,
            )
            # print( cam_pose_start[0, :3, 3])
            # print(cam_pose_end[0, :3, 3])
            # print(interp[0, 0, :3, 3])
        
            gen_pose_segments.append(interp)
    
        gen_poses_np = np.concatenate(gen_pose_segments, axis=1)
        gen_poses = torch.from_numpy(gen_poses_np).to(device).float()
        # Add rotated versions if rotate_angle is specified
        if rotate_angle > 0:
            # Rotate left by specified angle around y-axis (in camera's local frame)
            angle_left = np.radians(rotate_angle)
            rot_left = torch.tensor([
                [np.cos(angle_left), 0, np.sin(angle_left), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_left), 0, np.cos(angle_left), 0],
                [0, 0, 0, 1]
            ], dtype=gen_poses.dtype, device=gen_poses.device)
            # Apply rotation: gen_poses @ rot_matrix (rotate in camera's local frame)
            gen_poses_left = torch.einsum('bnik,kj->bnij', gen_poses, rot_left)
            
            # Rotate right by specified angle around y-axis (in camera's local frame)
            angle_right = np.radians(-rotate_angle)
            rot_right = torch.tensor([
                [np.cos(angle_right), 0, np.sin(angle_right), 0],
                [0, 1, 0, 0],
                [-np.sin(angle_right), 0, np.cos(angle_right), 0],
                [0, 0, 0, 1]
            ], dtype=gen_poses.dtype, device=gen_poses.device)
            # Apply rotation: gen_poses @ rot_matrix (rotate in camera's local frame)
            gen_poses_right = torch.einsum('bnik,kj->bnij', gen_poses, rot_right)
            # Add rotated poses to gen_poses
            gen_poses = torch.cat([gen_poses, gen_poses_left, gen_poses_right], dim=1)
    else:
        # Multi-seed rotation mode: generate seed poses with various rotations,
        # then generate poses for each seed
        if num_seed_rotations > 0:
            # Determine the list of rotation angles
            if isinstance(seed_rotation_angle, (list, tuple)):
                # Use explicit list of angles
                rotation_angles = list(seed_rotation_angle)
                if len(rotation_angles) != num_seed_rotations:
                    raise ValueError(f"seed_rotation_angle list length ({len(rotation_angles)}) "
                                   f"must match num_seed_rotations ({num_seed_rotations})")
            else:
                # Generate angles centered at 0
                if seed_rotation_angle is None:
                    # Default: evenly spaced angles
                    seed_rotation_angle = 15.0  # Default step
                
                # Generate symmetric angles around 0
                if num_seed_rotations % 2 == 1:
                    # Odd number: include 0 in the middle
                    # E.g., 5 rotations with step 5: [-10, -5, 0, 5, 10]
                    half = num_seed_rotations // 2
                    rotation_angles = [seed_rotation_angle * (i - half) for i in range(num_seed_rotations)]
                else:
                    # Even number: no 0, symmetric around it
                    # E.g., 4 rotations with step 5: [-7.5, -2.5, 2.5, 7.5]
                    half = num_seed_rotations // 2
                    rotation_angles = [seed_rotation_angle * (i - half + 0.5) for i in range(num_seed_rotations)]
            
            all_gen_poses = []
            all_recon_2_gen_mappings = []
            offset = 0  # Track index offset for gen poses
            
            # Prepare translation distances if provided
            if seed_translation_distance is not None:
                if isinstance(seed_translation_distance, (list, tuple)):
                    translation_distances = list(seed_translation_distance)
                    if len(translation_distances) != len(rotation_angles):
                        raise ValueError(f"seed_translation_distance list length ({len(translation_distances)}) "
                                       f"must match number of rotation angles ({len(rotation_angles)})")
                else:
                    # Use same translation magnitude for all, but sign depends on rotation angle
                    translation_distances = [
                        seed_translation_distance * np.sign(angle) if angle != 0 else 0.0
                        for angle in rotation_angles
                    ]
            else:
                translation_distances = [0.0] * len(rotation_angles)
            
            for angle_deg, trans_dist in zip(rotation_angles, translation_distances):
                # Start with original camera poses
                seed_poses = camera_poses
               
                # Step 1: Apply translation first (move left/right in camera's local x-axis)
                if trans_dist != 0.0:
                    # Create translation matrix in camera frame (x-axis is lateral)
                    trans_matrix = torch.tensor([
                        [1, 0, 0, trans_dist],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ], dtype=camera_poses.dtype, device=camera_poses.device)
                    
                    # Apply translation: pose @ trans_matrix (translate in camera's local frame)
                    seed_poses = torch.einsum('bnik,kj->bnij', seed_poses, trans_matrix)
                
                # Step 2: Apply rotation around y-axis
                angle_rad = np.radians(angle_deg)
                
                # Create rotation matrix around y-axis
                rot_matrix = torch.tensor([
                    [np.cos(angle_rad), 0, np.sin(angle_rad), 0],
                    [0, 1, 0, 0],
                    [-np.sin(angle_rad), 0, np.cos(angle_rad), 0],
                    [0, 0, 0, 1]
                ], dtype=camera_poses.dtype, device=camera_poses.device)
                
                # Apply rotation to all camera poses: pose @ rot_matrix (rotate in camera's local frame)
                seed_poses = torch.einsum('bnik,kj->bnij', seed_poses, rot_matrix)
                
                # Generate poses for this seed
                seed_gen_poses, seed_recon_2_gen_mapping = generate_novel_straight_rotated_poses(
                    seed_poses, views_per_interval, 
                    device, 
                    forward=forward,
                    rotate_angle=rotate_angle
                )
                
                all_gen_poses.append(seed_gen_poses)
                
                # Update mapping with offset for this seed's poses
                updated_mapping = {}
                for recon_idx, gen_indices in seed_recon_2_gen_mapping.items():
                    updated_mapping[recon_idx] = [idx + offset for idx in gen_indices]
                all_recon_2_gen_mappings.append(updated_mapping)
                
                offset += seed_gen_poses.shape[1]  # Update offset for next seed
            
            # Concatenate all generated poses
            gen_poses = torch.cat(all_gen_poses, dim=1)
            
            # Merge all mappings
            recon_2_gen_mapping = {}
            for mapping in all_recon_2_gen_mappings:
                for recon_idx, gen_indices in mapping.items():
                    if recon_idx not in recon_2_gen_mapping:
                        recon_2_gen_mapping[recon_idx] = []
                    recon_2_gen_mapping[recon_idx].extend(gen_indices)
        else:
            # Standard single-seed mode
            gen_poses, recon_2_gen_mapping = generate_novel_straight_rotated_poses(
                camera_poses, views_per_interval, 
                device, 
                forward=forward,
                rotate_angle=rotate_angle,
                lateral_translation=seed_translation_distance if seed_translation_distance is not None else 0.0,
            )
    return gen_poses, recon_2_gen_mapping


def create_voxel_prediction(pts3d_in_velo, has_semantic, semantic_2ds_th, conf_th, 
                            grid_size, voxel_origin, voxel_size, 
                            n_classes, other_class, empty_class=0):
    """
    Create voxel prediction from 3D points with optional semantic information.
    
    Args:
        pts3d_in_velo: 3D points in velodyne coordinates
        has_semantic: Boolean indicating if semantic information is available
        semantic_2ds_th: Semantic 2D predictions for current view
        conf_th: Confidence threshold for filtering points
        grid_size: Size of the voxel grid
        voxel_origin: Origin of the voxel grid
        voxel_size: Size of each voxel
        n_classes: Number of semantic classes
        empty_class: Class label for empty voxels (default: 0)
    
    Returns:
        torch.Tensor: Voxel prediction tensor with semantic labels
    """
    # Create geometry voxels (binary occupancy)
    geometry_2d_logits = torch.nn.functional.one_hot(
        torch.ones(pts3d_in_velo.shape[0], dtype=torch.long, device=pts3d_in_velo.device),
        num_classes=2
    ).float()
    vox_geometry, _ = pointcloud2voxel(pts3d_in_velo, geometry_2d_logits, grid_size,
                                        voxel_origin, voxel_size)
    # Initialize with empty_class instead of 0
    voxel_pred = torch.full(vox_geometry.shape[:3], empty_class, dtype=torch.long, device=vox_geometry.device)
    voxel_pred[vox_geometry > 0] = other_class # n_classes is unclassified
    
  
    # Add semantic information if available
    if has_semantic:
        # Filter out points with empty_class or other_class labels
        valid_semantic_mask = (semantic_2ds_th != empty_class) & (semantic_2ds_th != other_class)
        # valid_semantic_mask =  semantic_2ds_th != other_class
        
        # Only process points with valid semantic labels
        if valid_semantic_mask.any():
            pts3d_valid = pts3d_in_velo[valid_semantic_mask]
            semantic_2ds_valid = semantic_2ds_th[valid_semantic_mask]
            conf_valid = conf_th[valid_semantic_mask]
            
            # Create one-hot encoded semantic labels for valid points
            semantic_2d_logits = torch.nn.functional.one_hot(
                semantic_2ds_valid.long(), num_classes=n_classes
            ).float()
            
            # Weight by confidence: multiply each one-hot vector by its confidence value
            # Shape: (N_valid, n_classes) * (N_valid, 1) -> (N_valid, n_classes)
            semantic_2d_logits = semantic_2d_logits * (conf_valid.unsqueeze(-1) - 1.0)
            
            vox_occ, vox_sem = pointcloud2voxel(pts3d_valid, semantic_2d_logits, grid_size,
                                                voxel_origin, voxel_size)
        else:
            # No valid semantic points, return early with geometry-only voxels
            return voxel_pred
    else:
        # Set to arbitrary occupied class to just set as occupied
        semantic_2d_logits = torch.nn.functional.one_hot(
            torch.ones(pts3d_in_velo.shape[0], dtype=torch.long, 
                       device=pts3d_in_velo.device),
            num_classes=n_classes
        ).float()
        # Weight by confidence: multiply each one-hot vector by its confidence value
        # Shape: (N, n_classes) * (N, 1) -> (N, n_classes)
        semantic_2d_logits = semantic_2d_logits * (conf_th.unsqueeze(-1) - 1.0)
        
        vox_occ, vox_sem = pointcloud2voxel(pts3d_in_velo, semantic_2d_logits, grid_size,
                                            voxel_origin, voxel_size)

    vox_sem_argmax = torch.argmax(vox_sem, dim=-1)
    occ_mask = (vox_occ > 0) & (vox_sem_argmax != empty_class) & (vox_sem.sum(dim=-1) != 0)               
    voxel_pred[occ_mask] = vox_sem_argmax[occ_mask]
    
    return voxel_pred



def print_metrics_summary(total_ssc_metrics):
    """
    Print metrics summary for given threshold and metric names.
    
    Args:
        th: Threshold value
        total_ssc_metrics: Dictionary of SSCMetrics objects
        class_mapping: Class mapping object with id_2_kitti_classes
        metric_names: List of metric name prefixes (e.g., ['online', 'render'])
    """

    
    # Collect stats for all metrics
    all_stats = {}
    for key in total_ssc_metrics:
        all_stats[key] = total_ssc_metrics[key].get_stats()
    
    # Print main metrics line
    metrics_strs = []
    n_batches_strs = []
    for key in all_stats:
        stats = all_stats[key]
        precision = stats["precision"] * 100
        recall = stats["recall"] * 100
        iou = stats["iou"] * 100
        mIoU = stats["mIoU"] * 100
        class_names = stats["class_names"]
        n_batches = stats["n_batches"]
        metrics_strs = f"{precision:.2f}, {recall:.2f}, {iou:.2f}, {mIoU:.2f}"
        n_batches_strs = str(n_batches)
    
        print(f">>> [{key}]: {metrics_strs} {n_batches_strs}")
    
    class_name_str = ", ".join(f"{class_names[i]:5.5}" for i in range(len(class_names)))
    print(f"  Class:  {class_name_str}")
        
    for key in all_stats:
        iou_per_class = all_stats[key]["iou_per_class"]
        iou_str = ", ".join(f"{iou_per_class[i]*100:05.2f}" for i in range(len(iou_per_class)))
        print(f"  {key.capitalize():8s}: {iou_str}")
                

def pointcloud2voxel(pc: torch.Tensor, semantic_2d_logits, voxel_dim, voxel_origin: int, voxel_size=0.2, filter_outlier=True):
    n, _ = pc.shape
    
    
    # pc_grid = (pc + half_size) * (voxel_size - 1.) / grid_size  # thanks @heathentw
    pc_grid = (pc - voxel_origin) / voxel_size
    
    indices_floor = torch.floor(pc_grid)
    indices = indices_floor.long()
    # batch_indices = torch.arange(b, device=pc.device)
    # batch_indices = torch.reshape(batch_indices, (1, -1))
    # batch_indices = torch.tile(batch_indices, (1, n))
    # batch_indices = torch.reshape(batch_indices, (1, -1))
    # indices = torch.cat((batch_indices, indices), 2)
    # indices = torch.reshape(indices, (-1, 4))
    r = pc_grid - indices_floor
    rr = (1. - r, r)
    # if filter_outlier:
    #     valid = valid.flatten()
    #     indices = indices[valid]

    def interpolate_scatter3d(pos):
        updates_raw = rr[pos[0]][..., 0] * rr[pos[1]][..., 1] * rr[pos[2]][..., 2]
        updates = updates_raw.flatten()

        # if filter_outlier:
        #     updates = updates[valid]

        indices_shift = torch.tensor([pos]).to(pc.device)
        indices_loc = indices + indices_shift
        # out_shape = (b,) + (voxel_size,) * 3
        valid = (indices_loc[:, 0] >= 0) & (indices_loc[:, 0] < voxel_dim[0]) \
                & (indices_loc[:, 1] >= 0) & (indices_loc[:, 1] < voxel_dim[1]) \
                & (indices_loc[:, 2] >= 0) & (indices_loc[:, 2] < voxel_dim[2])
        
        out = torch.zeros(*voxel_dim).to(pc.device).flatten()
        indices_loc_flat = indices_loc[:, 0] * voxel_dim[1] * voxel_dim[2] + indices_loc[:, 1] * voxel_dim[2] + indices_loc[:, 2]
        out.scatter_add_(0, indices_loc_flat[valid], updates[valid])
        
        # Handle semantic logits
        num_classes = semantic_2d_logits.shape[1]
        out_sem = torch.zeros((*voxel_dim, num_classes)).to(pc.device)
        out_sem = out_sem.view(-1, num_classes)
        
        # Expand indices for each semantic class
        expanded_indices = indices_loc_flat[valid].unsqueeze(-1).expand(-1, num_classes)
        expanded_updates = updates[valid].unsqueeze(-1) * semantic_2d_logits[valid]
        # expanded_updates = semantic_2d_logits[valid]
        
        # out_sem = out_sem.view(*voxel_dim, num_classes)
        # Use max aggregation instead of sum - keeps highest confidence prediction per voxel
        # This is more robust than summing, which can accumulate errors
        out_sem.scatter_reduce_(0, expanded_indices, expanded_updates, reduce='amax')
        out_sem = out_sem.view(*voxel_dim, num_classes)
        
        return out.view(*voxel_dim), out_sem

    voxels = [interpolate_scatter3d([k, j, i]) for k in range(2) for j in range(2) for i in range(2)]
    occupancy, semantics = zip(*voxels)
    occupancy = sum(occupancy)
    semantics = sum(semantics)
    
    occupancy = torch.clamp(occupancy, 0., 1.)
    return occupancy, semantics



def save_semantic_2d_images(semantic_2ds, save_dir, color_map, verbose=False, rgb_images=None):
    """Save semantic 2D visualizations, optionally stacking RGB above each mask."""

    def _to_numpy(image):
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        return np.asarray(image)

    def _rgb_to_bgr_uint8(image):
        image = _to_numpy(image)
        if image.dtype != np.uint8:
            image_min, image_max = float(image.min()), float(image.max())
            if image_min >= -1.05 and image_max <= 1.05:
                image = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0.0, 1.0)
                image = (image * 255.0).round().astype(np.uint8)
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rgb_available = rgb_images is not None
    if rgb_available:
        rgb_images = [_to_numpy(img) for img in rgb_images]

    for i, semantic_2d in enumerate(semantic_2ds):
        assert len(semantic_2d.shape) == 2

        semantic_2d_colored = color_map[semantic_2d]
        if semantic_2d_colored.dtype != np.uint8:
            semantic_2d_colored = semantic_2d_colored.astype(np.uint8)
        semantic_2d_bgr = cv2.cvtColor(semantic_2d_colored, cv2.COLOR_RGBA2BGR)

        if rgb_available:
            if i >= len(rgb_images):
                raise ValueError("rgb_images does not have enough views to match semantic_2ds")
            rgb_bgr = _rgb_to_bgr_uint8(rgb_images[i])
            if rgb_bgr.shape[1] != semantic_2d_bgr.shape[1]:
                raise ValueError("RGB and semantic images must share the same width for concatenation")
            combined = np.concatenate((rgb_bgr, semantic_2d_bgr), axis=0)
        else:
            combined = semantic_2d_bgr

        cv2.imwrite(f"{save_dir}/semantic_2d_colored_frame_{(i):04d}.png", combined)
        if verbose:
            print(f"Saved {save_dir}/semantic_2d_colored_frame_{(i):04d}.png")


def normalize_poses(poses, scene_scale=[25.6, 6.4, 51.2], origin=[0, 0, -25.6]):
    assert poses.shape[-2:] == (4, 4) and poses.ndim == 3, f"poses.shape: {poses.shape}"
    scene_scale = np.array(scene_scale).reshape(1, -3)
    origin = np.array(origin).reshape(1, -3)
    normed_camera_poses = np.copy(poses)
    normed_camera_poses[:, :3, 3] = (normed_camera_poses[:, :3, 3] - origin) / scene_scale
    return normed_camera_poses
    

def save_rgb_images(rgb_tensors, idx, save_dir: str, stem: str, verbose: bool = False):
    """Save RGB tensor(s) to PNG images.

    The input tensor values are expected to be in the range ``[0, 1]``.

    Supported tensor shapes::
        (B, N, H, W, 3)
        (B, 3, H, W)
        (H, W, 3)

    For each frame an image ``{stem}_XXXX.png`` is written inside
    ``save_dir``.

    Args:
        rgb_tensors: A ``torch.Tensor`` or a list/tuple of tensors containing RGB
            data in the range ``[0, 1]``.
        save_dir: Target directory where PNG files will be stored.
        stem:     Prefix for output filenames.
        verbose:  If ``True`` prints the path of every saved image.
    """
    # Accept both single tensor and iterable of tensors
    if isinstance(rgb_tensors, (list, tuple)):
        tensors = rgb_tensors
    else:
        tensors = [rgb_tensors]

    os.makedirs(save_dir, exist_ok=True)

    for t_id, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("rgb_tensors must be torch.Tensor or list/tuple of Tensors")

        t = tensor.detach().cpu().clamp(0.0, 1.0) # # (B, N, H, W, 3)

      
  
        # for f_id, img in enumerate(t):
        for f_id, view_idx in enumerate(idx):
            img = t[0, f_id]
            img_uint8 = (img * 255.0).round().to(torch.uint8).numpy()  # RGB
            img_bgr = img_uint8[..., ::-1]  # Convert to BGR for OpenCV
            filename = os.path.join(
                save_dir,
                f"{stem}_{t_id:02d}_{view_idx:03d}.png" if len(tensors) > 1 else f"{stem}_{view_idx:03d}.png",
            )
            cv2.imwrite(filename, img_bgr)
            if verbose:
                print(f"Saved {filename}")

def get_ray_map_lsvm(c2w, fxfycxcy, h, w, device=None):
    """
    Args:
        c2w (torch.tensor): [b, v, 4, 4]
        fxfycxcy (torch.tensor): [b, v, 4]
        h (int): height of the image
        w (int): width of the image
    Returns:
        ray_o (torch.tensor): [b, v, 3, h, w]
        ray_d (torch.tensor): [b, v, 3, h, w]
    """

    b, v = c2w.size()[:2]
    c2w = c2w.reshape(b * v, 4, 4)

    # Infer device/dtype from c2w unless explicitly provided
    dev = c2w.device if device is None else torch.device(device)
    dtype = c2w.dtype

    # Move tensors to common device/dtype
    c2w = c2w.to(device=dev, dtype=dtype)
    fxfycxcy = fxfycxcy.reshape(b * v, 4).to(device=dev, dtype=dtype)

    # Validate camera intrinsics to prevent division by zero
    # Ensure reasonable focal lengths (e.g., between 1-2000 pixels)
    fx = torch.clamp(fxfycxcy[:, 0:1], 1.0, 2000.0)
    fy = torch.clamp(fxfycxcy[:, 1:2], 1.0, 2000.0)
    cx = fxfycxcy[:, 2:3]
    cy = fxfycxcy[:, 3:4]

    # Create pixel grid
    y, x = torch.meshgrid(
        torch.arange(h, device=dev, dtype=dtype),
        torch.arange(w, device=dev, dtype=dtype),
        indexing="ij",
    )
    x = x[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)
    y = y[None, :, :].expand(b * v, -1, -1).reshape(b * v, -1)

    # Normalize to camera rays with validated intrinsics
    x = (x + 0.5 - cx) / fx
    y = (y + 0.5 - cy) / fy
    
    # Clamp normalized coordinates to prevent extremely large values
    x = torch.clamp(x, -1.0, 1.0)
    y = torch.clamp(y, -1.0, 1.0)
    
    z = torch.ones_like(x)

    ray_d = torch.stack([x, y, z], dim=2)  # [b*v, h*w, 3]
    # Transform directions to world using cam2world rotation
    ray_d = torch.bmm(ray_d, c2w[:, :3, :3].transpose(1, 2))  # [b*v, h*w, 3]
    ray_d = ray_d / torch.norm(ray_d, dim=2, keepdim=True).clamp_min(1e-6)  # [b*v, h*w, 3]
    ray_o = c2w[:, :3, 3][:, None, :].expand_as(ray_d)  # [b*v, h*w, 3]

    ray_o = rearrange(ray_o, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)
    ray_d = rearrange(ray_d, "(b v) (h w) c -> b v c h w", b=b, v=v, h=h, w=w, c=3)

    oxd = torch.cross(ray_o, ray_d, dim=2)
    ray_map = torch.cat([oxd, ray_d], dim=2)
    
    # Optional: Check for NaN values and replace with zeros
    if torch.isnan(ray_map).any():
        ray_map = torch.where(torch.isnan(ray_map), torch.zeros_like(ray_map), ray_map)
    
    return ray_map


def crop_resize_if_necessary(image, depthmap, intrinsics, resolution, 
                              rng=None, info=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    # downscale with lanczos interpolation so that image.size == resolution
    # cropping centered on the principal point
    W, H = image.size
    cx, cy = intrinsics[:2, 2].round().astype(int)
    min_margin_x = min(cx, W-cx)
    min_margin_y = min(cy, H-cy)
    assert min_margin_x > W/5, f'Bad principal point in view={info}'
    assert min_margin_y > H/5, f'Bad principal point in view={info}'
    # the new window will be a rectangle of size (2*min_margin_x, 2*min_margin_y) centered on (cx,cy)
    l, t = cx - min_margin_x, cy - min_margin_y
    r, b = cx + min_margin_x, cy + min_margin_y
    crop_bbox = (l, t, r, b)
    image, depthmap, intrinsics = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)

    # high-quality Lanczos down-scaling
    target_resolution = np.array(resolution)
    image, depthmap, intrinsics = cropping.rescale_image_depthmap(image, depthmap, intrinsics, target_resolution) # slightly scale the image a bit larger than the target resolution
    # actual cropping (if necessary) with bilinear interpolation
    intrinsics2 = cropping.camera_matrix_of_crop(intrinsics, image.size, resolution, offset_factor=0.5)
    crop_bbox = cropping.bbox_from_intrinsics_in_out(intrinsics, intrinsics2, resolution)
    image, depthmap, intrinsics2 = cropping.crop_image_depthmap(image, depthmap, intrinsics, crop_bbox)
    return image, depthmap, intrinsics2


def visualize_depth(depth, save_path, colormap="Spectral",
                    min_val=0, max_val=80, mask=None):
    depth_colored = colorize_depth_maps(
        depth, min_val, max_val, cmap=colormap
    ).squeeze()  # [3, H, W], value in (0, 1)
    if mask is not None:
        depth_colored = depth_colored * mask[np.newaxis, :, :]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    depth_colored_img = Image.fromarray(depth_colored_hwc)
    # if mask is not None:
    #     for h, w in np.column_stack(np.where(mask)):
    #         cv2.circle(depth_colored_img, (w, h), radius=5, color=(0, 0, 255))
    depth_colored_img.save(save_path)



def project_lidar_world2camera(pc_world, img_w, img_h, camera_pose, cam_K, filter_outliers=True):
    """
    Projects a LiDAR point cloud from the world coordinate frame onto the camera image plane.

    This function transforms the input LiDAR points from world coordinates into the camera coordinate system by
    applying the inverse of the provided camera pose. It then projects the 3D points onto the 2D image plane using
    the camera intrinsic matrix. Optionally, it filters out points that are behind the camera or that project outside
    the image boundaries.

    Args:
        pc_world (np.ndarray): A 2D array of shape (N, 3) representing the 3D LiDAR points in world coordinates.
        img_w (int): The width of the target image in pixels.
        img_h (int): The height of the target image in pixels.
        camera_pose (np.ndarray): A 4x4 transformation matrix representing the camera pose (from camera to world coordinates).
        cam_K (np.ndarray): A 3x3 intrinsic matrix of the camera.
        filter_outliers (bool, optional): If True, filters out points with non-positive depth values or those projecting
                                          outside the image boundaries. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - depthmap (np.ndarray): A 2D array of shape (img_h, img_w) containing the depth values of the projected points.
            - points2d_camera (np.ndarray): A 2D array of shape (M, 2) with the 2D pixel coordinates of the projected points.
            - points3d_camera (np.ndarray): A 2D array of shape (M, 3) with the 3D coordinates of the points in the camera frame.
            - inliner_indices_arr (np.ndarray): A 1D array of indices corresponding to the valid inlier points in the original point cloud.
    """
    trans_lidar_to_camera = np.linalg.inv(camera_pose)
    points3d_lidar = pc_world
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + trans_lidar_to_camera[:3, 3].reshape(3, 1)

    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = cam_K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T
    points3d_camera = points3d_camera.T
    
    if filter_outliers:
        condition = np.logical_and(
            (points2d_camera[:, 1] < img_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < img_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = points3d_camera[condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    # depthmap = np.zeros((img_h, img_w))
    xs = np.round(points2d_camera[:, 0]).clip(0, img_w - 1).astype(np.int32)
    ys = np.round(points2d_camera[:, 1]).clip(0, img_h - 1).astype(np.int32)
    depthmap = np.zeros((img_h, img_w))
    depthmap[ys, xs] = points3d_camera[:, 2]    
    # Before the final assignment

    # depthmap = np.full((img_h, img_w), np.inf)  # Initialize with infinity
    # for i in range(len(points2d_camera)):
    #     x, y = xs[i], ys[i]
    #     depthmap[y, x] = min(depthmap[y, x], points3d_camera[i, 2])
    # depthmap[depthmap == np.inf] = 0  # Replace unprojected pixels with 0
    
    return depthmap, points2d_camera, points3d_camera, inliner_indices_arr


def save_depth_as_colored_png(gt_depth, invalid_value, save_png_path=None, 
    colormap=cv2.COLORMAP_JET, min_depth=None, max_depth=None):
    """
    Save depth map as colored PNG image with invalid values set to black.

    Args:
        gt_depth (np.ndarray): Depth map array
        invalid_value (float): Value representing invalid depth
        save_png_path (str): Path to save the output PNG
        colormap (int): OpenCV colormap to use (default: JET)
        min_depth (float, optional): Minimum depth value for normalization
        max_depth (float, optional): Maximum depth value for normalization
    """
    # Normalize valid depth values to [0, 1]
    valid_mask = gt_depth > 0
    if valid_mask.any():
        min_val = min_depth if min_depth is not None else gt_depth[valid_mask].min()
        max_val = max_depth if max_depth is not None else gt_depth[valid_mask].max()
        depth_normalized = np.clip((gt_depth - min_val) / (max_val - min_val), 0, 1)
    else:
        depth_normalized = np.zeros_like(gt_depth)

    # Apply colormap and convert to 8-bit
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), colormap)

    if invalid_value is not None:
        # Set invalid values to black
        depth_colored[gt_depth == invalid_value] = 0

    # Save as PNG
    if save_png_path is not None:
        cv2.imwrite(save_png_path, depth_colored)
    else:
        return depth_colored


def depth2rgb(depth, valid_mask=None, save_png_path=None, 
    colormap=cv2.COLORMAP_JET, min_depth=None, max_depth=None):
    """
    Save depth map as colored PNG image with invalid values set to black.

    Args:
        gt_depth (np.ndarray): Depth map array
        invalid_value (float): Value representing invalid depth
        save_png_path (str): Path to save the output PNG
        colormap (int): OpenCV colormap to use (default: JET)
        min_depth (float, optional): Minimum depth value for normalization
        max_depth (float, optional): Maximum depth value for normalization
    """
    # Normalize valid depth values to [0, 1]
    min_val = min_depth if min_depth is not None else depth[valid_mask].min()
    max_val = max_depth if max_depth is not None else depth[valid_mask].max()
    depth_normalized = np.clip((depth - min_val) / (max_val - min_val), 0, 1)

    # Apply colormap and convert to 8-bit
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), colormap)

    if valid_mask is not None:
        # Set invalid values to black
        depth_colored[~valid_mask] = 0

    # Save as PNG
    if save_png_path is not None:
        cv2.imwrite(save_png_path, depth_colored)
    else:
        return depth_colored

def voxel_to_pointcloud(voxel_label, voxel_size, voxel_origin, colors=None):
    occupied_mask = (voxel_label != 255) & (voxel_label != 0)
    coords = np.argwhere(occupied_mask)
    points = coords * voxel_size + voxel_origin

    point_labels = voxel_label[coords[:, 0], coords[:, 1], coords[:, 2]]
    point_colors = colors[point_labels.astype(np.uint16)]

    return points, point_colors


def save_pcd_as_txt(filename, points, features=None):
    with open(filename, 'w') as f:
        # Get number of columns from first point
        n_cols = points.shape[1]

        # Write each point's coordinates
        for i in range(len(points)):
            # Write all columns space-separated
            line = " ".join([f"{points[i,j]}" for j in range(n_cols)])

            # Add colors if provided
            if features is not None:
                line += " " + " ".join([f"{features[i,j]}" for j in range(features.shape[1])])

            f.write(line + "\n")

def voxel_to_pointcloud(voxel_label, voxel_size, voxel_origin, colors=None):
    occupied_mask = (voxel_label != 255) & (voxel_label != 0)

    coords = np.argwhere(occupied_mask)
    points = coords * voxel_size + voxel_origin

    if colors is not None:
        point_labels = voxel_label[coords[:, 0], coords[:, 1], coords[:, 2]]
        point_colors = colors[point_labels.astype(np.uint16)]
    else:
        point_colors = None

    return points, point_colors

def voxelize_points(points, voxel_origin, voxel_size, voxel_dim):
    voxelized_points = (points - voxel_origin) / voxel_size
    voxelized_points = np.floor(voxelized_points).astype(np.int32)
    keep_ind = (voxelized_points[:, 0] >= 0) & (voxelized_points[:, 0] < voxel_dim[0]) & \
               (voxelized_points[:, 1] >= 0) & (voxelized_points[:, 1] < voxel_dim[1]) & \
               (voxelized_points[:, 2] >= 0) & (voxelized_points[:, 2] < voxel_dim[2])
    voxelized_points = voxelized_points[keep_ind]
    return voxelized_points, keep_ind

def transform_points(T, points):
    h_points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    t_points = np.matmul(T, h_points.T).T
    points = t_points[:, :3]
    return points

def transform_points_torch(T, points):
    if T.dtype != points.dtype:
        T = T.to(dtype=points.dtype)
    if T.device != points.device:
        T = T.to(device=points.device)
    h_points = torch.cat([points, torch.ones_like(points[:, :1], device=points.device)], dim=-1)
    t_points = torch.matmul(T, h_points.T).T
    points = t_points[:, :3]
    return points


def vox2pix(cam_E, cam_k,
            vox_origin, voxel_size,
            img_W, img_H,
            scene_size):
    """
    compute the 2D projection of voxels centroids

    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    if fusion is None:
        raise ImportError(
            "occany.utils.fusion is not available (likely due to missing numba/llvmlite). "
            "vox2pix() requires it. Install numba/llvmlite or avoid calling vox2pix in this environment."
        )

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return projected_pix, fov_mask, pix_z


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                np.logical_and(pix_x < max_x,
                np.logical_and(pix_y >= min_y,
                np.logical_and(pix_y < max_y,
                pix_z > 0))))
    return valid_pix

def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies

    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ="NYU" or "kitti" (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size

    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists
