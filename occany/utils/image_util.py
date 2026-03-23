# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
# Last modified: 2024-05-24
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------


import matplotlib
import numpy as np
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms as tvf
from torchvision.transforms.functional import resize
from PIL import Image
from sam2.utils.transforms import SAM2Transforms
import torch.nn.functional as F
import groundingdino.datasets.transforms as T
from torchvision.transforms import v2

RESAMPLING = Image.Resampling if hasattr(Image, "Resampling") else Image
LANCZOS_RESAMPLE = RESAMPLING.LANCZOS
BICUBIC_RESAMPLE = RESAMPLING.BICUBIC


def crop_resize_if_necessary(image, resolution=None, rng=None, info=None):
    """ This function:
        - first downsizes the image with LANCZOS inteprolation,
            which is better than bilinear interpolation in
    """
    if resolution is None:
        raise ValueError("resolution must be provided")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    resolution = tuple(int(v) for v in resolution)
    W, H = image.size

    # transpose the resolution if necessary
    if H > 1.1*W and resolution[0] > resolution[1]:
        # image is portrait mode
        resolution = resolution[::-1]
        print(f"WARNING: image is portrait in view={info}")
    elif 0.9 < H/W < 1.1 and resolution[0] != resolution[1] and rng is not None:
        # image is square, so we chose (portrait, landscape) randomly
        if rng.integers(2):
            resolution = resolution[::-1]
        print(f"WARNING: image is square in view={info}")

    input_resolution = np.array(image.size, dtype=np.float32)
    target_resolution = np.array(resolution, dtype=np.float32)
    scale_final = max(target_resolution / input_resolution) + 1e-8
    resized_resolution = np.floor(input_resolution * scale_final).astype(int)
    resample = LANCZOS_RESAMPLE if scale_final < 1 else BICUBIC_RESAMPLE
    image = image.resize(tuple(resized_resolution.tolist()), resample=resample)

    crop_left = max(0, int((resized_resolution[0] - resolution[0]) / 2))
    crop_top = max(0, int((resized_resolution[1] - resolution[1]) / 2))
    crop_bbox = (
        crop_left,
        crop_top,
        crop_left + resolution[0],
        crop_top + resolution[1],
    )
    return image.crop(crop_bbox)


def convert_images_to_uint8_hwc(images: torch.Tensor) -> np.ndarray:
    """Convert [-1, 1] CHW images to uint8 HWC numpy arrays."""
    images_hwc = images.permute(0, 2, 3, 1).clamp(-1.0, 1.0)
    images_uint8 = ((images_hwc + 1.0) * 127.5).round().to(torch.uint8)
    return images_uint8.cpu().numpy()

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_SAM2_transforms(resolution=512):
    """Create SAM2 transforms with specified resolution.
    
    Args:
        resolution: Image resolution for SAM2 (default: 512, supports 768, 1024, etc.)
    
    Returns:
        SAM2Transforms object configured with the specified resolution
    """
    return SAM2Transforms(resolution=resolution,
                         mask_threshold=0.0,
                         max_hole_area=0.0,
                         max_sprinkle_area=0.0)

def get_SAM3_transforms(resolution=1008):
    """Create SAM3 transforms with specified resolution.
    
    Args:
        resolution: Image resolution for SAM3 (default: 1008, supports 1008, 1536, etc.)
    
    Returns:
        SAM3Transforms object configured with the specified resolution
    """
    return v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

# Backward compatibility: keep the default 512 resolution transform
SAM2_transforms = get_SAM2_transforms(resolution=512)

GroundingDinoImgNorm = T.Compose(
                    [
                        T.RandomResize([800], max_size=1333),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ]
                )

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def camera_to_pose_encoding(
    camera,
    pose_encoding_type="absT_quaR",
):
    """
    Inverse to pose_encoding_to_camera
    camera: opencv, cam2world
    """
    if pose_encoding_type == "absT_quaR":
        # Extract rotation and translation components based on camera's dimensions
        if camera.ndim == 3:  # [N, 4, 4]
            rotation = camera[:, :3, :3]
            translation = camera[:, :3, 3]
        elif camera.ndim == 4:  # [B, N, 4, 4]
            rotation = camera[:, :, :3, :3]
            translation = camera[:, :, :3, 3]
        
        # Convert rotation matrices to quaternions and concatenate with translation
        quaternion_R = matrix_to_quaternion(rotation)
        pose_encoding = torch.cat([translation, quaternion_R], dim=-1)
    else:
        raise ValueError(f"Unknown pose encoding {pose_encoding_type}")

    return pose_encoding


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def visualize_depth(depth, save_path, colormap="Spectral", min_val=0, max_val=1, mask=None):
    depth_colored = colorize_depth_maps(
        depth, min_val, max_val, cmap=colormap
    ).squeeze()  # [3, H, W], value in (0, 1)
    if mask is not None:
        depth_colored = depth_colored * mask[np.newaxis, :, :]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    depth_colored_img = Image.fromarray(depth_colored_hwc)
    depth_colored_img.save(save_path)
    
    
    
def resize_max_res(
    img: torch.Tensor,
    max_edge_resolution: int,
    resample_method: InterpolationMode = InterpolationMode.BILINEAR,
) -> torch.Tensor:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == img.dim(), f"Invalid input shape {img.shape}"

    original_height, original_width = img.shape[-2:]
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(img, (new_height, new_width), resample_method, antialias=True)
    return resized_img


def get_tv_resample_method(method_str: str) -> InterpolationMode:
    resample_method_dict = {
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "nearest": InterpolationMode.NEAREST_EXACT,
        "nearest-exact": InterpolationMode.NEAREST_EXACT,
    }
    resample_method = resample_method_dict.get(method_str, None)
    if resample_method is None:
        raise ValueError(f"Unknown resampling method: {resample_method}")
    else:
        return resample_method
