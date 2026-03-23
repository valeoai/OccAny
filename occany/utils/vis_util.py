import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
from PIL import Image

try:
    import mayavi
    from mayavi import mlab
except ImportError:
    print("[WARNING] Mayavi not installed")
    mayavi = None
    mlab = None

engine = None
    
# import torch
# import torch.nn.functional as F


KITTI_COLORS = np.array(
    [
        [0, 0, 0, 255],
        [100, 150, 245, 255],
        [100, 230, 245, 255],
        [30, 60, 150, 255],
        [80, 30, 180, 255],
        [100, 80, 250, 255],
        [255, 30, 30, 255],
        [255, 40, 200, 255],
        [150, 30, 90, 255],
        [255, 0, 255, 255],
        [255, 150, 255, 255],
        [75, 0, 75, 255],
        [175, 0, 75, 255],
        [255, 200, 0, 255],
        [255, 120, 50, 255],
        [0, 175, 0, 255],
        [135, 60, 0, 255],
        [150, 240, 80, 255],
        [255, 240, 150, 255],
        [255, 0, 0, 255],
        [255, 255, 255, 255],
    ],
    dtype=np.uint8,
)

REFERENCE_CLASS_COLORS = {
    "empty": [0, 0, 0, 255],
    "car": [100, 150, 245, 255],
    "bicycle": [100, 230, 245, 255],
    "motorcycle": [30, 60, 150, 255],
    "truck": [80, 30, 180, 255],
    "other-vehicle": [100, 80, 250, 255],
    "person": [255, 30, 30, 255],
    "rider": [255, 40, 200, 255],
    "motorcyclist": [150, 30, 90, 255],
    "road": [255, 0, 255, 255],
    "parking": [255, 150, 255, 255],
    "sidewalk": [75, 0, 75, 255],
    "other-ground": [175, 0, 75, 255],
    "building": [255, 200, 0, 255],
    "fence": [255, 120, 50, 255],
    "vegetation": [0, 175, 0, 255],
    "trunk": [135, 60, 0, 255],
    "terrain": [150, 240, 80, 255],
    "pole": [255, 240, 150, 255],
    "traffic-sign": [255, 0, 0, 255],
    "unknown": [255, 255, 255, 255],
}

NUSCENES_COLOR_KEYS = [
    "unknown",
    "fence",
    "bicycle",
    "other-vehicle",
    "car",
    "other-vehicle",
    "motorcycle",
    "person",
    "traffic-sign",
    "other-vehicle",
    "truck",
    "road",
    "other-ground",
    "sidewalk",
    "terrain",
    "building",
    "vegetation",
    "empty",
]

OCC3D_COLORS = np.array([REFERENCE_CLASS_COLORS[key] for key in NUSCENES_COLOR_KEYS], dtype=np.uint8)

DATASET_CONFIGS = {
    "kitti": {
        "colors": KITTI_COLORS,
        "empty_class": 0,
        "other_class": 20,
        "voxel_size": 0.2,
    },
    "nuscenes": {
        "colors": OCC3D_COLORS,
        "empty_class": 17,
        "other_class": 0,
        "voxel_size": 0.4,
    },
    "waymo": {
        "colors": OCC3D_COLORS,
        "empty_class": 17,
        "other_class": 0,
        "voxel_size": 0.4,
    },
    "ddad": {
        "colors": OCC3D_COLORS,
        "empty_class": 17,
        "other_class": 0,
        "voxel_size": 0.4,
    },
    "pandaset": {
        "colors": OCC3D_COLORS,
        "empty_class": 17,
        "other_class": 0,
        "voxel_size": 0.4,
    },
}

DATASET_BY_SHAPE = {
    (256, 256, 32): "kitti",
    (200, 200, 16): "nuscenes",
}

IGNORE_LABEL = 255
GT_NONOVERLAP_DARKEN_FACTOR = 0.5

T_CAM_TO_VOXEL = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

VOXEL_ORIGIN_BY_SHAPE = {
    (256, 256, 32): np.array([0.0, -25.6, -2.0], dtype=np.float32),
    (200, 200, 16): np.array([-40.0, -40.0, -1.0], dtype=np.float32),
}

CAMERA_FRUSTUM_COLORS: List[Tuple[float, float, float]] = [
    (0.55, 0.20, 0.65),  # dark violet
    (0.20, 0.20, 0.65),  # dark indigo
    (0.65, 0.15, 0.22),  # dark crimson
    (0.50, 0.78, 0.95),  # sky blue (bright)
    (0.95, 0.62, 0.15),  # bright orange (bright)
    (0.35, 0.65, 0.90),  # sky blue
    (0.90, 0.50, 0.05),  # bright orange
]

VIS_DIR = Path(__file__).resolve().parent
CAR_IMAGE_PATH = VIS_DIR / "car.png"

NUSCENES_CAMERA_SLOT_BY_INDEX = {
    0: "front",
    1: "left",
    2: "right",
    3: "back",
    4: "back_left",
    5: "back_right",
}

WAYMO_CAMERA_SLOT_BY_INDEX = {
    0: "front",
    1: "front_left",
    2: "front_right",
    3: "left",
    4: "right",
}

DDAD_CAMERA_SLOT_BY_INDEX = {
    0: "front",
    1: "front_left",
    2: "front_right",
    3: "back_left",
    4: "back_right",
    5: "back",
}

PANDASET_CAMERA_SLOT_BY_INDEX = {
    0: "front",
    1: "front_left",
    2: "front_right",
    3: "left",
    4: "right",
    5: "back",
}

SURROUND_CAMERA_SLOT_BY_DATASET: Dict[str, Dict[int, str]] = {
    "waymo": WAYMO_CAMERA_SLOT_BY_INDEX,
    "ddad": DDAD_CAMERA_SLOT_BY_INDEX,
    "pandaset": PANDASET_CAMERA_SLOT_BY_INDEX,
}

SURROUND_CAMERA_GRID_SLOTS = {
    "front_left": (0, 0),
    "front": (0, 1),
    "front_right": (0, 2),
    "left": (1, 0),
    "right": (1, 2),
    "back_left": (2, 0),
    "back": (2, 1),
    "back_right": (2, 2),
}


def infer_voxel_size(voxels: np.ndarray) -> float:
    """Infer voxel size from known OccAny output grid shapes."""
    shape = tuple(int(v) for v in voxels.shape)
    if shape == (256, 256, 32):
        return 0.2
    if shape == (200, 200, 16) or shape == (200, 200, 24):
        return 0.4
    return 0.2

def infer_voxel_origin(voxels: np.ndarray) -> np.ndarray:
    """Infer voxel-grid origin from known OccAny output grid shapes."""
    shape = (int(voxels.shape[0]), int(voxels.shape[1]), int(voxels.shape[2]))
    return VOXEL_ORIGIN_BY_SHAPE.get(shape, np.zeros(3, dtype=np.float32)).copy()

def resolve_dataset_config(dataset_name: str) -> Dict[str, object]:
    """Resolve dataset-specific rendering settings."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'; choose from {list(DATASET_CONFIGS.keys())}")
    dataset_config = DATASET_CONFIGS[dataset_name].copy()
    dataset_config["name"] = dataset_name
    return dataset_config

def as_homogeneous_transform(transform: np.ndarray) -> np.ndarray:
    """Convert a 3x4 or 4x4 transform to homogeneous 4x4 form."""
    if transform.shape == (4, 4):
        return transform.astype(np.float32, copy=False)
    if transform.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3, :] = transform
        return homogeneous
    raise ValueError(f"Unsupported transform shape: {transform.shape}")

def normalize_camera_metadata(
    camera_poses: Optional[np.ndarray],
    camera_intrinsics: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Normalize saved camera metadata to per-view arrays."""
    if camera_poses is None or camera_intrinsics is None:
        return None, None

    poses = np.asarray(camera_poses)
    intrinsics = np.asarray(camera_intrinsics)

    if poses.ndim == 2:
        poses = poses[None]
    if intrinsics.ndim == 2:
        intrinsics = intrinsics[None]

    if poses.ndim != 3 or poses.shape[-2:] not in {(4, 4), (3, 4)}:
        print(f"[WARNING] Unsupported camera pose shape: {poses.shape}")
        return None, None
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        print(f"[WARNING] Unsupported intrinsics shape: {intrinsics.shape}")
        return None, None

    poses = np.stack([as_homogeneous_transform(pose) for pose in poses], axis=0)
    intrinsics = intrinsics.astype(np.float32, copy=False)

    if len(poses) != len(intrinsics):
        if len(intrinsics) == 1:
            intrinsics = np.repeat(intrinsics, len(poses), axis=0)
        else:
            num_views = min(len(poses), len(intrinsics))
            print(
                f"[WARNING] Camera metadata length mismatch: {len(poses)} poses vs {len(intrinsics)} intrinsics; "
                f"using first {num_views} views"
            )
            poses = poses[:num_views]
            intrinsics = intrinsics[:num_views]

    return poses, intrinsics

def get_camera_frustum_colors(num_cameras: int) -> List[Tuple[float, float, float]]:
    """Return distinct categorical colors for camera frustums."""
    if num_cameras <= 0:
        return []
    return [CAMERA_FRUSTUM_COLORS[i % len(CAMERA_FRUSTUM_COLORS)] for i in range(num_cameras)]

def draw_camera_frustums(
    figure,
    camera_poses: Optional[np.ndarray],
    camera_intrinsics: Optional[np.ndarray],
    vox_origin: np.ndarray,
    colors: Optional[List[Tuple[float, float, float]]] = None,
    d: float = 2.5,
    t_cam_to_voxel: Optional[np.ndarray] = None,
) -> None:
    """Draw saved camera frustums in voxel-grid coordinates."""
    poses, intrinsics = normalize_camera_metadata(camera_poses, camera_intrinsics)
    if poses is None or intrinsics is None:
        return

    if t_cam_to_voxel is None:
        t_cam_to_voxel = T_CAM_TO_VOXEL

    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]

    if colors is None:
        colors = get_camera_frustum_colors(len(poses))

    for camera_idx, (c2w, intrinsic) in enumerate(zip(poses, intrinsics)):
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        if fx <= 0.0 or fy <= 0.0:
            print(f"[WARNING] Skipping camera with non-positive focal lengths fx={fx}, fy={fy}")
            continue

        img_width = max(float(intrinsic[0, 2]) * 2.0, 1.0)
        img_height = max(float(intrinsic[1, 2]) * 2.0, 1.0)
        camera_to_voxel = t_cam_to_voxel @ c2w

        x_extent = d * img_width / (2.0 * fx)
        y_extent = d * img_height / (2.0 * fy)
        tri_points = np.array(
            [
                [0.0, 0.0, 0.0, 1.0],
                [x_extent, y_extent, d, 1.0],
                [-x_extent, y_extent, d, 1.0],
                [-x_extent, -y_extent, d, 1.0],
                [x_extent, -y_extent, d, 1.0],
            ],
            dtype=np.float32,
        )
        tri_points = (camera_to_voxel @ tri_points.T).T

        coords = tri_points[:, :3].copy()
        coords[:, 0] -= vox_origin[0]
        coords[:, 1] -= vox_origin[1]
        coords[:, 2] -= vox_origin[2]

        frustum_mesh = mlab.triangular_mesh(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            triangles,
            representation="wireframe",
            color=colors[camera_idx % len(colors)],
            line_width=3.0,
            figure=figure,
        )
        frustum_mesh.actor.property.lighting = False

def normalize_input_images(input_images: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize saved input images to uint8 NHWC RGB arrays."""
    if input_images is None:
        return None

    images = np.asarray(input_images)
    if images.ndim == 3:
        if images.shape[-1] == 3:
            images = images[None]
        elif images.shape[0] == 3:
            images = np.transpose(images[None], (0, 2, 3, 1))
        else:
            print(f"[WARNING] Unsupported input image shape: {images.shape}")
            return None
    elif images.ndim == 4:
        if images.shape[-1] == 3:
            pass
        elif images.shape[1] == 3:
            images = np.transpose(images, (0, 2, 3, 1))
        else:
            print(f"[WARNING] Unsupported input image shape: {images.shape}")
            return None
    else:
        print(f"[WARNING] Unsupported input image rank: {images.shape}")
        return None

    if images.dtype == np.uint8:
        return images

    if np.issubdtype(images.dtype, np.floating):
        image_min = float(images.min())
        image_max = float(images.max())
        if image_min >= -1.05 and image_max <= 1.05:
            images = ((np.clip(images, -1.0, 1.0) + 1.0) * 127.5).round().astype(np.uint8)
        elif image_min >= 0.0 and image_max <= 1.05:
            images = (np.clip(images, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        else:
            images = np.clip(images, 0.0, 255.0).round().astype(np.uint8)
    else:
        images = np.clip(images, 0, 255).astype(np.uint8)

    return images

def resize_image_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    """Resize an RGB image to the requested width while preserving aspect ratio."""
    if image.shape[1] == target_width:
        return image

    target_height = max(1, int(round(image.shape[0] * target_width / image.shape[1])))
    pil_image = Image.fromarray(image)
    if hasattr(Image, "Resampling"):
        resized = pil_image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    else:
        resized = pil_image.resize((target_width, target_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)

def resize_image_to_fit(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize an RGB image to fit inside a target box while preserving aspect ratio."""
    if max_width <= 0 or max_height <= 0:
        raise ValueError(f"Invalid resize target {(max_width, max_height)}")

    scale = min(max_width / image.shape[1], max_height / image.shape[0])
    target_width = max(1, int(round(image.shape[1] * scale)))
    target_height = max(1, int(round(image.shape[0] * scale)))
    if target_width == image.shape[1] and target_height == image.shape[0]:
        return image

    pil_image = Image.fromarray(image)
    if hasattr(Image, "Resampling"):
        resized = pil_image.resize((target_width, target_height), Image.Resampling.BILINEAR)
    else:
        resized = pil_image.resize((target_width, target_height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)

def whiten_near_white_pixels(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Convert near-white pixels to pure white."""
    whitened = image.copy()
    near_white_mask = np.all(whitened >= threshold, axis=-1)
    whitened[near_white_mask] = 255
    return whitened

def crop_white_margin(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """Crop away near-white margins around an RGB image."""
    content_mask = np.any(image < threshold, axis=-1)
    if not np.any(content_mask):
        return image

    rows = np.where(np.any(content_mask, axis=1))[0]
    cols = np.where(np.any(content_mask, axis=0))[0]
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

def load_nuscenes_center_car_image(car_image_path: Path = CAR_IMAGE_PATH) -> Optional[np.ndarray]:
    """Load and clean the center car illustration used for NuScenes input layouts."""
    if not car_image_path.exists():
        print(f"[WARNING] Missing center car image: {car_image_path}")
        return None

    car_image = np.asarray(Image.open(car_image_path).convert("RGB"), dtype=np.uint8)
    car_image = whiten_near_white_pixels(car_image)
    car_image = crop_white_margin(car_image)
    return car_image

def add_colored_border(
    image: np.ndarray,
    color: Tuple[float, float, float],
    border_size: int = 10,
) -> np.ndarray:
    """Wrap an RGB image with a solid border color."""
    color_uint8 = (np.clip(np.asarray(color), 0.0, 1.0) * 255.0).round().astype(np.uint8)
    bordered = np.empty(
        (image.shape[0] + (2 * border_size), image.shape[1] + (2 * border_size), 3),
        dtype=np.uint8,
    )
    bordered[...] = color_uint8
    bordered[border_size:-border_size, border_size:-border_size] = image
    return bordered

def paste_image_centered(
    canvas: np.ndarray,
    image: np.ndarray,
    top: int,
    left: int,
    region_height: int,
    region_width: int,
) -> None:
    """Paste an image centered inside a rectangular region of a canvas."""
    paste_top = top + max((region_height - image.shape[0]) // 2, 0)
    paste_left = left + max((region_width - image.shape[1]) // 2, 0)
    canvas[paste_top : paste_top + image.shape[0], paste_left : paste_left + image.shape[1]] = image

def build_nuscenes_surround_input_canvas(
    bordered_images: List[np.ndarray],
    car_image_path: Path = CAR_IMAGE_PATH,
    gap: int = 24,
    outer_margin: int = 24,
) -> Optional[np.ndarray]:
    """Build a NuScenes surround-view canvas on a 4x3 implicit grid."""
    if len(bordered_images) != 6:
        return None

    car_image = load_nuscenes_center_car_image(car_image_path)
    if car_image is None:
        return None

    image_by_slot = {
        NUSCENES_CAMERA_SLOT_BY_INDEX[idx]: bordered_images[idx]
        for idx in range(len(bordered_images))
    }
    tile_width = max(image.shape[1] for image in bordered_images)
    tile_height = max(image.shape[0] for image in bordered_images)

    canvas_height = (2 * outer_margin) + (4 * tile_height) + (3 * gap)
    canvas_width = (2 * outer_margin) + (3 * tile_width) + (2 * gap)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    x_positions = [
        outer_margin,
        outer_margin + tile_width + gap,
        outer_margin + (2 * (tile_width + gap)),
    ]
    y_positions = [
        outer_margin,
        outer_margin + tile_height + gap,
        outer_margin + (2 * (tile_height + gap)),
        outer_margin + (3 * (tile_height + gap)),
    ]

    slot_layout = {
        "front": (y_positions[0], x_positions[1], tile_height, tile_width),
        "left": (y_positions[1], x_positions[0], tile_height, tile_width),
        "right": (y_positions[1], x_positions[2], tile_height, tile_width),
        "back_left": (y_positions[2], x_positions[0], tile_height, tile_width),
        "back_right": (y_positions[2], x_positions[2], tile_height, tile_width),
        "back": (y_positions[3], x_positions[1], tile_height, tile_width),
    }

    for slot_name, (top, left, region_height, region_width) in slot_layout.items():
        paste_image_centered(
            canvas=canvas,
            image=image_by_slot[slot_name],
            top=top,
            left=left,
            region_height=region_height,
            region_width=region_width,
        )

    center_car = resize_image_to_fit(
        car_image,
        max_width=max(1, int(round(tile_width * 0.88))),
        max_height=max(1, int(round(((2 * tile_height) + gap) * 0.9))),
    )
    paste_image_centered(
        canvas=canvas,
        image=center_car,
        top=y_positions[1],
        left=x_positions[1],
        region_height=(2 * tile_height) + gap,
        region_width=tile_width,
    )
    return canvas

def build_surround_input_canvas(
    bordered_images: List[np.ndarray],
    camera_slot_by_index: Dict[int, str],
    car_image_path: Path = CAR_IMAGE_PATH,
    gap: int = 24,
    outer_margin: int = 24,
) -> Optional[np.ndarray]:
    """Build a generic surround-view canvas with a centered car image."""
    if len(bordered_images) != len(camera_slot_by_index):
        return None

    car_image = load_nuscenes_center_car_image(car_image_path)
    if car_image is None:
        return None

    image_by_slot: Dict[str, np.ndarray] = {}
    for idx, image in enumerate(bordered_images):
        slot_name = camera_slot_by_index.get(idx)
        if slot_name is None:
            continue
        image_by_slot[slot_name] = image

    if len(image_by_slot) == 0:
        return None

    unknown_slots = sorted(slot for slot in image_by_slot if slot not in SURROUND_CAMERA_GRID_SLOTS)
    if len(unknown_slots) > 0:
        print(f"[WARNING] Unsupported surround-camera slots: {unknown_slots}")
        return None

    tile_width = max(image.shape[1] for image in bordered_images)
    tile_height = max(image.shape[0] for image in bordered_images)

    used_rows = [SURROUND_CAMERA_GRID_SLOTS[slot][0] for slot in image_by_slot]
    num_rows = max(max(used_rows), 1) + 1
    num_cols = 3

    canvas_height = (2 * outer_margin) + (num_rows * tile_height) + ((num_rows - 1) * gap)
    canvas_width = (2 * outer_margin) + (num_cols * tile_width) + ((num_cols - 1) * gap)
    canvas = np.full((canvas_height, canvas_width, 3), 255, dtype=np.uint8)

    x_positions = [
        outer_margin,
        outer_margin + tile_width + gap,
        outer_margin + (2 * (tile_width + gap)),
    ]
    y_positions = [outer_margin + (row_idx * (tile_height + gap)) for row_idx in range(num_rows)]

    for slot_name, image in image_by_slot.items():
        row_idx, col_idx = SURROUND_CAMERA_GRID_SLOTS[slot_name]
        paste_image_centered(
            canvas=canvas,
            image=image,
            top=y_positions[row_idx],
            left=x_positions[col_idx],
            region_height=tile_height,
            region_width=tile_width,
        )

    center_car = resize_image_to_fit(
        car_image,
        max_width=max(1, int(round(tile_width * 0.88))),
        max_height=max(1, int(round(tile_height * 0.88))),
    )
    paste_image_centered(
        canvas=canvas,
        image=center_car,
        top=y_positions[1],
        left=x_positions[1],
        region_height=tile_height,
        region_width=tile_width,
    )
    return canvas

def save_stacked_input_images(
    input_images: Optional[np.ndarray],
    out_filepath: Path,
    dataset_name: Optional[str] = None,
    border_colors: Optional[List[Tuple[float, float, float]]] = None,
    border_size: int = 10,
) -> bool:
    """Save input images with dataset-specific layouts and matching borders."""
    normalized_images = normalize_input_images(input_images)
    if normalized_images is None:
        return False
    if len(normalized_images) == 0:
        print(f"[WARNING] No input images available for {out_filepath.name}")
        return False

    if border_colors is None or len(border_colors) == 0:
        border_colors = get_camera_frustum_colors(len(normalized_images))

    target_width = max(int(image.shape[1]) for image in normalized_images)
    stacked_images: List[np.ndarray] = []
    for image_idx, image in enumerate(normalized_images):
        resized_image = resize_image_to_width(image, target_width=target_width)
        border_color = border_colors[image_idx % len(border_colors)]
        stacked_images.append(add_colored_border(resized_image, color=border_color, border_size=border_size))

    canvas: Optional[np.ndarray] = None
    if dataset_name == "nuscenes" and len(stacked_images) == 6:
        canvas = build_nuscenes_surround_input_canvas(stacked_images)
        if canvas is None:
            first_row = np.concatenate(stacked_images[:3], axis=1)
            second_row = np.concatenate(stacked_images[3:], axis=1)
            canvas = np.concatenate([first_row, second_row], axis=0)
    elif dataset_name in SURROUND_CAMERA_SLOT_BY_DATASET:
        camera_slot_by_index = SURROUND_CAMERA_SLOT_BY_DATASET[dataset_name]
        canvas = build_surround_input_canvas(stacked_images, camera_slot_by_index=camera_slot_by_index)

    if canvas is None:
        canvas = np.concatenate(stacked_images, axis=0)

    Image.fromarray(canvas).save(out_filepath)
    return True

def get_grid_coords(
    dims: Tuple[int, int, int],
    resolution: float,
) -> Tuple[np.ndarray, None, None, None]:
    """Return voxel center coordinates for a dense grid."""
    g_xx = np.arange(0, dims[0] + 1)
    g_yy = np.arange(0, dims[1] + 1)
    g_zz = np.arange(0, dims[2] + 1)

    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(float)
    coords_grid = (coords_grid * resolution) + resolution / 2

    swapped = coords_grid.copy()
    swapped[:, 0] = coords_grid[:, 1]
    swapped[:, 1] = coords_grid[:, 0]
    coords_grid = swapped.copy()
    return coords_grid, None, None, None

def darken_semantic_colors(colors: np.ndarray, factor: float = GT_NONOVERLAP_DARKEN_FACTOR) -> np.ndarray:
    """Return a darker copy of the semantic palette."""
    darkened_colors = colors.copy()
    darkened_colors[:, :3] = np.clip(
        np.round(darkened_colors[:, :3].astype(np.float32) * factor),
        0,
        255,
    ).astype(np.uint8)
    return darkened_colors

def build_lut(
    colors: np.ndarray,
    max_label: int,
    include_darker_colors: bool = False,
) -> np.ndarray:
    """Build a semantic color table from the active dataset palette."""
    lut_colors = colors.copy()
    if include_darker_colors:
        lut_colors = np.concatenate([lut_colors, darken_semantic_colors(colors)], axis=0)

    if max_label < len(lut_colors):
        return lut_colors

    lut = np.zeros((max_label + 1, 4), dtype=np.uint8)
    lut[: len(lut_colors)] = lut_colors
    lut[len(lut_colors) :] = lut_colors[-1]
    return lut

def position_scene_view(scene, view=3):
    if view not in {2, 3}:
        raise ValueError(f"Unsupported view {view}; only view=2 and view=3 are available")

    scene.x_minus_view()
    scene.x_minus_view()
    if engine is not None:
        active_scene = engine.scenes[0].scene if len(engine.scenes) > 0 else scene
    else:
        active_scene = scene

    if view == 2:
        active_scene.camera.position = [-105.44804511130367, -29.61938241182379, 129.95200398593926]
        active_scene.camera.focal_point = [34.014877645151564, 31.356224643180944, 8.866392378561098]
        active_scene.camera.clipping_range = [138.15970362215074, 316.08730809606186]
    else:
        active_scene.camera.position = [-63.77078599869435, -13.39609264351163, 81.84096163277799]
        active_scene.camera.focal_point = [29.140641646621093, 27.2267823776251, 1.172919170310438]
        active_scene.camera.clipping_range = [92.04254719189133, 210.5786290028288]
    active_scene.camera.view_angle = 30.0
    active_scene.camera.view_up = [0.4874235998530473, 0.4129442265555842, 0.769347320825066]
    active_scene.camera.compute_view_plane_normal()
    active_scene.render()

def draw_semantic_voxels(
    figure,
    coords: np.ndarray,
    labels: np.ndarray,
    lut: np.ndarray,
    scale_factor: float,
    opacity: float,
) -> None:
    """Draw a semantic voxel subset with a fixed opacity."""
    scalar_labels = labels.astype(np.float32, copy=False)
    plt_plot = mlab.points3d(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        scalar_labels,
        colormap="viridis",
        scale_factor=scale_factor,
        mode="cube",
        opacity=opacity,
        vmin=0,
        vmax=max(len(lut) - 1, 1),
        figure=figure,
    )
    plt_plot.glyph.scale_mode = "scale_by_vector"
    plt_plot.module_manager.scalar_lut_manager.lut.table = lut
