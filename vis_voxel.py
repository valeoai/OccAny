import argparse
import os
import pickle
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast


import numpy as np
from mayavi import mlab
mlab.options.offscreen = True

from tqdm import tqdm
from PIL import Image

from occany.utils.vis_util import (
    DATASET_CONFIGS,
    IGNORE_LABEL,
    OCC3D_COLORS,
    build_lut,
    draw_camera_frustums,
    draw_semantic_voxels,
    get_camera_frustum_colors,
    get_grid_coords,
    infer_voxel_origin,
    infer_voxel_size,
    position_scene_view,
    resolve_dataset_config,
    save_stacked_input_images,
)

try:
    engine = mayavi.engine
except NameError:
    from mayavi.api import Engine

    engine = Engine()
    engine.start()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render semantic voxel predictions from inference scene pickle files"
    )
    parser.add_argument(
        "--input_path",
        "--input_root",
        dest="input_root",
        type=str,
        default="./demo_data/output",
        help=(
            "Path to a demo output root containing scene folders, a single scene folder, "
            "or a voxel_predictions.pkl file"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./demo_data/output_vis",
        help="Directory to save rendered images as <scene_id>_ssc.png",
    )
    parser.add_argument(
        "--pkl_name",
        type=str,
        default="voxel_predictions.pkl",
        help="Preferred pickle filename to search for recursively",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing rendered images",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=None,
        help="Override voxel size in meters; otherwise infer from grid shape",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_CONFIGS.keys()),
        default="nuscenes",
        help="Dataset name to determine visualization layout (car-centered for surround datasets)",
    )
    parser.add_argument(
        "--empty_class",
        type=int,
        default=17,
        help="Override empty semantic class; otherwise infer from voxel values",
    )
    parser.add_argument(
        "--other_class",
        type=int,
        default=0,
        help="Additional semantic class to treat as empty when rendering (default: 0)",
    )
    parser.add_argument(
        "--prediction_key",
        type=str,
        default="render_recon_gen_recon4.0_gen4.0",
        help="Voxel prediction key to render from each pickle",
    )
    parser.add_argument(
        "--demo_folders",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Optional subfolders under input_root to visualise. "
            "Leave unset to use input_root directly."
        ),
    )
    parser.add_argument(
        "--color_by_height",
        action="store_true",
        help="Color occupied voxels by height using a spectral colormap (low=blue, high=red)",
    )
    parser.add_argument(
        "--save_input_images",
        action="store_true",
        help="Also save stacked input images next to the rendered semantic scene capture",
    )
    return parser.parse_args()


def build_height_spectral_lut(num_colors: int = 256) -> np.ndarray:
    """Build a spectral LUT with low=blue and high=red."""
    key_colors = np.array(
        [
            [0, 0, 255],
            [0, 255, 255],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ],
        dtype=np.float32,
    )
    key_positions = np.linspace(0.0, 1.0, key_colors.shape[0], dtype=np.float32)
    sample_positions = np.linspace(0.0, 1.0, num_colors, dtype=np.float32)

    lut_rgb = np.empty((num_colors, 3), dtype=np.uint8)
    for channel_idx in range(3):
        lut_rgb[:, channel_idx] = np.interp(
            sample_positions,
            key_positions,
            key_colors[:, channel_idx],
        ).round().astype(np.uint8)

    alpha = np.full((num_colors, 1), 255, dtype=np.uint8)
    return np.concatenate([lut_rgb, alpha], axis=1)


def render_voxel_grid(
    voxels: np.ndarray,
    out_filepath: Path,
    figure,
    colors: np.ndarray,
    empty_class: Optional[int] = None,
    voxel_size: float = 0.2,
    voxel_origin: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
    other_class: Optional[int] = None,
    estimated_camera_poses: Optional[np.ndarray] = None,
    estimated_input_intrinsics: Optional[np.ndarray] = None,
    t_cam_to_voxel: Optional[np.ndarray] = None,
    color_by_height: bool = False,
) -> None:
    """Render a voxel grid to disk."""
    resolved_voxel_size = voxel_size
    resolved_voxel_origin = voxel_origin

    frustum_colors = None
    if estimated_camera_poses is not None:
        estimated_camera_poses_array = np.asarray(estimated_camera_poses)
        num_cameras = 1 if estimated_camera_poses_array.ndim == 2 else int(estimated_camera_poses_array.shape[0])
        frustum_colors = get_camera_frustum_colors(num_cameras)

    occupied_mask = (voxels != empty_class) & (voxels != IGNORE_LABEL) & (voxels >= 0)
    occupied_mask &= voxels != other_class
    flat_mask = occupied_mask.reshape(-1)

    mlab.clf(figure)

    if np.any(flat_mask):
        grid_coords, _, _, _ = get_grid_coords(tuple(voxels.shape), resolved_voxel_size)
        occupied_coords = grid_coords[flat_mask]

        if color_by_height:
            height_values = occupied_coords[:, 2].astype(np.float32, copy=False)
            height_min = 0.1
            height_max = 4.4
            if np.isclose(height_min, height_max):
                height_max = height_min + 1e-6

            plt_plot = mlab.points3d(
                occupied_coords[:, 0],
                occupied_coords[:, 1],
                occupied_coords[:, 2],
                height_values,
                colormap="spectral",
                scale_factor=resolved_voxel_size * 0.95,
                mode="cube",
                opacity=1.0,
                vmin=height_min,
                vmax=height_max,
                figure=figure,
            )
            plt_plot.glyph.scale_mode = "scale_by_vector"
            plt_plot.module_manager.scalar_lut_manager.lut.table = build_height_spectral_lut()
        else:
            semantic_labels = voxels.reshape(-1)[flat_mask]
            lut = build_lut(
                colors=colors,
                max_label=int(semantic_labels.max()),
                include_darker_colors=bool(np.any(semantic_labels >= len(colors))),
            )

            draw_semantic_voxels(
                figure=figure,
                coords=occupied_coords,
                labels=semantic_labels,
                lut=lut,
                scale_factor=resolved_voxel_size * 0.95,
                opacity=1.0,
            )
    else:
        print(f"[WARNING] No occupied voxels found for {out_filepath.name}")

    draw_camera_frustums(
        figure=figure,
        camera_poses=estimated_camera_poses,
        camera_intrinsics=estimated_input_intrinsics,
        vox_origin=resolved_voxel_origin,
        colors=frustum_colors,
        t_cam_to_voxel=t_cam_to_voxel,
    )

    position_scene_view(figure.scene, view=3)
    mlab.savefig(str(out_filepath), figure=figure)


def iter_pickle_files(input_root: Path, pkl_name: str) -> List[Path]:
    """Yield pickle files from a direct file, a scene folder, or recursively under a root."""
    if input_root.is_file():
        if input_root.suffix.lower() != ".pkl":
            raise ValueError(f"Expected a .pkl file, got: {input_root}")
        return [input_root]

    preferred_pickle = input_root / pkl_name
    if preferred_pickle.is_file():
        return [preferred_pickle]

    direct_pickles = sorted(input_root.glob("*.pkl"))
    if direct_pickles:
        return direct_pickles

    recursive_matches = sorted(input_root.rglob(pkl_name))
    if recursive_matches:
        return recursive_matches

    return sorted(input_root.rglob("*.pkl"))


def parse_sample_id(sample_id: str) -> Tuple[str, str, str]:
    """Parse supported sample identifier formats."""
    scene_match = re.match(r"^(?P<scene>.+)_(?P<frame>\d+)_(?P<camera>.+)$", sample_id)
    if scene_match is not None:
        return scene_match.group("scene"), scene_match.group("frame"), scene_match.group("camera")

    seq_match = re.match(r"^(?P<sequence>.+)_(?P<frame>\d+)$", sample_id)
    if seq_match is not None:
        return seq_match.group("sequence"), seq_match.group("frame"), ""

    frame_match = re.match(r"^(?P<frame>\d+)$", sample_id)
    if frame_match is not None:
        return "", frame_match.group("frame"), ""

    raise ValueError(f"Sample id '{sample_id}' does not match supported formats")


def sanitize_path_component(value: str) -> str:
    """Make a string safe for directory creation."""
    sanitized = value.replace(os.sep, "_")
    if os.altsep is not None:
        sanitized = sanitized.replace(os.altsep, "_")
    return sanitized


def infer_scene_id(pickle_path: Path) -> str:
    """Infer a stable scene identifier from a scene pickle path."""
    candidate_dirs = [pickle_path.parent]
    candidate_dirs.extend(list(pickle_path.parents)[1:3])
    for candidate_dir in candidate_dirs:
        if any(candidate_dir.glob("pts3d_*.npy")):
            return sanitize_path_component(candidate_dir.name)
    return sanitize_path_component(pickle_path.parent.name)


def build_output_stem(scene_id: str, pickle_path: Path, scene_id_counts: Dict[str, int]) -> str:
    """Build an output stem that avoids collisions when multiple pickles map to one scene id."""
    if scene_id_counts.get(scene_id, 0) <= 1:
        return scene_id
    return f"{scene_id}_{sanitize_path_component(pickle_path.stem)}"


def resolve_prediction_grid(
    voxel_predictions: Dict[str, np.ndarray],
    requested_key: str,
    pickle_path: Path,
) -> Tuple[str, np.ndarray]:
    """Pick the requested render grid or the best available render* voxel fallback."""
    requested_value = voxel_predictions.get(requested_key)
    if (
        requested_key.startswith("render")
        and isinstance(requested_value, np.ndarray)
        and requested_value.ndim == 3
    ):
        return requested_key, requested_value

    candidate_keys: List[Tuple[int, str]] = []
    for key, value in voxel_predictions.items():
        if not key.startswith("render"):
            continue
        if not isinstance(value, np.ndarray) or value.ndim != 3:
            continue
        priority = 2
        if key.startswith("render_recon_gen"):
            priority = 0
        elif key.startswith("render_th"):
            priority = 1
        candidate_keys.append((priority, key))

    if not candidate_keys:
        raise ValueError(f"No render voxel grids found in {pickle_path}")

    candidate_keys.sort(key=lambda item: (item[0], item[1]))
    resolved_key = candidate_keys[0][1]
    if requested_value is None:
        print(
            f"[INFO] Missing voxel key '{requested_key}' in {pickle_path.name}; "
            f"using '{resolved_key}' instead"
        )
    else:
        print(
            f"[INFO] Requested voxel key '{requested_key}' is not a render voxel grid in {pickle_path.name}; "
            f"using '{resolved_key}' instead"
        )
    return resolved_key, cast(np.ndarray, voxel_predictions[resolved_key])


def main(args: argparse.Namespace) -> None:
    input_root = Path(os.path.expanduser(args.input_root)).resolve()
    output_root = Path(args.output_root).resolve()
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")

    if not input_root.exists():
        raise ValueError(f"Input path does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    resolve_dataset_config(args.dataset)

    if args.demo_folders:
        pickle_paths: List[Path] = []
        for folder in args.demo_folders:
            folder_path = input_root / folder
            if not folder_path.exists():
                print(f"[WARNING] Demo folder does not exist: {folder_path}")
                continue
            pickle_paths.extend(iter_pickle_files(input_root=folder_path, pkl_name=args.pkl_name))
        print(f"[INFO] Found {len(pickle_paths)} pickle file(s) across {len(args.demo_folders)} folder(s)")
    else:
        pickle_paths = list(iter_pickle_files(input_root=input_root, pkl_name=args.pkl_name))
        print(f"[INFO] Found {len(pickle_paths)} pickle file(s) under {input_root}")

    if len(pickle_paths) == 0:
        raise ValueError(f"No '.pkl' files found under {input_root}")

    scene_id_counts = Counter(infer_scene_id(path) for path in pickle_paths)
    figure = mlab.figure(size=(1600, 1200), bgcolor=(1, 1, 1), engine=engine)

    saved_count = 0
    pickle_progress = tqdm(pickle_paths, desc="Rendering pickle files", unit="file")
    for pickle_path in pickle_progress:
        scene_id = infer_scene_id(pickle_path)
        output_stem = build_output_stem(scene_id, pickle_path, scene_id_counts)
        out_filepath = output_root / f"{output_stem}_ssc.png"
        input_out_filepath = output_root / f"{output_stem}_inputs.png"

        if out_filepath.exists() and not args.overwrite:
            continue

        with open(pickle_path, "rb") as f:
            voxel_predictions: Dict[str, np.ndarray] = pickle.load(f)

        if not isinstance(voxel_predictions, dict):
            print(f"[WARNING] Unexpected pickle payload in {pickle_path}; expected a dictionary")
            continue

        try:
            resolved_key, voxel_grid = resolve_prediction_grid(
                voxel_predictions=voxel_predictions,
                requested_key=args.prediction_key,
                pickle_path=pickle_path,
            )
        except ValueError as exc:
            print(f"[WARNING] {exc}")
            continue

        pickle_progress.set_postfix_str(f"{scene_id}:{resolved_key}")

        estimated_camera_poses = voxel_predictions.get("estimated_input_camera_poses")
        estimated_input_intrinsics = voxel_predictions.get("estimated_input_intrinsics")
        estimated_input_images = voxel_predictions.get("estimated_input_images")

        voxel_size = voxel_predictions.get("voxel_size")
        if voxel_size is None:
            voxel_size = args.voxel_size if args.voxel_size is not None else infer_voxel_size(voxel_grid)

        voxel_origin = voxel_predictions.get("voxel_origin")
        if voxel_origin is None:
            voxel_origin = infer_voxel_origin(voxel_grid)

        render_voxel_grid(
            colors=OCC3D_COLORS,
            voxels=voxel_grid,
            out_filepath=out_filepath,
            figure=figure,
            empty_class=args.empty_class,
            voxel_size=float(voxel_size),
            voxel_origin=np.asarray(voxel_origin, dtype=np.float32),
            other_class=args.other_class,
            estimated_camera_poses=estimated_camera_poses,
            estimated_input_intrinsics=estimated_input_intrinsics,
            t_cam_to_voxel=None,
            color_by_height=args.color_by_height,
        )

        if args.save_input_images and estimated_input_images is not None:
            save_stacked_input_images(
                input_images=estimated_input_images,
                out_filepath=input_out_filepath,
                dataset_name=args.dataset,
            )

        saved_count += 1

    mlab.clf(figure)
    print(f"[INFO] Saved {saved_count} renders to {output_root}")


if __name__ == "__main__":
    main(parse_args())
