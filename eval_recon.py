import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from occany.utils.recon_eval_helper import evaluate_3d_reconstruction

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

torch.backends.cuda.matmul.allow_tf32 = True

METRIC_KEYS = ("acc", "comp", "overall", "precision", "recall", "fscore")
PERCENT_METRIC_KEYS = {"precision", "recall", "fscore"}
MAX_POINT_CLOUD_DEPTH_METERS = 50.0
DEFAULT_DA3_MODEL_NAME = "depth-anything/DA3-GIANT-1.1"
DEFAULT_DA3_METRIC_MODEL_NAME = "depth-anything/DA3METRIC-LARGE"
IMAGE_SIZE = 512


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate OccAny reconstruction with sparse lidar-derived GT points.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", "occany_da3"),
        choices=["occany_da3", "da3"],
        help="Model family to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        "-ds",
        type=str,
        default=os.environ.get("DATASET", "nuscenes"),
        choices=["kitti", "nuscenes"],
        help="Dataset to evaluate.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default=os.environ.get("SETTING", "5frames"),
        choices=["5frames", "surround"],
        help="Evaluation setting. Supported: KITTI 5frames, nuScenes surround.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=os.environ.get("SPLIT", "val"),
        choices=["val"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "cuda"),
        help="PyTorch device.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory where metric results are written.",
    )
    parser.add_argument(
        "--occany_recon_ckpt",
        dest="occany_recon_ckpt",
        type=str,
        default=os.environ.get("OCCANY_RECON_CKPT") or os.environ.get("DA3_RECON_CKPT") or "",
        help=(
            "Optional path to an OccAny or DA3 reconstruction checkpoint used with --model occany_da3. "
            "If unset, the evaluator falls back to the plain DA3 backbone from --da3_model_name."
        ),
    )
    parser.add_argument(
        "--da3_model_name",
        type=str,
        default=os.environ.get("DA3_MODEL_NAME", DEFAULT_DA3_MODEL_NAME),
        help="Hugging Face model name/path used when --occany_recon_ckpt is not provided.",
    )
    parser.add_argument(
        "--da3_metric_model_name",
        type=str,
        default=os.environ.get("DA3_METRIC_MODEL_NAME", DEFAULT_DA3_METRIC_MODEL_NAME),
        help="Hugging Face model name/path for the DA3 metric model (used with --model da3).",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=os.environ.get("EXP_NAME", ""),
        help="Optional experiment prefix for the output subdirectory.",
    )
    parser.add_argument(
        "--frame_interval",
        type=int,
        default=int(os.environ.get("FRAME_INTERVAL", 5)),
        help="Relative timestep spacing fed to the model for selected views.",
    )
    parser.add_argument(
        "--kitti_root",
        type=str,
        default=os.environ.get("PROJECT") + "/data/kitti",
        help="KITTI dataset root.",
    )
    parser.add_argument(
        "--nuscenes_root",
        type=str,
        default=os.environ.get("PROJECT") + "/data/nuscenes",
        help="nuScenes dataset root.",
    )
    parser.add_argument(
        "--metric_threshold",
        type=float,
        default=float(os.environ.get("METRIC_THRESHOLD", 0.5)),
        help="Distance threshold in meters for precision / recall / f-score.",
    )
    parser.add_argument(
        "--save_pointcloud_pairs",
        type=int,
        default=int(os.environ.get("SAVE_POINTCLOUD_PAIRS", 0)),
        help=(
            "Number of predicted/GT point-cloud pairs to save as TXT for CloudCompare. "
            "Use 0 to disable export."
        ),
    )
    parser.add_argument(
        "--save_pointcloud_stride",
        type=int,
        default=int(os.environ.get("SAVE_POINTCLOUD_STRIDE", 10)),
        help="Save one point-cloud pair every N samples when --save_pointcloud_pairs > 0.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", 4)),
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--pose_from_depth_ray",
        action="store_true",
        default=os.environ.get("POSE_FROM_DEPTH_RAY", "").strip().lower() in {"1", "true", "yes", "on"},
        help="OccAny+ only: estimate poses from predicted raymaps before building pointmaps.",
    )
    parser.add_argument(
        "--point_from_depth_and_pose",
        action="store_true",
        default=os.environ.get("POINT_FROM_DEPTH_AND_POSE", "").strip().lower() in {"1", "true", "yes", "on"},
        help="OccAny+ only: rebuild pointmaps from predicted depth + pose.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        default=os.environ.get("SILENT", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Reduce per-sample logging.",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset == "kitti" and args.setting != "5frames":
        raise ValueError("This simplified evaluator only supports KITTI with --setting 5frames.")
    if args.dataset == "nuscenes" and args.setting != "surround":
        raise ValueError("This simplified evaluator only supports nuScenes with --setting surround.")
    if args.save_pointcloud_pairs < 0:
        raise ValueError("--save_pointcloud_pairs must be >= 0.")
    if args.save_pointcloud_stride < 1:
        raise ValueError("--save_pointcloud_stride must be >= 1.")


def convert_depth_to_point_cloud(
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    c2w: torch.Tensor,
) -> torch.Tensor:
    b, t, h, w = depth.shape
    device = depth.device

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=depth.dtype),
        torch.arange(w, device=device, dtype=depth.dtype),
        indexing="ij",
    )
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1)
    pixel_coords = pixel_coords[None, None, :, :, :].expand(b, t, -1, -1, -1)

    fx = intrinsics[:, :, 0, 0][:, :, None, None]
    fy = intrinsics[:, :, 1, 1][:, :, None, None]
    cx = intrinsics[:, :, 0, 2][:, :, None, None]
    cy = intrinsics[:, :, 1, 2][:, :, None, None]

    x_cam = (pixel_coords[..., 0] - cx) / fx
    y_cam = (pixel_coords[..., 1] - cy) / fy
    points_cam = torch.stack([x_cam * depth, y_cam * depth, depth], dim=-1)

    rotation = c2w[:, :, :3, :3]
    translation = c2w[:, :, :3, 3]
    points_cam_flat = points_cam.reshape(b, t, h * w, 3)
    points_world = torch.matmul(rotation, points_cam_flat.transpose(-2, -1)).transpose(-2, -1)
    points_world = points_world + translation[:, :, None, :]
    return points_world.reshape(b, t, h, w, 3)


def get_output_resolution(model_name: str, dataset_name: str) -> Tuple[int, int]:
    if dataset_name == "kitti":
        return 518, 168
    if dataset_name == "nuscenes":
        return 518, 294
    raise ValueError(f"Unsupported output resolution setup for model={model_name}, dataset={dataset_name}")


def build_output_dir(args: argparse.Namespace) -> Path:
    parts = [args.exp_name, args.model, args.dataset, args.setting, f"img{IMAGE_SIZE}"]
    run_name = "_".join(part for part in parts if part)
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_occany_plus_model(
    device: str,
    output_resolution: Tuple[int, int],
    occany_recon_ckpt: Optional[str],
    da3_model_name: str,
) -> torch.nn.Module:
    from occany.model.model_da3 import DA3Wrapper
    from occany.utils.io_da3 import load_da3_model_from_checkpoint

    if occany_recon_ckpt:
        recon_weights = Path(occany_recon_ckpt)
        if not recon_weights.is_file():
            raise FileNotFoundError(f"OccAny reconstruction checkpoint not found: {recon_weights}")

        print(f"Loading OccAny 1B reconstruction checkpoint: {recon_weights}")

        da3_model_recon, _ = load_da3_model_from_checkpoint(
            weights_path=str(recon_weights),
            output_resolution=output_resolution,
            semantic_feat_src=None,
            semantic_family=None,
            device=device,
            is_gen_model=False,
        )
        if da3_model_recon is None:
            raise RuntimeError("Failed to load DA3 reconstruction model from checkpoint.")
        return da3_model_recon

    print(f"Using DA3 model: {da3_model_name}")
    da3_model_recon = DA3Wrapper.from_pretrained(
        da3_model_name,
        img_size=output_resolution[0],
    )
    da3_model_recon = da3_model_recon.to(device)
    da3_model_recon.eval()
    da3_model_recon.requires_grad_(False)
    return da3_model_recon


def load_da3_models(
    device: str,
    da3_model_name: str,
    da3_metric_model_name: str,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load a standalone DA3 backbone + a DA3 metric model for metric-scaled inference."""
    from depth_anything_3.api import DepthAnything3

    print(f"Loading DA3 model: {da3_model_name}")
    da3 = DepthAnything3.from_pretrained(da3_model_name)
    da3 = da3.to(device)
    da3.eval()
    da3.requires_grad_(False)

    print(f"Loading DA3 metric model: {da3_metric_model_name}")
    da3_metric = DepthAnything3.from_pretrained(da3_metric_model_name)
    da3_metric = da3_metric.to(device)
    da3_metric.eval()
    da3_metric.requires_grad_(False)

    return da3, da3_metric


def build_recon_views(
    data: Dict[str, torch.Tensor],
    recon_view_idx: List[int],
    device: str,
    frame_interval: int,
) -> Tuple[List[Dict[str, torch.Tensor]], torch.Tensor, torch.Tensor, torch.Tensor]:
    imgs = data["imgs"].to(device)
    gt_depths = data["gt_depths"].to(device)
    camera_poses = data["cam_poses_in_cam0"].to(device)
    intrinsics = data["cam_k_resized"].to(device)
    batch_size, _, _, height, width = imgs.shape

    recon_views: List[Dict[str, torch.Tensor]] = []
    for original_view_idx in recon_view_idx:
        recon_views.append(
            {
                "img": imgs[:, original_view_idx],
                "timestep": torch.full(
                    (batch_size,),
                    float(original_view_idx * frame_interval),
                    dtype=torch.float32,
                    device=device,
                ),
                "true_shape": torch.tensor([height, width], device=device).view(1, 2).expand(batch_size, 2),
                "camera_pose": camera_poses[:, original_view_idx],
                "gt_depth": gt_depths[:, original_view_idx],
                "is_raymap": False,
            }
        )

    recon_gt_depths = torch.stack([view["gt_depth"] for view in recon_views], dim=1)
    recon_camera_poses = torch.stack([view["camera_pose"] for view in recon_views], dim=1)
    if intrinsics.dim() == 4:
        recon_intrinsics = intrinsics[:, recon_view_idx]
    elif intrinsics.dim() == 3:
        recon_intrinsics = intrinsics.unsqueeze(1).expand(-1, len(recon_views), -1, -1)
    else:
        raise ValueError(f"Unsupported intrinsics rank: {intrinsics.dim()}")

    return recon_views, recon_gt_depths, recon_intrinsics, recon_camera_poses


def _run_da3_metric_scaling_inference(
    recon_views: List[Dict[str, torch.Tensor]],
    da3_model: torch.nn.Module,
    da3_metric_model: torch.nn.Module,
    device: str,
) -> Dict[str, torch.Tensor]:
    """Run standalone DA3 inference with metric scaling, mirroring da3_predictor._inference_batch."""
    from depth_anything_3.utils.alignment import (
        apply_metric_scaling,
        compute_alignment_mask,
        compute_sky_mask,
        least_squares_scale_scalar,
        sample_tensor_for_quantile,
        set_sky_regions_to_max_depth,
    )
    from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
    from occany.utils.helpers import convert_depth_to_point_cloud as da3_depth_to_pointcloud

    # Stack images from views: (B, T, C, H, W)
    images = torch.stack([v["img"] for v in recon_views], dim=1).to(device)
    _, _, _, height, width = images.size()

    export_feat_layers = list(da3_model.model.backbone.out_layers)

    # 1. Run DA3 backbone to extract features
    feats, aux_feats = da3_model.model.backbone(
        images,
        cam_token=None,
        export_feat_layers=export_feat_layers,
        ref_view_strategy="first",
    )

    # 2. Run DA3 metric model to get metric depth
    metric_output = da3_metric_model(images, export_feat_layers=[])

    # 3. Run depth head
    depth_head = da3_model.model.head
    output = depth_head(feats, height, width, patch_start_idx=0)

    depth = output["depth"]
    depth_conf = output["depth_conf"]
    ray = output["ray"]
    ray_conf = output["ray_conf"]

    # 4. Estimate pose from depth ray
    device_type = images.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
            ray.float(),
            ray_conf.float().clone(),
            ray.shape[-3],
            ray.shape[-2],
        )
        c2w = pred_extrinsic[:, :, :3, :]
        intrinsics = (
            torch.eye(3, dtype=c2w.dtype, device=c2w.device)[None, None]
            .repeat(c2w.shape[0], c2w.shape[1], 1, 1)
            .clone()
        )
        intrinsics[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
        intrinsics[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
        intrinsics[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
        intrinsics[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5

    # 5. Apply metric scaling to metric depth
    metric_output.depth = apply_metric_scaling(metric_output.depth, intrinsics)

    # 6. Apply depth alignment (least-squares scaling)
    non_sky_mask = compute_sky_mask(metric_output.sky, threshold=0.3)
    bsz = depth.shape[0]
    scale_factors = torch.ones((bsz,), dtype=depth.dtype, device=depth.device)

    for b in range(bsz):
        non_sky_mask_b = non_sky_mask[b]
        if non_sky_mask_b.sum() <= 10:
            continue

        depth_conf_ns = depth_conf[b][non_sky_mask_b]
        if depth_conf_ns.numel() == 0:
            continue

        depth_conf_sampled = sample_tensor_for_quantile(depth_conf_ns, max_samples=100000)
        median_conf = torch.quantile(depth_conf_sampled, 0.5)

        align_mask = compute_alignment_mask(
            depth_conf[b],
            non_sky_mask_b,
            depth[b],
            metric_output.depth[b],
            median_conf,
        )

        if align_mask.sum() == 0:
            continue

        valid_depth = depth[b][align_mask]
        valid_metric_depth = metric_output.depth[b][align_mask]
        scale_factor = least_squares_scale_scalar(valid_metric_depth, valid_depth)
        if torch.isfinite(scale_factor):
            scale_factors[b] = scale_factor

    depth = depth * scale_factors[:, None, None, None]
    c2w[:, :, :3, 3] = c2w[:, :, :3, 3] * scale_factors[:, None, None]
    ray[..., 3:] = ray[..., 3:] * scale_factors[:, None, None, None, None]

    # 7. Handle sky regions
    non_sky_depth = depth[non_sky_mask]
    if non_sky_depth.numel() > 100000:
        idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
        sampled_depth = non_sky_depth[idx]
    else:
        sampled_depth = non_sky_depth
    non_sky_max = min(torch.quantile(sampled_depth, 0.99), 200.0)

    depth, depth_conf = set_sky_regions_to_max_depth(
        depth,
        depth_conf,
        non_sky_mask,
        max_depth=non_sky_max,
    )

    # 8. Compute point cloud from depth and pose
    pointmap = da3_depth_to_pointcloud(depth, intrinsics, c2w)

    return {
        "pointmap": pointmap,
        "depth": depth,
        "depth_conf": depth_conf,
        "c2w": c2w,
        "intrinsics": intrinsics,
    }


def run_reconstruction(
    args: argparse.Namespace,
    recon_views: List[Dict[str, torch.Tensor]],
    model_assets,
) -> Dict[str, torch.Tensor]:
    from occany.da3_inference import inference_occany_da3
    from occany.utils.inference_helper import convert_da3_output_to_occany_format

    if args.model == "da3":
        da3_model, da3_metric_model = model_assets
        da3_output = _run_da3_metric_scaling_inference(
            recon_views,
            da3_model,
            da3_metric_model,
            args.device,
        )
        return convert_da3_output_to_occany_format(da3_output)

    if args.model == "occany_da3":
        recon_output = inference_occany_da3(
            recon_views,
            model_assets,
            args.device,
            dtype=torch.float32,
            sam_model="SAM2",
            pose_from_depth_ray=args.pose_from_depth_ray,
            point_from_depth_and_pose=args.point_from_depth_and_pose,
        )
        recon_output.pop("aux_feats", None)
        recon_output.pop("aux_outputs", None)
        return convert_da3_output_to_occany_format(recon_output)

    raise ValueError(f"Unsupported model: {args.model}")


def flatten_valid_points(pointmap: torch.Tensor, valid_mask: torch.Tensor) -> np.ndarray:
    points_flat = pointmap.reshape(-1, 3)
    valid_mask_flat = valid_mask.reshape(-1)
    return points_flat[valid_mask_flat].detach().cpu().numpy()


def sanitize_sample_id(sample_id: str) -> str:
    safe_id = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in sample_id)
    safe_id = safe_id.strip("_")
    return safe_id if safe_id else "sample"


def save_point_cloud_txt(point_cloud: np.ndarray, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if point_cloud.size == 0:
        save_path.write_text("", encoding="utf-8")
        return
    np.savetxt(save_path, point_cloud, fmt="%.6f")


def extract_pred_point_cloud(
    recon_output: Dict[str, torch.Tensor],
    sample_idx: int,
) -> np.ndarray:
    pts3d = recon_output["pts3d"][sample_idx]
    conf = recon_output["conf"][sample_idx]
    depth = recon_output.get("pts3d_local", recon_output["pts3d"])[sample_idx][..., 2]
    valid_mask = (
        torch.isfinite(conf)
        & torch.isfinite(pts3d).all(dim=-1)
        & torch.isfinite(depth)
        & (depth <= MAX_POINT_CLOUD_DEPTH_METERS)
    )
    return flatten_valid_points(pts3d, valid_mask)


def extract_gt_point_cloud(
    gt_depths: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_poses: torch.Tensor,
    sample_idx: int,
) -> np.ndarray:
    sample_depths = gt_depths[sample_idx : sample_idx + 1]
    sample_intrinsics = intrinsics[sample_idx : sample_idx + 1]
    sample_camera_poses = camera_poses[sample_idx : sample_idx + 1]
    gt_pointmap = convert_depth_to_point_cloud(sample_depths, sample_intrinsics, sample_camera_poses)
    valid_mask = (
        (sample_depths > 0)
        & (sample_depths <= MAX_POINT_CLOUD_DEPTH_METERS)
        & torch.isfinite(sample_depths)
        & torch.isfinite(gt_pointmap).all(dim=-1)
    )
    return flatten_valid_points(gt_pointmap, valid_mask)


def get_sample_id(args: argparse.Namespace, data: Dict[str, List[str]], sample_idx: int) -> str:
    if args.dataset == "kitti":
        return f"{data['sequence'][sample_idx]}_{int(data['begin_frame_id'][sample_idx]):06d}"
    return f"{data['scene_name'][sample_idx]}_{data['begin_frame_token'][sample_idx]}"


def finite_or_none(value: float):
    return float(value) if np.isfinite(value) else None


def aggregate_metrics(per_sample_results: List[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {
        "num_samples": len(per_sample_results),
        "num_valid_samples": sum(
            1 for item in per_sample_results if item["overall"] is not None
        ),
        "mean_pred_points": float(
            np.mean([item["num_pred_points"] for item in per_sample_results])
        ) if per_sample_results else 0.0,
        "mean_gt_points": float(
            np.mean([item["num_gt_points"] for item in per_sample_results])
        ) if per_sample_results else 0.0,
    }
    for key in METRIC_KEYS:
        finite_values = [
            float(item[key])
            for item in per_sample_results
            if item[key] is not None
        ]
        aggregate[key] = float(np.mean(finite_values)) if finite_values else None
    return aggregate


def print_aggregate_metrics(title: str, aggregate: Dict[str, object]) -> None:
    def _format_value(value: object, is_percent: bool = False) -> str:
        if value is None:
            return "None"
        if isinstance(value, (float, np.floating)):
            numeric_value = float(value) * 100.0 if is_percent else float(value)
            suffix = "%" if is_percent else ""
            return f"{numeric_value:.5f}{suffix}"
        return str(value)

    print("=" * 60)
    print(title)
    print(f"num_samples:      {aggregate['num_samples']}")
    print(f"num_valid_samples:{aggregate['num_valid_samples']}")
    for key in METRIC_KEYS:
        print(f"{key:>10}: {_format_value(aggregate[key], is_percent=(key in PERCENT_METRIC_KEYS))}")
    print(f"mean_pred_points: {_format_value(aggregate['mean_pred_points'])}")
    print(f"mean_gt_points:   {_format_value(aggregate['mean_gt_points'])}")


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()
    validate_args(args)

    from occany.datasets.eval_helper import prepare_eval_setting
    from occany.model.must3r_blocks.attention import toggle_memory_efficient_attention

    toggle_memory_efficient_attention(enabled=True)
    output_resolution = get_output_resolution(args.model, args.dataset)
    output_dir = build_output_dir(args)

    if not args.silent:
        print(f"Output directory: {output_dir}")
        print(f"Output resolution: {output_resolution}")

    if args.model == "da3":
        model_assets = load_da3_models(
            args.device,
            args.da3_model_name,
            args.da3_metric_model_name,
        )
    else:
        model_assets = load_occany_plus_model(
            args.device,
            output_resolution,
            args.occany_recon_ckpt,
            args.da3_model_name,
        )

    base_model = "da3"
    dataset, collate_fn, recon_view_idx = prepare_eval_setting(
        dataset=args.dataset,
        setting=args.setting,
        image_size=IMAGE_SIZE,
        process_id=0,
        num_worlds=1,
        split=args.split,
        base_model=base_model,
        kitti_root=args.kitti_root,
        nuscenes_root=args.nuscenes_root,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    per_sample_results: List[Dict[str, object]] = []
    point_cloud_dir = output_dir / "cloudcompare_txt"
    saved_pointcloud_pairs = 0

    for data in tqdm(data_loader, desc=f"Evaluating {args.dataset}/{args.setting}"):
        recon_views, recon_gt_depths, recon_intrinsics, recon_camera_poses = build_recon_views(
            data=data,
            recon_view_idx=recon_view_idx,
            device=args.device,
            frame_interval=args.frame_interval,
        )

        with torch.inference_mode():
            recon_output = run_reconstruction(args, recon_views, model_assets)

        batch_size = recon_output["pts3d"].shape[0]
        for sample_idx in range(batch_size):
            sample_id = get_sample_id(args, data, sample_idx)
            pred_point_cloud = extract_pred_point_cloud(
                recon_output=recon_output,
                sample_idx=sample_idx,
            )
            gt_point_cloud = extract_gt_point_cloud(
                gt_depths=recon_gt_depths,
                intrinsics=recon_intrinsics,
                camera_poses=recon_camera_poses,
                sample_idx=sample_idx,
            )
            metrics = evaluate_3d_reconstruction(
                pred_point_cloud,
                gt_point_cloud,
                threshold=args.metric_threshold,
            )

            result = {
                "sample_id": sample_id,
                "num_pred_points": int(pred_point_cloud.shape[0]),
                "num_gt_points": int(gt_point_cloud.shape[0]),
            }
            for key in METRIC_KEYS:
                result[key] = finite_or_none(metrics[key])
            per_sample_results.append(result)

            should_save_pair = (
                args.save_pointcloud_pairs > 0
                and saved_pointcloud_pairs < args.save_pointcloud_pairs
                and ((len(per_sample_results) - 1) % args.save_pointcloud_stride == 0)
            )
            if should_save_pair:
                safe_sample_id = sanitize_sample_id(sample_id)
                pair_prefix = f"{saved_pointcloud_pairs:04d}_{safe_sample_id}"
                pred_path = point_cloud_dir / f"{pair_prefix}_pred.txt"
                gt_path = point_cloud_dir / f"{pair_prefix}_gt.txt"
                save_point_cloud_txt(pred_point_cloud, pred_path)
                save_point_cloud_txt(gt_point_cloud, gt_path)
                saved_pointcloud_pairs += 1

                if not args.silent:
                    print(
                        f"Saved CloudCompare pair {saved_pointcloud_pairs}/{args.save_pointcloud_pairs}: "
                        f"pred={pred_path.name}, gt={gt_path.name}",
                    )

            if not args.silent and len(per_sample_results) % 50 == 0:
                running_aggregate = aggregate_metrics(per_sample_results)
                print_aggregate_metrics(
                    f"Aggregate reconstruction metrics after {len(per_sample_results)} samples",
                    running_aggregate,
                )
                print("=" * 60)

        del recon_output
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    aggregate = aggregate_metrics(per_sample_results)
    results = {
        "config": {
            "model": args.model,
            "dataset": args.dataset,
            "setting": args.setting,
            "split": args.split,
            "image_size": IMAGE_SIZE,
            "metric_threshold": args.metric_threshold,
            "save_pointcloud_pairs": args.save_pointcloud_pairs,
            "save_pointcloud_stride": args.save_pointcloud_stride,
        },
        "aggregate": aggregate,
        "per_sample": per_sample_results,
    }

    results_path = output_dir / "recon_metrics.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    print_aggregate_metrics("Aggregate reconstruction metrics", aggregate)
    if args.save_pointcloud_pairs > 0:
        print(
            f"Saved {saved_pointcloud_pairs} point-cloud pair(s) under: {point_cloud_dir}",
        )
    print(f"Saved metrics to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
