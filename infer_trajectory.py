"""NuScenes trajectory evaluation with OccAny DA3 model.

Runs OccAny inference on NuScenes Vista video sequences, extracts camera
poses from predicted outputs, and computes ADE trajectory metrics against
ground-truth annotations.

Usage:
    python infer_trajectory.py \
        --data-folder /path/to/nuscenes \
        --anno-file vista_nuscenes_anno/nuScenes_val.json \
        --occany-ckpt ./pretrained_ckpts/occany_plus_1B.pth \
        --img-size 294 518
"""

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _collate_fn(batch):
    """Stack images and trajectories from individual samples."""
    images = []
    trajectories = {k: [] for k in ("gt_trajectory",) if k in batch[0]}

    for sample in batch:
        images.append(sample["images"])
        for k in trajectories:
            trajectories[k].append(sample[k])

    collated = {"images": torch.stack(images, dim=0)}
    for k, v in trajectories.items():
        collated[k] = torch.stack(v, dim=0)
    return collated


def _build_occany_recon_views(images):
    """Prepare list-of-dicts input expected by inference_occany_da3."""
    if images.ndim != 5:
        raise ValueError(f"Expected (B,T,C,H,W), got {tuple(images.shape)}")

    bsz, n_frames, _, h, w = images.shape
    true_shape = [[int(h), int(w)] for _ in range(bsz)]
    views = []
    for t in range(n_frames):
        views.append({
            "img": images[:, t],
            "true_shape": true_shape,
            "timestep": torch.full((bsz,), t, dtype=torch.int64),
        })
    return views


def load_occany_model(args):
    """Load the OccAny DA3 reconstruction model from checkpoint."""
    from occany.utils.io_da3 import load_da3_model_from_checkpoint

    print(f"Loading OccAny model from: {args.occany_ckpt}")
    model, checkpoint_args = load_da3_model_from_checkpoint(
        weights_path=args.occany_ckpt,
        output_resolution=args.img_size,
        semantic_feat_src=None,
        semantic_family=None,
        device=args.device,
        is_gen_model=False,
    )
    return model


@torch.inference_mode()
def run_inference(args):
    """Run trajectory evaluation over the NuScenes Vista dataset."""
    # Import directly to avoid heavy __init__.py in occany.datasets
    from occany.datasets.nuscenes_vista import NuscenesVistaDataset
    from occany.da3_inference import inference_occany_da3
    from occany.trajectory_eval import evaluate_trajectory_batch, save_trajectory_metrics

    # Dataset
    dataset = NuscenesVistaDataset(
        data_root=args.data_folder,
        num_frames=args.n_frames,
        img_size=args.img_size,
        crop_img_size=args.crop_img_size,
        anno_file=args.anno_file,
    )
    print(f"NuScenes Vista samples: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=_collate_fn,
    )

    # Model
    model = load_occany_model(args)
    model.eval()

    # Output dirs
    os.makedirs(args.output_dir, exist_ok=True)
    trajectory_plot_dir = os.path.join(args.output_dir, "trajectory_plots")
    metrics_path = os.path.join(args.output_dir, "ade_metrics.json")
    if args.plot_every > 0:
        os.makedirs(trajectory_plot_dir, exist_ok=True)

    # Inference loop
    ade_scores = []
    evaluated_samples = 0

    bar = tqdm(dataloader, leave=False, dynamic_ncols=True, desc="Trajectory eval")
    for i_bar, batch in enumerate(bar):
        if batch["images"].numel() == 0:
            continue

        video_tensor = batch["images"].to(args.device)
        expected_hw = args.crop_img_size if args.crop_img_size is not None else args.img_size
        assert (
            video_tensor.ndim == 5
            and tuple(video_tensor.shape[1:]) == (args.n_frames, 3, *expected_hw)
        ), (
            f"Expected (B,T,C,H,W) with T={args.n_frames}, C=3, HxW={tuple(expected_hw)}; "
            f"got {tuple(video_tensor.shape)}"
        )

        recon_views = _build_occany_recon_views(video_tensor)
  
        output = inference_occany_da3(
            recon_views,
            model,
            args.device,
            dtype=torch.float32,
            sam_model="SAM2",
            pose_from_depth_ray=True,
            point_from_depth_and_pose=False,
        )

        batch_ade, batch_ades = evaluate_trajectory_batch(
            batch=batch,
            output=output,
            evaluated_samples=evaluated_samples,
            plot_every=args.plot_every,
            trajectory_plot_dir=trajectory_plot_dir,
        )
        if len(batch_ades) > 0:
            ade_scores.extend(batch_ades)
            evaluated_samples += len(batch_ades)

        if batch_ade is not None and len(ade_scores) > 0:
            bar.set_postfix(last_ADE=f"{batch_ade:.3f}", mean_ADE=f"{np.mean(ade_scores):.3f}")

        if args.max_batches > 0 and (i_bar + 1) >= args.max_batches:
            break

    # Save metrics
    if len(ade_scores) > 0:
        save_trajectory_metrics(
            metrics_path=metrics_path,
            ade_scores=ade_scores,
            n_frames=args.n_frames,
            height=expected_hw[0],
            width=expected_hw[1],
            data_root=args.data_folder,
            anno_file=args.anno_file,
        )
        mean_ade = float(np.mean(ade_scores))
        print(f"\nFinished {len(ade_scores)} samples.")
        print(f"Mean ADE: {mean_ade:.4f}")
        print(f"Saved ADE metrics to: {metrics_path}")
        if args.plot_every > 0:
            print(f"Saved trajectory plots to: {trajectory_plot_dir}")
    else:
        print("No trajectory samples evaluated.")


def main():
    parser = argparse.ArgumentParser(
        description="NuScenes trajectory evaluation with OccAny DA3 model"
    )

    # Data
    parser.add_argument("--data-folder", type=str, required=True,
                        help="NuScenes data root folder")
    parser.add_argument("--anno-file", type=str, default="vista_nuscenes_anno/nuScenes_val.json",
                        help="Vista NuScenes annotation JSON")
    parser.add_argument("--img-size", type=int, nargs=2, required=True, metavar=("H", "W"),
                        help="Image size as H W")
    parser.add_argument("--crop-img-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Optional center crop size as H W")
    parser.add_argument("--n-frames", type=int, default=25,
                        help="Number of frames per sequence")

    # Model
    parser.add_argument("--occany-ckpt", type=str, required=True,
                        help="Path to OccAny checkpoint (.pth)")

    # Inference control
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size")
    parser.add_argument("--num-workers", type=int, default=16,
                        help="DataLoader workers")
    parser.add_argument("--max-batches", type=int, default=-1,
                        help="Limit number of batches (-1 = all)")
    parser.add_argument("--plot-every", type=int, default=20,
                        help="Save trajectory plot every N samples (0 disables)")
    parser.add_argument("--output-dir", type=str, default="./outputs/occany_nuscenes_traj",
                        help="Output directory for metrics and plots")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    args = parser.parse_args()

    # Validate
    args.img_size = tuple(args.img_size)
    if args.crop_img_size is not None:
        args.crop_img_size = tuple(args.crop_img_size)

    if not os.path.exists(args.occany_ckpt):
        parser.error(f"OccAny checkpoint not found: {args.occany_ckpt}")

    if not os.path.exists(args.anno_file):
        parser.error(f"Annotation file not found: {args.anno_file}")

    args.device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Running on device: {args.device}")

    run_inference(args)


if __name__ == "__main__":
    main()
