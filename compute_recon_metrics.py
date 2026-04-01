#!/usr/bin/env python3
"""
Compute reconstruction metrics from saved extraction outputs.

Loads per-sample .npz files produced by extract_recon.py and computes
3D reconstruction and depth metrics.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
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
    REPO_ROOT / "third_party" / "pyTorchChamferDistance",
]
for vendored_path in reversed(VENDORED_IMPORT_PATHS):
    vendored_path_str = str(vendored_path)
    if vendored_path.exists() and vendored_path_str not in sys.path:
        sys.path.insert(0, vendored_path_str)

from occany.utils.recon_eval_helper import (
    evaluate_all_metrics,
    evaluate_depth_metrics,
)

METRIC_KEYS = ("acc", "comp", "overall", "precision", "recall", "fscore")
DEPTH_METRIC_KEYS = ("abs_rel", "sq_rel", "rmse", "log_rmse", "delta_1", "delta_2", "delta_3")
ALL_METRIC_KEYS = METRIC_KEYS + DEPTH_METRIC_KEYS
PERCENT_METRIC_KEYS = {"precision", "recall", "fscore", "delta_1", "delta_2", "delta_3"}


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute reconstruction metrics from saved extraction outputs.",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Directory containing per-sample .npz files from extract_recon.py.",
    )
    parser.add_argument(
        "--metric_threshold",
        type=float,
        default=float(os.environ.get("METRIC_THRESHOLD", 0.5)),
        help="Distance threshold in meters for precision / recall / f-score.",
    )
    parser.add_argument(
        "--depth_max",
        type=float,
        default=float(os.environ.get("DEPTH_MAX", 80.0)),
        help="Maximum GT depth (metres) for depth-metric validity mask.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=os.environ.get("DEVICE", "cuda"),
        help="PyTorch device for accelerated point-cloud metric computation.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        default=os.environ.get("SILENT", "").strip().lower() in {"1", "true", "yes", "on"},
        help="Reduce per-sample logging.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=int(os.environ.get("NUM_WORKERS", 16)),
        help="Number of DataLoader worker processes for parallel .npz loading.",
    )
    return parser


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
    for key in ALL_METRIC_KEYS:
        finite_values = [
            float(item[key])
            for item in per_sample_results
            if item.get(key) is not None
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

    print("--- 3D Reconstruction ---")
    for key in METRIC_KEYS:
        print(f"{key:>16}: {_format_value(aggregate[key], is_percent=(key in PERCENT_METRIC_KEYS))}")

    print("--- Depth Metrics ---")
    for key in DEPTH_METRIC_KEYS:
        print(f"{key:>16}: {_format_value(aggregate[key], is_percent=(key in PERCENT_METRIC_KEYS))}")

    print(f"mean_pred_points: {_format_value(aggregate['mean_pred_points'])}")
    print(f"mean_gt_points:   {_format_value(aggregate['mean_gt_points'])}")


class ReconNpzDataset(torch.utils.data.Dataset):
    def __init__(self, npz_files):
        self.npz_files = npz_files

    def __len__(self):
        return len(self.npz_files)

    def __getitem__(self, idx):
        npz_path = self.npz_files[idx]
        data = np.load(npz_path)
        return {
            "sample_id": npz_path.stem,
            "pred_points": data["pred_points"],
            "gt_points": data["gt_points"],
            "pred_depth": data["pred_depth"],
            "gt_depth": data["gt_depth"],
        }


def main() -> None:
    parser = get_args_parser()
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    npz_files = sorted(exp_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in: {exp_dir}")

    print(f"Found {len(npz_files)} sample(s) in: {exp_dir}")
    print(f"Metric threshold: {args.metric_threshold} m, depth max: {args.depth_max} m")

    # Load config from any extraction metadata file if available
    config = {}
    metadata_files = sorted(exp_dir.glob("extraction_metadata_pid*.json"))
    if metadata_files:
        with metadata_files[0].open("r", encoding="utf-8") as f:
            metadata = json.load(f)
            config = metadata.get("config", {})

    per_sample_results: List[Dict[str, object]] = []

    dataset = ReconNpzDataset(npz_files)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.num_workers,
        collate_fn=lambda batch: batch[0],
    )

    for sample in tqdm(loader, desc="Computing reconstruction metrics", disable=args.silent):
        sample_id = sample["sample_id"]
        pred_points = sample["pred_points"]
        gt_points = sample["gt_points"]
        pred_depth = torch.from_numpy(sample["pred_depth"])
        gt_depth = torch.from_numpy(sample["gt_depth"])

        # Single accelerated pass for 3D reconstruction metrics
        all_metrics = evaluate_all_metrics(
            pred_points,
            gt_points,
            threshold=args.metric_threshold,
            device=args.device,
        )
        metrics = all_metrics["recon"]

        depth_metrics = evaluate_depth_metrics(pred_depth, gt_depth, max_depth=args.depth_max)

        result: Dict[str, object] = {
            "sample_id": sample_id,
            "num_pred_points": int(pred_points.shape[0]),
            "num_gt_points": int(gt_points.shape[0]),
        }
        for key in METRIC_KEYS:
            result[key] = finite_or_none(metrics[key])
        for key in DEPTH_METRIC_KEYS:
            result[key] = finite_or_none(depth_metrics[key]) if depth_metrics[key] is not None else None
        per_sample_results.append(result)

        if not args.silent and len(per_sample_results) % max(1, len(npz_files) // 10) == 0:
            running_aggregate = aggregate_metrics(per_sample_results)
            print_aggregate_metrics(
                f"Running metrics after {len(per_sample_results)}/{len(npz_files)} samples",
                running_aggregate,
            )

    aggregate = aggregate_metrics(per_sample_results)
    results = {
        "config": {
            **config,
            "metric_threshold": args.metric_threshold,
            "depth_max": args.depth_max,
        },
        "aggregate": aggregate,
        "per_sample": per_sample_results,
    }

    results_path = exp_dir / "recon_metrics.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)

    print_aggregate_metrics("Aggregate reconstruction metrics", aggregate)
    print(f"Saved metrics to: {results_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
