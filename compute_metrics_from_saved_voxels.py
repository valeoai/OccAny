#!/usr/bin/env python3
"""
Compute SSC metrics from saved voxel predictions.
This script aggregates voxel predictions saved by multiple processes and computes metrics.
Ground truth labels are loaded directly from the dataset.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

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

from occany.metrics.ssc import SSCMetrics
from occany.utils.helpers import (
    apply_majority_pooling,
    apply_unified_majority_pooling,
    print_metrics_summary,
)
from occany.datasets.eval_helper import prepare_metric_eval_setting


def get_args_parser():
    parser = argparse.ArgumentParser(description='Compute SSC metrics from saved voxel predictions')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Base output directory containing sample subdirectories')
    parser.add_argument('--dataset', type=str, required=True, choices=['kitti', 'nuscenes'],
                        help='Dataset name (kitti or nuscenes)')
    parser.add_argument('--threshold', type=float, nargs='+', required=False,
                        help='Confidence threshold value(s) to evaluate (can specify multiple, for backward compatibility)')
    parser.add_argument('--recon_threshold', type=float, nargs='+', required=False,
                        help='Reconstruction confidence threshold value(s) to evaluate')
    parser.add_argument('--gen_threshold', type=float, nargs='+', required=False,
                        help='Generation confidence threshold value(s) to evaluate')
    
    # Dataset-specific arguments
    parser.add_argument('--kitti_root', type=str, default=None,
                        help='Path to the KITTI root containing SemanticKITTI voxel labels')
    parser.add_argument('--nuscenes_root', type=str, default=None,
                        help='Path to NuScenes root')
    parser.add_argument('--output_type', type=str, nargs='+', default=['render'],
                        help='Output type(s) to evaluate (e.g., render, render_gen, render_recon_gen)')
    parser.add_argument('--setting', type=str, default='1frame', choices=['1frame', '5frames', 'surround'],
                        help='Evaluation setting')
    parser.add_argument('--eval_superclass', action='store_true', default=False,
                        help='Enable superclass evaluation in addition to original')
    parser.add_argument('--apply_majority_pooling', action='store_true', default=False,
                        help='Apply majority pooling to voxel predictions (3x3x3 neighborhood)')
    parser.add_argument('--pooling_mode', type=str, default='separate', choices=['separate', 'unified'],
                        help='Pooling mode when --apply_majority_pooling is enabled (default: separate)')
    parser.add_argument('--geometry_only', action='store_true', default=False,
                        help='Only for separate pooling: apply geometry-only occupancy pooling')
    parser.add_argument('--no_majority_pooling', action='store_true', default=False,
                        help='Disable majority pooling even if --apply_majority_pooling is set (useful to override exp list defaults)')
    return parser







def save_pts3d_txt(pts3d, labels, filepath):
    """Save point cloud as txt file for CloudCompare.
    
    Format: X Y Z Label (space-separated)
    
    Args:
        pts3d: (N, 3) array of point coordinates
        labels: (N,) array of semantic labels
        filepath: path to save the txt file
    """
    # Combine coordinates and labels
    data = np.column_stack([pts3d, labels])
    
    # Save as space-separated txt file
    np.savetxt(filepath, data, fmt='%.6f %.6f %.6f %d', 
               header='X Y Z Label', comments='')


def main():
    args = get_args_parser().parse_args()
    
    # Determine if using new format (separate recon/gen thresholds) or old format (single threshold)
    use_separate_thresholds = args.recon_threshold is not None or args.gen_threshold is not None
    
    if use_separate_thresholds:
        if args.recon_threshold is None or args.gen_threshold is None:
            raise ValueError("Both --recon_threshold and --gen_threshold must be specified when using separate thresholds")
        recon_thresholds = args.recon_threshold if isinstance(args.recon_threshold, list) else [args.recon_threshold]
        gen_thresholds = args.gen_threshold if isinstance(args.gen_threshold, list) else [args.gen_threshold]
        thresholds = None  # Not used in new format
    else:
        if args.threshold is None:
            raise ValueError("Either --threshold or both --recon_threshold and --gen_threshold must be specified")
        thresholds = args.threshold if isinstance(args.threshold, list) else [args.threshold]
        recon_thresholds = None
        gen_thresholds = None
    
    output_types = args.output_type if isinstance(args.output_type, list) else [args.output_type]
    
    print("Evaluating predictions with:")
    print(f"  Output types: {output_types}")
    if use_separate_thresholds:
        print(f"  Recon thresholds: {recon_thresholds}")
        print(f"  Gen thresholds: {gen_thresholds}")
    else:
        print(f"  Thresholds: {thresholds}")
    print(f"  Output dir: {args.exp_dir}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Setting: {args.setting}")
    print(f"  Apply majority pooling: {args.apply_majority_pooling}")
    print(f"  Pooling mode: {args.pooling_mode}")
    print(f"  Geometry only (separate mode): {args.geometry_only}")


    if args.apply_majority_pooling and args.geometry_only and args.pooling_mode != 'separate':
        print("[WARNING] --geometry_only is ignored when --pooling_mode is not 'separate'")

    # Initialize dataset to load ground truth
    dataset, collate_fn, recon_view_idx = prepare_metric_eval_setting(
        dataset=args.dataset,
        setting=args.setting,
        process_id=0,
        num_worlds=1,
        split='val',
        kitti_root=args.kitti_root,
        nuscenes_root=args.nuscenes_root,
    )
    
    if args.dataset == 'kitti':
        ignore_other_class_in_mIoU = True
        
    elif args.dataset == 'nuscenes':
        ignore_other_class_in_mIoU = False
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    eval_modes = ['original']
    if args.eval_superclass:
        eval_modes.append('superclass')
    
    print(f"Initialized {args.dataset} dataset with {len(dataset)} samples")
    print(f"Evaluation modes: {eval_modes}")
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Determine which label types to evaluate
    label_types = ["full"]
    print(f"Evaluating label types: {label_types}")
    
    # Initialize metrics for all combinations
    all_metrics = {}
    import itertools
    for eval_mode in eval_modes:
        for label_type in label_types:
            for output_type in output_types:
                # Set up classes based on eval mode
                if eval_mode == 'original':
                    n_classes = len(dataset.CLASS_NAMES)
                    class_names = dataset.CLASS_NAMES
                    empty_class = dataset.empty_class
                else:  # superclass
                    n_classes = len(dataset.SUPERCLASS_NAMES)
                    class_names = dataset.SUPERCLASS_NAMES
                    empty_class = dataset.superclass_empty_class
                
                # For render_recon_gen: use dual thresholds or single threshold
                if 'render_recon_gen' in output_type:
                    if use_separate_thresholds:
                        # Dual threshold mode: use _recon{}_gen{} format
                        for recon_th, gen_th in itertools.product(recon_thresholds, gen_thresholds):
                            metric_key = f"{eval_mode}_{output_type}_recon{recon_th}_gen{gen_th}_{label_type}"
                            all_metrics[metric_key] = SSCMetrics(
                                n_classes=n_classes,
                                class_names=class_names,
                                other_class=dataset.other_class,
                                empty_class=empty_class,
                                ignore_other_class_in_mIoU=ignore_other_class_in_mIoU
                            )
                    else:
                        # Single threshold mode: use _th{} format
                        for threshold in thresholds:
                            metric_key = f"{eval_mode}_{output_type}_{threshold}_{label_type}"
                            all_metrics[metric_key] = SSCMetrics(
                                n_classes=n_classes,
                                class_names=class_names,
                                other_class=dataset.other_class,
                                empty_class=empty_class,
                                ignore_other_class_in_mIoU=ignore_other_class_in_mIoU
                            )
                # For render: use single threshold
                else:
                    if use_separate_thresholds:
                        # Use recon_thresholds for render output
                        threshold_list = recon_thresholds
                    else:
                        threshold_list = thresholds
                    
                    for threshold in threshold_list:
                        metric_key = f"{eval_mode}_{output_type}_{threshold}_{label_type}"
                        all_metrics[metric_key] = SSCMetrics(
                            n_classes=n_classes,
                            class_names=class_names,
                            other_class=dataset.other_class,
                            empty_class=empty_class,
                            ignore_other_class_in_mIoU=ignore_other_class_in_mIoU
                        )
    
    # Track processed and skipped counts for each combination
    counts = {key: {'processed': 0, 'skipped': 0} for key in all_metrics.keys()}

    def maybe_apply_pooling(voxel_pred_np):
        if not args.apply_majority_pooling or args.no_majority_pooling:
            return voxel_pred_np

        if args.pooling_mode == 'separate':
            return apply_majority_pooling(
                voxel_pred_np,
                n_classes=len(dataset.CLASS_NAMES),
                other_class=dataset.other_class,
                empty_class=dataset.empty_class,
                is_geometry_only=args.geometry_only,
            )

        return apply_unified_majority_pooling(
            voxel_pred_np,
            n_classes=len(dataset.CLASS_NAMES),
            other_class=dataset.other_class,
            empty_class=dataset.empty_class,
        )
    
    # Setup class mapping for superclass mode
    class_mapping_array = np.array(dataset.MAPPING, dtype=np.uint8)    
    # Process each sample in the data loader
    print(f"\nProcessing {len(dataset)} samples from dataset...")
    pbar = tqdm(data_loader, desc="Processing samples")
    sample_idx = 0
    for data in pbar:
        sample_idx += 1
        
        # Get sample identifiers (handle batch dimension)
        if args.dataset == 'kitti':
            seq_name = str(data['sequence'][0])
            frame_str = f"{data['begin_frame_id'][0]:06d}"
        else:  # nuscenes
            seq_name = data['scene_name'][0]
            frame_str = data['begin_frame_token'][0]
        
        # Get ground truth voxel label (remove batch dimension)
        voxel_label_original = data['voxel_label'][0].numpy()
    
        
        # Load voxel predictions dictionary if it exists (for new format)
        voxel_predictions_dict = None
        dict_path = os.path.join(args.exp_dir, f"{seq_name}_{frame_str}", "voxel_predictions.pkl")
        if os.path.exists(dict_path):
            with open(dict_path, 'rb') as f:
                voxel_predictions_dict = pickle.load(f)
        assert voxel_predictions_dict is not None, f"Voxel predictions not found for {seq_name}_{frame_str}"

        
        # For each item, evaluate all combinations of eval_mode, output_type and threshold
        for output_type in output_types:
            # For render_recon_gen: use dual thresholds or single threshold
            if 'render_recon_gen' in output_type:
                if use_separate_thresholds:
                    # Dual threshold mode: use _recon{}_gen{} format
                    threshold_configs = []
                    for recon_th, gen_th in itertools.product(recon_thresholds, gen_thresholds):
                        threshold_configs.append((recon_th, gen_th, f"_recon{recon_th}_gen{gen_th}", f"_recon{recon_th}_gen{gen_th}"))
                else:
                    # Single threshold mode: use _th{} format
                    threshold_configs = [(th, th, f"_th{th}", f"_{th}") for th in thresholds]
                
                for recon_th, gen_th, file_suffix, metric_suffix in threshold_configs:
                    # Load voxel prediction
                    voxel_pred_np = None
                    key = f"{output_type}{file_suffix}"
                    if voxel_predictions_dict is not None and key in voxel_predictions_dict:
                        voxel_pred_np = voxel_predictions_dict[key]
                    else:
                        # Fallback to .npy file
                        pred_path = os.path.join(args.exp_dir, f"{seq_name}_{frame_str}", f"{output_type}{file_suffix}.npy")
                        if os.path.exists(pred_path):
                            voxel_pred_np = np.load(pred_path)
                    
                    if voxel_pred_np is None:
                        # Mark as skipped for all eval modes and label types
                        for eval_mode in eval_modes:
                            for label_type in label_types:
                                metric_key = f"{eval_mode}_{output_type}{metric_suffix}_{label_type}"
                                counts[metric_key]['skipped'] += 1
                        continue
                    
                    voxel_pred_original = maybe_apply_pooling(voxel_pred_np)
                    
                    # Evaluate for each eval mode
                    for eval_mode in eval_modes:
                        # Prepare voxel_label and voxel_pred based on eval_mode
                        if eval_mode == 'original':
                            voxel_label = voxel_label_original
                            voxel_pred = voxel_pred_original
                        else:  # superclass - remap both pred and gt
                            # Remap ground truth
                            voxel_label = voxel_label_original.copy()
                            valid_mask = voxel_label < len(class_mapping_array)
                            voxel_label[valid_mask] = class_mapping_array[voxel_label[valid_mask].astype(np.int32)]
                            
                            # Remap prediction
                            voxel_pred = voxel_pred_original.copy()
                            valid_mask_pred = voxel_pred < len(class_mapping_array)
                            voxel_pred[valid_mask_pred] = class_mapping_array[voxel_pred[valid_mask_pred].astype(np.int32)]
                        
                        for label_type in label_types: 
                            metric_key = f"{eval_mode}_{output_type}{metric_suffix}_{label_type}"
                            # Add batch to metrics
                            
                            if label_type == "full":
                                all_metrics[metric_key].add_batch(voxel_pred, voxel_label)
                            counts[metric_key]['processed'] += 1
            # For render: use single threshold
            else:
                # Determine threshold list based on format
                if use_separate_thresholds:
                    threshold_list = recon_thresholds
                else:
                    threshold_list = thresholds
                
                for threshold in threshold_list:
                    # Load voxel prediction
                    voxel_pred_np = None
                    key = f"{output_type}_th{threshold}"
                    if voxel_predictions_dict is not None and key in voxel_predictions_dict:
                        voxel_pred_np = voxel_predictions_dict[key]
                    else:
                        # Fallback to .npy file
                        pred_path = os.path.join(args.exp_dir, f"{seq_name}_{frame_str}", f"{output_type}_th{threshold}.npy")
                        if os.path.exists(pred_path):
                            voxel_pred_np = np.load(pred_path)
                    
                    if voxel_pred_np is None:
                        # Mark as skipped for all eval modes and label types
                        for eval_mode in eval_modes:
                            for label_type in label_types:
                                metric_key = f"{eval_mode}_{output_type}_{threshold}_{label_type}"
                                counts[metric_key]['skipped'] += 1
                        continue
                    
                    voxel_pred_original = maybe_apply_pooling(voxel_pred_np)

                  
                    # Evaluate for each eval mode
                    for eval_mode in eval_modes:
                        # Prepare voxel_label and voxel_pred based on eval_mode
                        if eval_mode == 'original':
                            voxel_label = voxel_label_original
                            voxel_pred = voxel_pred_original
                        else:  # superclass - remap both pred and gt
                            # Remap ground truth
                            voxel_label = voxel_label_original.copy()
                            valid_mask = voxel_label < len(class_mapping_array)
                            voxel_label[valid_mask] = class_mapping_array[voxel_label[valid_mask].astype(np.int32)]
                            
                            # Remap prediction
                            voxel_pred = voxel_pred_original.copy()
                            valid_mask_pred = voxel_pred < len(class_mapping_array)
                            voxel_pred[valid_mask_pred] = class_mapping_array[voxel_pred[valid_mask_pred].astype(np.int32)]
                        
                        for label_type in label_types: 
                            metric_key = f"{eval_mode}_{output_type}_{threshold}_{label_type}"
                            # Add batch to metrics
                            if label_type == "full":
                                all_metrics[metric_key].add_batch(voxel_pred, voxel_label)
                            counts[metric_key]['processed'] += 1
        
        # Print intermediate metrics every 10 items
        if sample_idx % 10 == 0 or sample_idx == len(dataset):
            print("\n" + "=" * 80, flush=True)
            print("=" * 80, flush=True)
            # Print metrics grouped by eval_mode to avoid class name confusion
            for eval_mode in eval_modes:
                mode_metrics = {k: v for k, v in all_metrics.items() if k.startswith(eval_mode)}
                if mode_metrics:
                    print(f"\n{'='*40} {eval_mode.upper()} MODE {'='*40}", flush=True)
                    print_metrics_summary(mode_metrics)
    
   


if __name__ == "__main__":
    main()
