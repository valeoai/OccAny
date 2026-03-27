# --------------------------------------------------------
# gradio demo
# --------------------------------------------------------

import os
import sys
from pathlib import Path
import torch
import numpy as np
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
from occany.datasets.kitti import KittiDataset, collate_kitti_identity
from occany.datasets.class_mapping import ClassMapping
from occany.semantic_inference import infer_semantic
from torch.utils.data import DataLoader
import argparse
from occany.utils.helpers import save_semantic_2d_images

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def scale_depth_with_gt(depths, gt_depths, conf):
    """
    Scale depth using ground truth depth information.
    
    Args:
        pts3d_local: Local 3D points tensor
        gt_depths: Ground truth depth data
        pts3d: Global 3D points tensor to be scaled
    
    Returns:
        torch.Tensor: Scaled 3D points
    """
    # Extract depth from local 3D points
    assert conf.shape == depths.shape, f"conf.shape: {conf.shape}, depths.shape: {depths.shape}"
    assert conf.shape == gt_depths.shape, f"conf.shape: {conf.shape}, gt_depths.shape: {gt_depths.shape}"
    # Compute scale factor using ground truth depth
    scale, mask = compute_gt_depth_scale(
                            depths.reshape(-1), gt_depths.reshape(-1), conf.reshape(-1),
                            optimize_shift=False,
                            max_depth=50,
                            use_gpu=True,
                            max_iters=300,
                            tol=1e-6,
                            lr=0.1)
    return scale
    # # Apply scale to the 3D points


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", "-sz", type=int, default=1216, choices=[512, 1024, 768, 1216], help="image size")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument('--semantic', type=str, choices=['pretrained@SAM2_large'], default='pretrained@SAM2_large', help='Semantic processing option')
    parser.add_argument('--box_threshold', type=float, default=0.1, help='Box threshold for Grounding DINO')
    parser.add_argument('--text_threshold', type=float, default=0.0, help='Text threshold for Grounding DINO')
    parser.add_argument('--pid', type=int, default=0, help='Process ID for parallel processing')
    parser.add_argument('--world', type=int, default=1, help='Total number of parallel processes')
    
    return parser





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



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    if args.image_size == 512:
        output_resolution = (512, 160)
    elif args.image_size == 768:
        output_resolution = (768, 240)
    elif args.image_size == 1024:
        output_resolution = (1024, 288)
    elif args.image_size == 1216:
        output_resolution = (1216, 342)
    else:
        raise ValueError(f"Invalid image size: {args.image_size}")


    preprocessed_root = os.path.join(os.environ['SCRATCH'], 'data/kitti_processed')
    if not args.silent:
        print('Outputting stuff in', preprocessed_root)


    class_mapping = ClassMapping()
    
    
    if "DSDIR" in os.environ:
        kitti_root = os.environ["DSDIR"]
    else:
        kitti_root = os.environ.get("KITTI_ROOT", f"{os.environ['PROJECT']}/data/kitti")


    kitti_dataset = KittiDataset(split='val',
                                 output_resolution=output_resolution,
                                 semkitti_root=kitti_root,
                                 kittiodo_root=kitti_root,
                                 remap_lut_path=os.path.join(os.path.dirname(__file__), 'occany', 'datasets', 'semantic_kitti.yaml'),
                                 video_length=1,
                                 video_interval=5,
                                 frame_interval=5,
                                 pid=args.pid,
                                 world=args.world)

    kitti_loader = DataLoader(
        kitti_dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        collate_fn=collate_kitti_identity,
    )

    # Set up classes and prompts
    TEXT_PROMPT_LIST = kitti_dataset.PROMPT
    CLASS_NAMES = kitti_dataset.CLASS_NAMES
    COLORS = kitti_dataset.COLORS
    
    FINE_CLASSES = sum(TEXT_PROMPT_LIST, [])
    TEXT_PROMPT = '. '.join(FINE_CLASSES)
    COARSE_TO_FINE_INDEX_MAPPING = [
        outer_index for outer_index, inner_list in enumerate(TEXT_PROMPT_LIST)
        for _ in inner_list
    ]


    # Process the subset assigned to this worker
    for batch_i, data in enumerate(tqdm(kitti_loader, desc=f"Process {args.pid}/{args.world}")):
        gdino_imgs = data["gdino_imgs"].to(args.device)
        sam2_imgs = data["sam2_imgs"].to(args.device)
       
        seq_name = f"{data['sequence'][0]}"
        begin_frame_id = data['begin_frame_id'][0]
        box_th_str = f"{int(args.box_threshold * 100)}"
        text_th_str = f"{int(args.text_threshold * 100)}"
        save_dir = os.path.join(preprocessed_root, f"resized_{args.image_size}_box{box_th_str}_text{text_th_str}_DINOB", f"{seq_name}_{begin_frame_id:06d}")
        os.makedirs(save_dir, exist_ok=True)
      
        fine_semantic_2ds, boxes, confidences, fine_labels = infer_semantic(
            gdino_imgs[0], sam2_imgs[0], args.semantic, FINE_CLASSES, args.device,
            box_threshold=args.box_threshold, text_threshold=args.text_threshold,
            image_size=sam2_imgs.shape[-1]
        )
        fine_label_ids = [FINE_CLASSES.index(fine_label) for fine_label in fine_labels]
     
        label_ids = [COARSE_TO_FINE_INDEX_MAPPING[fine_label_id] for fine_label_id in fine_label_ids]
        labels = [CLASS_NAMES[label_id] for label_id in label_ids]
        semantic_2ds = np.zeros_like(fine_semantic_2ds)
        for fine_label_id, label_id in zip(fine_label_ids, label_ids):
            semantic_2ds[fine_semantic_2ds==fine_label_id] = label_id
            
        save_semantic_2d_images(semantic_2ds[None], save_dir, COLORS, verbose=not args.silent)
       
        np.savez_compressed(os.path.join(save_dir, "boxes.npz"),
                    boxes=boxes,
                    confidences=confidences,
                    labels=labels)
        if not args.silent:
            print(f"Saved {save_dir}/boxes.npz")
        torch.cuda.empty_cache()
            
      
