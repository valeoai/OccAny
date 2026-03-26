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
from occany.datasets.nuscenes import NuScenesDataset, collate_nuscenes_identity
from occany.datasets.class_mapping import ClassMapping
from occany.semantic_inference import infer_semantic
from torch.utils.data import DataLoader
import argparse
from occany.utils.helpers import save_semantic_2d_images

pl.ion()

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", "-sz", type=int, default=512, choices=[512, 1024, 1328], help="image size")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--silent", action='store_true', default=False,
                        help="silence logs")
    parser.add_argument('--semantic', type=str, choices=['pretrained@SAM2_large'], default='pretrained@SAM2_large', help='Semantic processing option')
    parser.add_argument('--box_threshold', type=float, default=0.1, help='Box threshold for Grounding DINO')
    parser.add_argument('--text_threshold', type=float, default=0.0, help='Text threshold for Grounding DINO')
    parser.add_argument('--pid', type=int, default=0, help='Process ID for parallel processing')
    parser.add_argument('--world', type=int, default=1, help='Total number of parallel processes')
    
    return parser



if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    print(args.box_threshold, args.text_threshold)
    
    if args.image_size == 224:
        output_resolution = (224, 144)
    elif args.image_size == 512:
        output_resolution = (512, 288)
    elif args.image_size == 1024:
        output_resolution = (1024, 576)
    elif args.image_size == 1328:
        output_resolution = (1328, 747)
    else:
        raise ValueError(f"Invalid image size: {args.image_size}")

    # print("scale by gt depth:", args.scale_by_gt_depth)

    preprocessed_root = os.path.join(os.environ['SCRATCH'], 'data', 'nuscenes_processed')
    if not args.silent:
        print('Outputting stuff in', preprocessed_root)


    class_mapping = ClassMapping()
    
    
    if "TRG_WORK" in os.environ:
        raise ValueError("This script only works on karolina")
    else:
        nuscenes_root = os.path.join(os.environ['PROJECT'], "data", "nuscenes")


    dataset = NuScenesDataset(split="val",
                                root=nuscenes_root,
                                output_resolution=output_resolution,
                                boxes_dir=None,
                                apply_camera_mask=True,
                                camera_names=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                                'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                                video_length=1,
                                apply_lidar_mask=False,
                                pid=args.pid,
                                world=args.world)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=10,
        shuffle=False,
        collate_fn=collate_nuscenes_identity,
    )
    
    if len(dataset.OCC3D_CATEGORIES) != len(dataset.CLASS_NAMES):
        raise ValueError(
            "NuScenes coarse prompt groups must stay aligned with CLASS_NAMES: "
            f"{len(dataset.OCC3D_CATEGORIES)} groups vs {len(dataset.CLASS_NAMES)} class names."
        )
    
    FINE_CLASSES = sum(dataset.OCC3D_CATEGORIES[1:], []) # exclude other
    TEXT_PROMPT = '. '.join(FINE_CLASSES)
    COARSE_TO_FINE_INDEX_MAPPING = [
        outer_index for outer_index, inner_list in enumerate(dataset.OCC3D_CATEGORIES[1:])
        for _ in inner_list
    ]

    saved_one_to_verify = False

    # Process the subset assigned to this worker
    for batch_i, data in enumerate(tqdm(data_loader, desc=f"Process {args.pid}/{args.world}")):
        gdino_imgs = data["gdino_imgs"].to(args.device)
        sam2_imgs = data["sam2_imgs"].to(args.device)
        
        scene_name = data['scene_name'][0]
        frame_token = data['begin_frame_token'][0]
        camera_names = data['camera_names'][0]  # List of all camera names
        
        # Process each camera in the surround view
        num_cameras = len(camera_names)
        for cam_idx in range(num_cameras):
            camera_name = camera_names[cam_idx]
            box_th_str = f"{int(args.box_threshold * 100)}"
            text_th_str = f"{int(args.text_threshold * 100)}"
            save_dir = os.path.join(preprocessed_root, f"resized_{args.image_size}_box{box_th_str}_text{text_th_str}_DINOB", scene_name, f"{frame_token}_{camera_name}")
            os.makedirs(save_dir, exist_ok=True)

            fine_semantic_2ds, boxes, confidences, fine_labels = infer_semantic(
                gdino_imgs[0, cam_idx].unsqueeze(0), sam2_imgs[0, cam_idx].unsqueeze(0), args.semantic, FINE_CLASSES, args.device,
                box_threshold=args.box_threshold, text_threshold=args.text_threshold,
                image_size=sam2_imgs.shape[-1],
                return_boxes=saved_one_to_verify
            )
            fine_label_ids = [FINE_CLASSES.index(fine_label) for fine_label in fine_labels]
            
            label_ids = [COARSE_TO_FINE_INDEX_MAPPING[fine_label_id] + 1 for fine_label_id in fine_label_ids] # Account for excluding other class
            labels = [dataset.CLASS_NAMES[label_id] for label_id in label_ids]
            
            keep_indices = [idx for idx, label_id in enumerate(label_ids) if label_id != dataset.other_class]
            boxes = np.asarray(boxes)[keep_indices]
            confidences = np.asarray(confidences)[keep_indices]
            labels = [labels[idx] for idx in keep_indices]

            if not saved_one_to_verify:                
                semantic_2ds = np.zeros_like(fine_semantic_2ds)
                for fine_label_id, label_id in zip(fine_label_ids, label_ids):
                    semantic_2ds[fine_semantic_2ds==fine_label_id] = label_id
                save_semantic_2d_images(semantic_2ds[None], save_dir, dataset.COLORS, verbose=not args.silent)
                saved_one_to_verify = True
            
            np.savez_compressed(os.path.join(save_dir, "boxes.npz"),
                        boxes=boxes,
                        confidences=confidences,
                        labels=labels)
            if not args.silent:
                print(f"Saved {save_dir}/boxes.npz")
        torch.cuda.empty_cache()
            
      
