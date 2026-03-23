import torch
import sys
import importlib
import yaml
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import tv_tensors
from torch.cuda.amp.autocast_mode import autocast
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
from tqdm import tqdm
import argparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
# CUDA 12.4
import cv2
import supervision as sv
import logging
logger = logging.getLogger(__name__)


if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



def main():
    # Add argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequences', nargs='+', default=["08", "00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
                        help="List of sequences to process")
    parser.add_argument('--box_threshold', type=float, default=0.1,
                        help="Box threshold for grounding dino")
    parser.add_argument('--no_high_res_feature', action='store_true',
                        help="not use high resolution feature")
    args = parser.parse_args()
    
    
    
    kitti_colors = np.array([
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
            [0, 205, 254, 255]
        ]).astype(np.uint8)

    kitti_class_names = [
        "empty", # "empty", we map sky to empty
        "car",
        "bicycle",
        "motorcycle",
        "truck",
        "other vehicle",
        "person",
        "bicyclist",
        "motorcyclist",
        "road",
        "parking",
        "sidewalk",
        "other ground",
        "building",
        "fence",
        "vegetation",
        "trunk",
        "terrain",
        "pole",
        "traffic sign",
        "sky",
    ]

    kitti2idx = {name: i for i, name in enumerate(kitti_class_names)}


    device = 0
    SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
    SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # BOX_THRESHOLD = 0.35
    BOX_THRESHOLD = args.box_threshold
    # TEXT_THRESHOLD = 0.25
    TEXT_THRESHOLD = 0.0 # this should be 0.0 when setting remove_combined=True
    
    GROUNDING_DINO_CONFIG = "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "gdino_checkpoints/groundingdino_swint_ogc.pth" 
    
    kitti_class_names = [f"{name}." for name in kitti_class_names]
    text = " ".join(kitti_class_names[1:])
    
    # build SAM2 image predictor
    sam2_checkpoint = SAM2_CHECKPOINT
    model_cfg = SAM2_MODEL_CONFIG
    
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE, 
                            use_high_res_features_in_sam=not args.no_high_res_feature)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    logger.info("Loaded SAM2 model")
    
    # build grounding dino model
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG, 
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE
    )
    logger.info("Loaded Grounding DINO model")


    # save_root_dir = "/lustre/fsn1/projects/rech/kvd/uyl37fq/monst3r_output/semantic_kitti_output"
    SCRATCH_DIR = os.environ["SCRATCH"]
    save_root_dir = os.path.join(SCRATCH_DIR, "data/kitti_processed")
    


    
        
    # image_dir = "/lustre/fsmisc/dataset/KITTI/odometry/dataset/sequences/08/image_2"
    kitti_root = "/lustre/fsmisc/dataset/KITTI/odometry/dataset"
    # sequences = ["08"]
    logger.info("Processing sequences: ", args.sequences)
    logger.info(f"Box threshold: {BOX_THRESHOLD:.2f}")
    # Use args.sequences instead of hardcoded list
    for sequence in args.sequences:
        image_dir = os.path.join(kitti_root, "sequences", sequence, "image_2")
        folder_name = f"image_2_grounded_sam2"
        if args.no_high_res_feature:
            folder_name = folder_name + "_nohighRes"
        save_dir = os.path.join(save_root_dir, "sequences", sequence, f"{folder_name}_{BOX_THRESHOLD:.2f}")
        
        os.makedirs(save_dir, exist_ok=True)
        for root, dirs, files in tqdm(os.walk(image_dir), desc=f"Processing sequence {sequence}"):
            for img_name in tqdm(files, desc=f"Processing images", leave=False):
                img_path = os.path.join(root, img_name)
                
                image_name = img_name.split('.')[0]
                png_save_path = os.path.join(save_dir, f"{image_name}.png")
                sem2d_save_path = os.path.join(save_dir, f"{image_name}_sem2d.png")
                sem2d_color_save_path = os.path.join(save_dir, f"{image_name}_sem2d_color.jpg")
                mask_save_path = os.path.join(save_dir, f"{image_name}_mask.jpg")
                
                if os.path.exists(png_save_path) and os.path.exists(sem2d_save_path):
                    print(f"skipping {img_path} because it already exists")
                    continue

                if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    image_source, image = load_image(img_path)
                    sam2_predictor.set_image(image_source)
                    
                    boxes, confidences, labels = predict(
                        model=grounding_model,
                        image=image,
                        caption=text,
                        box_threshold=BOX_THRESHOLD,
                        text_threshold=TEXT_THRESHOLD,
                        remove_combined=True,
                    )
                    # Filter labels to only include those in kitti2idx
                    valid_indices = [i for i, label in enumerate(labels) if label in kitti2idx]
                    boxes = boxes[valid_indices]
                    confidences = confidences[valid_indices]
                    labels = [labels[i] for i in valid_indices]
                    
                    
              
                    # process the box prompt for SAM 2
                    h, w, _ = image_source.shape
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        # operations that should use mixed precision
                        masks, scores, logits = sam2_predictor.predict(
                            point_coords=None,
                            point_labels=None,
                            box=input_boxes,
                            multimask_output=False,
                        )
                        
                    # convert the shape to (n, H, W)
                    if masks.ndim == 4:
                        masks = masks.squeeze(1)
                    
                    class_names = labels
                    confidences = confidences.numpy().tolist()
                    class_ids = np.array(list(range(len(class_names))))

                    labels = [
                        f"{class_name} {confidence:.2f}"
                        for class_name, confidence
                        in zip(class_names, confidences)
                    ]

                    """
                    Visualize image with supervision useful API
                    """
                    img = cv2.imread(img_path)
                    detections = sv.Detections(
                        xyxy=input_boxes,  # (n, 4)
                        mask=masks.astype(bool),  # (n, h, w)
                        class_id=class_ids
                    )

                    # box_annotator = sv.BoxAnnotator()
                    # annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

                    
                    # cv2.imwrite(png_save_path, annotated_frame)
                    

                    mask_annotator = sv.MaskAnnotator()
                    annotated_frame = mask_annotator.annotate(scene=img.copy(), detections=detections)
                    label_annotator = sv.LabelAnnotator()
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    cv2.imwrite(mask_save_path, annotated_frame)
                    print(f"saved {mask_save_path}")
                    
                    kitti_class_ids = [kitti2idx[name] for name in class_names]
                    
                    # Sort masks by confidence descending so higher confidence masks get priority
                    sorted_indices = np.argsort(confidences)[::-1]
                    masks = masks[sorted_indices]
                    kitti_class_ids = np.array(kitti_class_ids)[sorted_indices]

                    # Create empty semantic map and apply masks
                    sem2d = np.zeros((h, w))
                    for class_id, mask in zip(kitti_class_ids, masks):
                        # Use logical AND with inverse of existing mask to prevent overwrites
                        sem2d = np.where(mask.astype(bool) & (sem2d == 0), class_id, sem2d)
                    sem2d = sem2d.astype(np.uint8)
                    
                    # Save as single-channel PNG
                    # NOTE: to read this image, use cv2.imread(sem2d_save_path, cv2.IMREAD_UNCHANGED) 
                    cv2.imwrite(sem2d_save_path, sem2d)
                    print(f"saved {sem2d_save_path}")
                    
                    
                    
                    sem2d_color = kitti_colors[sem2d]
                    # Convert from RGBA to BGR format for proper JPG saving
                    sem2d_color_bgr = cv2.cvtColor(sem2d_color, cv2.COLOR_RGBA2BGR)
                    cv2.imwrite(sem2d_color_save_path, sem2d_color_bgr)
                    print(f"saved {sem2d_color_save_path}")
                    
                    # breakpoint()

                # break  # Stop after processing the first image in the current folder
            
            
if __name__ == "__main__":
    main()
