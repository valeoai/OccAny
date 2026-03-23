# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import numpy as np
from occany.model.model_sam2 import SAM2
from occany.model.sam3_model import Sam3ModelManager
from groundingdino.util.inference import load_model, predict
from torchvision.ops import box_convert
import os
import inspect
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _normalize_grounding_label(label: str) -> str:
    return label.replace("-", " ").replace("_", " ").strip()


def _resolve_repo_path(*candidates: str) -> str:
    for candidate in candidates:
        candidate_path = REPO_ROOT / candidate
        if candidate_path.exists():
            return str(candidate_path)
    return str(REPO_ROOT / candidates[0])


def _ensure_groundingdino_transformers_compat() -> None:
    """Patch the installed `transformers` Bert API to match GroundingDINO's expectations.

    GroundingDINO's vendored `BertModelWarper` was written against an older
    `transformers` release where `BertModel` exposed `get_head_mask()` and where
    `get_extended_attention_mask()` accepted the older call pattern used by the
    model wrapper. The local environment uses a newer `transformers`, so we
    adapt the runtime API here instead of modifying vendored code.
    """
    from transformers.models.bert.modeling_bert import BertModel

    # Older GroundingDINO code expects `BertModel.get_head_mask(...)` to exist.
    # Newer `transformers` versions removed it, so we recreate the small helper
    # that the wrapper relies on when it builds attention masks.
    if not hasattr(BertModel, "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers

            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

            if head_mask.dim() != 5:
                raise ValueError(
                    f"head_mask.dim != 5, got {head_mask.dim()} with shape {tuple(head_mask.shape)}"
                )

            head_mask = head_mask.to(dtype=self.embeddings.word_embeddings.weight.dtype)
            if is_attention_chunked:
                head_mask = head_mask.unsqueeze(-1)
            return head_mask

        BertModel.get_head_mask = _get_head_mask

    # `get_extended_attention_mask(...)` changed signature across transformers
    # releases. GroundingDINO still calls the older form, so we wrap the current
    # method and translate the legacy `device` argument into the dtype-based API.
    get_extended_attention_mask = getattr(BertModel, "get_extended_attention_mask", None)
    if get_extended_attention_mask is None:
        raise AttributeError("BertModel is missing get_extended_attention_mask")

    if getattr(get_extended_attention_mask, "__occany_device_compat__", False):
        return

    param_names = list(inspect.signature(get_extended_attention_mask).parameters)
    if len(param_names) >= 4 and param_names[3] == "dtype":
        original_get_extended_attention_mask = get_extended_attention_mask

        def _compat_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            # GroundingDINO passes a device-like value here; the current
            # implementation expects a dtype, so fall back to the model dtype.
            if isinstance(dtype, (torch.device, str)):
                dtype = self.embeddings.word_embeddings.weight.dtype
            return original_get_extended_attention_mask(self, attention_mask, input_shape, dtype=dtype)

        # Mark the wrapper so we only install it once even if the loader is
        # called repeatedly during a session.
        _compat_get_extended_attention_mask.__occany_device_compat__ = True
        BertModel.get_extended_attention_mask = _compat_get_extended_attention_mask


class ModelManager:
    """Singleton class to manage model loading and caching"""
    _instance = None
    
    GROUNDING_DINO_CONFIG = _resolve_repo_path(
        "third_party/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    )
    GROUNDING_DINO_CHECKPOINT = _resolve_repo_path(
        "checkpoints/groundingdino_swinb_cogcoor.pth",
        "third_party/Grounded-SAM-2/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
    )
    
    SAM2_CONFIGS = {
        "SAM2_large": {
            "checkpoint_path": _resolve_repo_path(
                "checkpoints/sam2.1_hiera_large.pt",
                "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt",
            ),
            "config_path": "configs/sam2.1/sam2.1_hiera_l.yaml"
        }
    }
    
    def __new__(cls, device=None):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance._initialize(device)
        return cls._instance
    
    def _initialize(self, device):
        """Initialize the model manager"""
        self.device = device
        self.grounding_dino_model = None
        self.sam2_model = None

    def get_grounding_dino(self):
        """Get or load the Grounding DINO model"""
        if self.grounding_dino_model is None:
            print("Loading Grounding DINO model...")
            _ensure_groundingdino_transformers_compat()
            self.grounding_dino_model = load_model(
                self.GROUNDING_DINO_CONFIG, 
                self.GROUNDING_DINO_CHECKPOINT, 
                device=self.device
            )
        return self.grounding_dino_model
    
    def get_sam2(self, model_type, load_video_model=False, image_size=512):
        """Get or load a SAM2 model by type"""
        if self.sam2_model is None or (not self.sam2_model.load_video_model and load_video_model):
            print(f"Loading {model_type} model...")
            config = self.SAM2_CONFIGS.get(model_type)
            if not config:
                raise ValueError(f"Unknown SAM2 model type: {model_type}")
        
            self.sam2_model = SAM2(
                model_cfg=config["config_path"],
                sam2_checkpoint=config["checkpoint_path"],
                device=self.device,
                load_video_model=load_video_model,
                image_size=image_size
            )
        return self.sam2_model




def select_sam_feature_views(
    sam_feats: Optional[List[torch.Tensor]],
    view_ids: List[int],
    n_total_views: int,
    context: str = "",
) -> Optional[List[torch.Tensor]]:
    """Select view subsets from distilled SAM features for memory-efficient inference."""
    if sam_feats is None or len(view_ids) == 0:
        return None

    prefix = f"[{context}] " if context else ""
    selected_feats: List[torch.Tensor] = []
    for level_idx, feat in enumerate(sam_feats[:3]):
        if feat.dim() == 5:
            selected_feats.append(feat[:, view_ids])
            continue

        if feat.dim() == 4:
            if n_total_views <= 0 or feat.shape[0] % n_total_views != 0:
                print(
                    f"[WARNING] {prefix}Cannot subset SAM features at level {level_idx}: "
                    f"shape {tuple(feat.shape)} is incompatible with n_total_views={n_total_views}"
                )
                return sam_feats[:3]
            batch_size = feat.shape[0] // n_total_views
            reshaped_feat = feat.reshape(batch_size, n_total_views, *feat.shape[1:])
            selected = reshaped_feat[:, view_ids]
            selected_feats.append(selected.reshape(batch_size * len(view_ids), *feat.shape[1:]))
            continue

        print(
            f"[WARNING] {prefix}Cannot subset SAM features at level {level_idx}: "
            f"unsupported feature rank {feat.dim()}"
        )
        return sam_feats[:3]

    return selected_feats

    
def build_sam3_inference_state(
    sam_feats: Optional[List[torch.Tensor]],
    batch_idx: int,
    n_views: int,
    original_height: int,
    original_width: int,
    pos_enc: Optional[Any] = None,
    context: str = "",
) -> Optional[Dict[str, Any]]:
    """Build SAM3 inference state from distilled feature maps."""
    prefix = f"[{context}] " if context else ""
    if sam_feats is None:
        print(f"[WARNING] {prefix}SAM3 distilled features are missing")
        return None
    if len(sam_feats) < 3:
        print(f"[WARNING] {prefix}Expected at least 3 SAM3 feature levels, got {len(sam_feats)}")
        return None

    feat_levels = sam_feats[:3]
    if feat_levels[0].dim() == 5:
        feat_s0 = feat_levels[0][batch_idx]
        feat_s1 = feat_levels[1][batch_idx]
        feat_s2 = feat_levels[2][batch_idx]
    elif feat_levels[0].dim() == 4:
        start_idx = batch_idx * n_views
        end_idx = (batch_idx + 1) * n_views
        feat_s0 = feat_levels[0][start_idx:end_idx]
        feat_s1 = feat_levels[1][start_idx:end_idx]
        feat_s2 = feat_levels[2][start_idx:end_idx]
    else:
        print(f"[WARNING] {prefix}Unexpected SAM3 feature rank: {feat_levels[0].dim()}")
        return None

    if feat_s0.shape[0] != n_views or feat_s1.shape[0] != n_views or feat_s2.shape[0] != n_views:
        print(
            f"[WARNING] {prefix}SAM3 feature view count mismatch "
            f"(expected {n_views}, got {feat_s0.shape[0]}, {feat_s1.shape[0]}, {feat_s2.shape[0]})"
        )
        return None

    if pos_enc is None:
        from sam3.model.position_encoding import PositionEmbeddingSine

        pos_enc = PositionEmbeddingSine(num_pos_feats=256, normalize=True)

    pos_s0 = pos_enc(feat_s0).to(feat_s0.dtype)
    pos_s1 = pos_enc(feat_s1).to(feat_s1.dtype)
    pos_s2 = pos_enc(feat_s2).to(feat_s2.dtype)

    return {
        "backbone_out": {
            "backbone_fpn": [feat_s0, feat_s1, feat_s2],
            "vision_features": feat_s2,
            "vision_pos_enc": [pos_s0, pos_s1, pos_s2],
        },
        "original_height": original_height,
        "original_width": original_width,
    }


def infer_sam2_feats(sam2_model_type, sam2_imgs, device):
    # Initialize model manager
    model_manager = ModelManager(device)

    sam2_model = model_manager.get_sam2(sam2_model_type)

    image_embed, feat_s1, feat_s0 = sam2_model.forward(sam2_imgs)
    high_res_feats = [feat_s0, feat_s1]
    return {"high_res_feats": high_res_feats, "image_embed": image_embed}


def infer_sam2_boxes(
    gdino_imgs,
    class_names,
    device,
    box_threshold=0.1,
    text_threshold=0.0,
):
    assert gdino_imgs.shape[0] == 1, "Only support batch size of 1"
    assert len(gdino_imgs.shape) == 4, "B, C, H, W"

    model_manager = ModelManager(device)
    grounding_dino_model = model_manager.get_grounding_dino()

    normalized_class_names = [_normalize_grounding_label(name) for name in class_names]
    normalized_to_original = {}
    for original_name, normalized_name in zip(class_names, normalized_class_names):
        normalized_to_original.setdefault(normalized_name, original_name)
    text = " ".join([f"{name}." for name in normalized_class_names])

    boxes, confidences, labels = predict(
        model=grounding_dino_model,
        image=gdino_imgs[0],
        caption=text,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
        remove_combined=True,
    )

    valid_indices = [i for i, label in enumerate(labels) if label in normalized_to_original]
    boxes = boxes[valid_indices]
    confidences = confidences[valid_indices]
    labels = [normalized_to_original[labels[i]] for i in valid_indices]

    return boxes.cpu().numpy(), confidences.cpu().numpy(), labels

def infer_sam3_feats(sam3_imgs, original_height, original_width, device, sam3_resolution=1008):
    sam3_manager = Sam3ModelManager(
            resolution=sam3_resolution,
            confidence_threshold=0.5,
        )
    sam3_processor = sam3_manager.get_sam3(device)
    state = sam3_processor.forward(sam3_imgs, 
                               original_height=original_height, 
                               original_width=original_width)
    return state

def infer_semantic_from_classname_and_sam3_inference_state(
    prompts,
    prompt_to_class_mapping,
    sam3_inference_state, 
    ignore_ids=[],
    empty_class=0,
    device='cuda',
    confidence_threshold=0.5,
    sam3_resolution=1008,
    view_batch_size=4
    # view_batch_size=2
):
    """Infer 2D semantics by querying SAM3 with text prompts and remapping to KITTI classes."""
    ignore_ids_set = set(ignore_ids)
    max_label_id = max([empty_class] + prompt_to_class_mapping)
    assert max_label_id <= torch.iinfo(torch.uint8).max, "Class indices must fit within torch.uint8"

    sam3_manager = Sam3ModelManager(
        resolution=sam3_resolution,
        confidence_threshold=confidence_threshold,
    )
    sam3_processor = sam3_manager.get_sam3(device)
    
    H = sam3_inference_state["original_height"]
    W = sam3_inference_state["original_width"]

    n_views = sam3_inference_state["backbone_out"]["vision_features"].shape[0]
    sem2d = torch.full((n_views, H, W), fill_value=empty_class, dtype=torch.uint8, device=device)
    
    # Prepare valid prompts and their corresponding class IDs
    valid_prompts = []
    prompt_to_class_id = []
    
    for prompt_idx, prompt in enumerate(prompts):
        class_id = prompt_to_class_mapping[prompt_idx]
        if class_id == empty_class or class_id in ignore_ids_set:
            continue
        valid_prompts.append(prompt)
        prompt_to_class_id.append(class_id)
    
    if len(valid_prompts) > 0:
        # Use batched prediction for all valid prompts and chunks of views to avoid OOM
        sam3_processor.reset_all_prompts(sam3_inference_state)
        
        num_prompts = len(valid_prompts)
        class_lookup = torch.tensor(prompt_to_class_id, device=device, dtype=torch.long)

        for i in tqdm(range(0, n_views, view_batch_size), desc="Inference 2D semantics (chunked)"):
            chunk_view_ids = list(range(i, min(i + view_batch_size, n_views)))
            
            mask_output = sam3_processor.predict_batched(
                prompts=valid_prompts, 
                state=sam3_inference_state, 
                image_ids=chunk_view_ids
            )
            
            if "masks" not in mask_output or len(mask_output["masks"]) == 0:
                continue
                
            masks = mask_output["masks"]  # [N, 1, H, W] boolean
            scores = mask_output["scores"]  # [N]
            task_indices = mask_output["prompt_indices"]  # [N]
            
            # Calculate relative view_id (within chunk) and prompt_idx for each prediction
            rel_view_ids = task_indices // num_prompts
            pred_prompt_indices = task_indices % num_prompts
            
            # Map back to absolute view_id
            chunk_view_ids_tensor = torch.tensor(chunk_view_ids, device=device, dtype=torch.long)
            pred_view_ids = chunk_view_ids_tensor[rel_view_ids]
            
            # Get label IDs
            pred_label_ids = class_lookup[pred_prompt_indices]
            
            # Iterate over each view that has predictions in this chunk
            unique_views = torch.unique(pred_view_ids)
            
            for view_id in unique_views.tolist():
                # Select predictions for this view
                view_mask_indices = (pred_view_ids == view_id)
                
                view_scores = scores[view_mask_indices]
                
                # Sort by confidence descending
                sorted_indices_local = torch.argsort(view_scores, descending=True)
                
                view_labels_sorted = pred_label_ids[view_mask_indices][sorted_indices_local]
                view_masks_sorted = masks[view_mask_indices][sorted_indices_local]
                
                for j in range(len(view_labels_sorted)):
                    label_id = view_labels_sorted[j]
                    if label_id.item() in ignore_ids_set:
                        continue
                    
                    mask = view_masks_sorted[j].squeeze(0) # [H, W]
                    
                    sem2d[view_id] = torch.where(mask & (sem2d[view_id] == empty_class), label_id, sem2d[view_id])
    
    return sem2d


def infer_semantic_from_boxes(sam2_model_type, 
                              H, W, label_ids, ignore_ids,
                              boxes, confidences, labels, 
                              sam2_feats, device):
    """
    boxes: np array of [N, 4]
    confidences: np array of [N]
    labels: list of labels [N]
    sam2_feats: {high_res_feats, image_embed}
    """

   
  
    boxes = torch.from_numpy(boxes)
    confidences = torch.from_numpy(confidences)
    
    # Initialize model manager
    model_manager = ModelManager(device)

    sam2_model = model_manager.get_sam2(sam2_model_type)
 
    
    # # Filter labels to only include those in kitti2idx
    # valid_indices = [i for i, label in enumerate(labels) if label in kitti2idx]
    # boxes = boxes[valid_indices]
    # confidences = confidences[valid_indices]
    # labels = [labels[i] for i in valid_indices]
    
    # process the box prompt for SAM 2
    boxes = boxes * torch.Tensor([W, H, W, H])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    sam2_model.set_image_features(
        high_res_feats=sam2_feats['high_res_feats'],
        image_embed=sam2_feats['image_embed'],
        img_hws=[[H, W]]
    )
    
    masks = sam2_model.predict_masks(boxes=input_boxes)
    
    # Sort masks by confidence descending so higher confidence masks get priority
    confidences = confidences.cpu().numpy()
    sorted_indices = np.argsort(confidences)[::-1]
    masks = masks[sorted_indices]
    # class_ids = [class2idx[name] for name in labels]
    label_ids = np.array(label_ids)[sorted_indices]
    
    # common_class_ids = class_mapping.map_kitti_id_2_common_id(kitti_class_ids)
    
    # Create empty semantic map and apply masks
    sem2d = np.zeros((H, W))
    for label_id, mask in zip(label_ids, masks):
        if label_id in ignore_ids:
            continue
        # Use logical AND with inverse of existing mask to prevent overwrites
        sem2d = np.where(mask.astype(bool) & (sem2d == 0), label_id, sem2d)
    sem2d = sem2d.astype(np.uint8)
    return sem2d


def reproject_boxes(boxes_xyxy, depth_map, c2w_src, c2w_tgt, focal_src, focal_tgt, H, W):
    """
    Reproject 2D boxes from source view to target view using depth and camera poses.
    
    Args:
        boxes_xyxy: np array of [N, 4] in format (x1, y1, x2, y2) in pixel coordinates
        depth_map: np array of shape [H, W] depth map of source view
        c2w_src: np array of shape [4, 4] camera-to-world matrix of source view
        c2w_tgt: np array of shape [4, 4] camera-to-world matrix of target view
        focal_src: float, focal length of source view
        focal_tgt: float, focal length of target view
        H, W: int, image height and width
    
    Returns:
        boxes_tgt: np array of [N, 4] reprojected boxes in target view (x1, y1, x2, y2) in pixel coordinates
    """
    N = boxes_xyxy.shape[0]
    boxes_tgt = []
    
    # Compute world-to-camera matrix for target view
    w2c_tgt = np.linalg.inv(c2w_tgt)
    
    for i in range(N):
        x1, y1, x2, y2 = boxes_xyxy[i]
        
        # Project the 4 corners
        points_2d = np.array([
            [x1, y1],
            [x2, y1],
            [x1, y2],
            [x2, y2]
        ])
        
        # Clip corners to image bounds
        points_2d = np.clip(points_2d, [0, 0], [W-1, H-1])
        
        # Get depth at corner points
        points_2d_int = points_2d.astype(int)
        depths = depth_map[points_2d_int[:, 1], points_2d_int[:, 0]]
        
        # Filter out corners with invalid depth
        valid_mask = depths > 0
        if valid_mask.sum() == 0:
            # If no valid depth at any corner, use original box as fallback
            boxes_tgt.append(boxes_xyxy[i])
            continue
            
        points_2d = points_2d[valid_mask]
        depths = depths[valid_mask]
        
        # Unproject to 3D in source camera frame
        points_3d_cam_src = np.zeros((len(points_2d), 3))
        points_3d_cam_src[:, 0] = (points_2d[:, 0] - W/2) * depths / focal_src
        points_3d_cam_src[:, 1] = (points_2d[:, 1] - H/2) * depths / focal_src
        points_3d_cam_src[:, 2] = depths
        
        # Transform to homogeneous coordinates
        points_3d_cam_src_homo = np.concatenate([points_3d_cam_src, np.ones((len(points_3d_cam_src), 1))], axis=-1)
        
        # Transform to world coordinates
        points_3d_world = (c2w_src @ points_3d_cam_src_homo.T).T[:, :3]
        
        # Transform to target camera frame
        points_3d_world_homo = np.concatenate([points_3d_world, np.ones((len(points_3d_world), 1))], axis=-1)
        points_3d_cam_tgt = (w2c_tgt @ points_3d_world_homo.T).T[:, :3]
        
        # Project to target image
        # Filter out points behind the camera
        valid_depth = points_3d_cam_tgt[:, 2] > 0
        if valid_depth.sum() == 0:
            continue
            
        points_3d_cam_tgt = points_3d_cam_tgt[valid_depth]
        
        points_2d_tgt = np.zeros((len(points_3d_cam_tgt), 2))
        points_2d_tgt[:, 0] = points_3d_cam_tgt[:, 0] * focal_tgt / points_3d_cam_tgt[:, 2] + W/2
        points_2d_tgt[:, 1] = points_3d_cam_tgt[:, 1] * focal_tgt / points_3d_cam_tgt[:, 2] + H/2
        
        # Compute bounding box from projected points
        x_min, y_min = points_2d_tgt.min(axis=0)
        x_max, y_max = points_2d_tgt.max(axis=0)
        
        # Clip to image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W, x_max)
        y_max = min(H, y_max)
        
        # Return in xyxy format (pixel coordinates)
        box_tgt = np.array([x_min, y_min, x_max, y_max])
        boxes_tgt.append(box_tgt)
    
    return np.array(boxes_tgt)


def infer_semantic_from_boxes_and_sam2_feat_list(sam2_model_type, 
                              H, W, label_ids, ignore_ids,
                              boxes, confidences,
                              other_class, empty_class,
                              use_sam_video=False, 
                              sam2_feats_list=None, 
                              poses=None, 
                              focals=None, 
                              depth_maps=None,
                              device='cuda', 
                              output_type="numpy", box_conf_thres=0.05,
                              merge_masks=True):
    """
    boxes: np array of [N, 4]
    confidences: np array of [N]
    labels: list of labels [N]
    use_sam_video: bool, whether to use SAM video mode, if false, reproject the boxes of the first frames into other frames
    sam2_feats_list: [{high_res_feats, image_embed}]
    poses: np array of [num_views, 4, 4] camera-to-world matrices (required if use_sam_video=False)
    focals: np array of [num_views] focal lengths (required if use_sam_video=False)
    depth_maps: np array of [num_views, H, W] depth maps (required if use_sam_video=False)
    merge_masks: bool, whether to merge masks by label and binned confidence (default: True)
    """
    assert output_type in ["numpy", "torch"], "Output_type must be either numpy or torch"
    
    if not use_sam_video:
        # Validate required parameters for reprojection
        assert poses is not None, "poses must be provided when use_sam_video=False"
        assert focals is not None, "focals must be provided when use_sam_video=False"
        assert depth_maps is not None, "depth_maps must be provided when use_sam_video=False"
        assert len(poses) == len(sam2_feats_list), "Number of poses must match number of SAM2 features"
        assert len(focals) == len(sam2_feats_list), "Number of focals must match number of SAM2 features"
        assert len(depth_maps) == len(sam2_feats_list), "Number of depth maps must match number of SAM2 features"

    sam2_feats_0 = sam2_feats_list[0]
    
    boxes = torch.from_numpy(boxes) # N, 4
    confidences = torch.from_numpy(confidences) # N
    label_ids = np.array(label_ids) # N
    
    # Check if there are any boxes remaining after confidence filtering
    if len(boxes) == 0:
        print("No boxes remaining after confidence thresholding. Returning empty semantic map.")
        # Return empty results when no boxes are detected
        if output_type == "numpy":
            sem2d = np.full((len(sam2_feats_list), H, W), fill_value=empty_class, dtype=np.uint8)
            return sem2d
        else:
            sem2d = torch.full((len(sam2_feats_list), H, W), fill_value=empty_class, dtype=torch.uint8)
            return sem2d
    
    # Initialize model manager
    model_manager = ModelManager(device)

    # Only load video model if using SAM video mode
    sam2_model = model_manager.get_sam2(sam2_model_type, load_video_model=use_sam_video)

    
    # # Filter labels to only include those in kitti2idx
    # valid_indices = [i for i, label in enumerate(labels) if label in kitti2idx]
    # boxes = boxes[valid_indices]
    # confidences = confidences[valid_indices]
    # labels = [labels[i] for i in valid_indices]
    
    # process the box prompt for SAM 2
    boxes_np = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
    boxes = boxes * torch.Tensor([W, H, W, H])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()


    # For reprojection, we need pixel xyxy format
    input_boxes_xyxy = input_boxes
    
    if use_sam_video:
        # Use SAM2 video propagation mode
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Get masks of the first frame
            sam2_model.set_image_features(
                high_res_feats=sam2_feats_0['high_res_feats'],
                image_embed=sam2_feats_0['image_embed'],
                img_hws=[[H, W]]
            )
            masks = sam2_model.predict_masks(boxes=input_boxes) # (N, 160, 512)
            
            # Merge masks by label and binned confidence (0.01 bins)
            if merge_masks:
                confidences_np = confidences.cpu().numpy()
                conf_bins = np.round(confidences_np / 0.01).astype(int)  # Bin into 0.01 intervals
                
                # Create unique keys for (label_id, conf_bin) pairs
                merge_keys = list(zip(label_ids, conf_bins))
                unique_keys = []
                merged_masks = []
                merged_label_ids = []
                merged_confidences = []
                
                for key in set(merge_keys):
                    # Find all masks with this (label_id, conf_bin) combination
                    indices = [i for i, k in enumerate(merge_keys) if k == key]
                    
                    if len(indices) == 1:
                        # Only one mask, keep as is
                        merged_masks.append(masks[indices[0]])
                        merged_label_ids.append(key[0])
                        merged_confidences.append(confidences_np[indices[0]])
                    else:
                        # Multiple masks, merge by taking max (union)
                        mask_group = [masks[i] for i in indices]
                        merged_mask = np.maximum.reduce(mask_group)
                        merged_masks.append(merged_mask)
                        merged_label_ids.append(key[0])
                        # Use max confidence from the group
                        merged_confidences.append(max(confidences_np[i] for i in indices))
                        # print(f"Merged {len(indices)} masks for label_id={key[0]}, conf_bin={key[1]*0.01:.2f}")
                
                masks = merged_masks
                label_ids = merged_label_ids
                confidences = torch.from_numpy(np.array(merged_confidences)).to(confidences.device)
                print(f"After merging: {len(masks)} unique masks from {len(merge_keys)} original masks")
            
            inference_state = sam2_model.video_predictor.init_state(
                image_feats=sam2_feats_list,
                video_height=H,
                video_width=W
            )
            ann_frame_idx = 0
            for object_id, (label_id, mask) in enumerate(zip(label_ids, masks), start=1):
                _, out_obj_ids, out_mask_logits = sam2_model.video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=ann_frame_idx,
                    obj_id=object_id,
                    mask=mask
                )
            """
            Step 4: Propagate the video predictor to get the segmentation results for each frame
            """
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.video_predictor.propagate_in_video(inference_state, run_mem_encoder=True):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            # Reset SAM2 video predictor state to free memory
            sam2_model.video_predictor.reset_state(inference_state)
    else:
        # Use box reprojection mode
        video_segments = {}
        
        # Process each frame
        for frame_idx in range(len(sam2_feats_list)):
            # Reproject boxes from first frame to current frame
            if frame_idx == 0:
                # First frame uses original boxes (already in pixel xyxy format)
                input_boxes_frame = input_boxes_xyxy
                
            else:
                # Reproject boxes from first frame to current frame
                # reproject_boxes expects and returns pixel xyxy format
                boxes_frame_xyxy = reproject_boxes(
                    input_boxes_xyxy,  # pixel xyxy format
                    depth_maps[0],  # depth map of first frame
                    poses[0],  # pose of first frame
                    poses[frame_idx],  # pose of current frame
                    focals[0],  # focal of first frame
                    focals[frame_idx],  # focal of current frame
                    H, W
                )
                # Check if we got empty boxes after reprojection
                if boxes_frame_xyxy.shape[0] == 0:
                    video_segments[frame_idx] = {}
                    print(f"Empty boxes after reprojection for frame {frame_idx}")
                    continue
                input_boxes_frame = boxes_frame_xyxy
            
            # Get masks for current frame
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                sam2_model.set_image_features(
                    high_res_feats=sam2_feats_list[frame_idx]['high_res_feats'],
                    image_embed=sam2_feats_list[frame_idx]['image_embed'],
                    img_hws=[[H, W]]
                )
                masks_frame = sam2_model.predict_masks(boxes=input_boxes_frame)
            
            # Store masks in video_segments format (object_id -> mask)
            # Convert to numpy boolean arrays to match video mode format
            video_segments[frame_idx] = {}
            for object_id in range(len(masks_frame)):
                mask = masks_frame[object_id:object_id+1]
                # Convert to numpy boolean if it's a tensor
                if torch.is_tensor(mask):
                    mask = (mask > 0.0).cpu().numpy()
                else:
                    mask = mask > 0.0
                video_segments[frame_idx][object_id + 1] = mask
        
    # Sort by confidence descending so higher confidence objects get priority
    # if output_type == "numpy":
    # NumPy version (original)
    confidences_np = confidences.cpu().numpy()
    sorted_indices = np.argsort(confidences_np)[::-1]
    label_ids_sorted = np.array(label_ids)[sorted_indices]
    
    # Create semantic maps for all frames
    sem2d = np.full((len(sam2_feats_list), H, W), fill_value=empty_class, dtype=np.uint8)
    
    # Process each frame
    for frame_idx in range(len(sam2_feats_list)):
        if frame_idx not in video_segments:
            continue
            
        # Process objects in order of confidence (sorted_indices gives us the order)
        for sorted_idx, label_id in enumerate(label_ids_sorted):
            if label_id in ignore_ids:
                continue
            
            # Object IDs start from 1, and correspond to the original order before sorting
            # We need to map back: sorted_idx -> original_idx -> object_id
            original_idx = sorted_indices[sorted_idx]
            object_id = original_idx + 1  # object_id starts from 1
            
            # For frames other than 0, only use masks with confidence > box_conf_thres
            if frame_idx != 0:
                if confidences[original_idx] <= box_conf_thres:
                    continue
            
            if object_id in video_segments[frame_idx]:
                mask = video_segments[frame_idx][object_id].squeeze()
                # Use logical AND with inverse of existing mask to prevent overwrites
                sem2d[frame_idx] = np.where(mask & (sem2d[frame_idx] == empty_class), label_id, sem2d[frame_idx])
    
    # Clean up video segments to free memory
    del video_segments
    
    return sem2d



def infer_semantic(
    gdino_imgs,
    sam2_imgs,
    semantic_txt,
    class_names,
    device,
    box_threshold=0.1,
    text_threshold=0.0,
    image_size=512,
    sam2_feats=None,
):
    assert gdino_imgs.shape[0] == 1, "Only support batch size of 1"
    assert sam2_imgs.shape[0] == 1, "Only support batch size of 1"
    assert len(gdino_imgs.shape) == 4, "B, C, H, W"
    assert len(sam2_imgs.shape) == 4, "B, C, H, W"
    
    # Initialize model manager
    model_manager = ModelManager(device)
    
    # Parse semantic text format
    feat_src, sam2_model_type = semantic_txt.split('@')
    
    # Get models from manager
    sam2_model = model_manager.get_sam2(sam2_model_type, image_size=image_size)

    
    # kitti_class_names = class_mapping.id_2_kitti_classes
    class2idx = {name: i for i, name in enumerate(class_names)}
    H, W = gdino_imgs.shape[2:]

    boxes, confidences, labels = infer_sam2_boxes(
        gdino_imgs=gdino_imgs,
        class_names=class_names,
        device=device,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # Check if there are any boxes detected
    if len(boxes) == 0:
        # Return empty results when no boxes are detected
        sem2d = np.zeros((H, W), dtype=np.uint8)
        return sem2d, boxes, confidences, labels
    
    # process the box prompt for SAM 2
    boxes_scaled = torch.from_numpy(boxes) * torch.Tensor([W, H, W, H])
    input_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    # feat_src, _ = semantic.split('@')
    if feat_src == 'pretrained':
        # img_pil = Image.open(img_path).convert("RGB")
        # img_pil = img_pil.resize((W, H))
        # sam2_model.predictor.set_image(sam2_imgs[0])
        # features = sam2_model.predictor.get_image_features(sam2_imgs[0])
        # features = sam2_model.predictor._features
        image_embed, feat_s1, feat_s0 = sam2_model.forward(sam2_imgs)
        high_res_feats = [feat_s0, feat_s1]
        
        sam2_model.set_image_features(
            high_res_feats=high_res_feats,
            image_embed=image_embed,
            img_hws=[[H, W]]
        )
    elif feat_src == 'distill':
        if sam2_feats is None:
            raise ValueError("sam2_feats must be provided for distill SAM2 semantic inference")
        sam2_model.set_image_features(
            high_res_feats=sam2_feats['high_res_feats'],
            image_embed=sam2_feats['image_embed'],
            img_hws=[[H, W]]
        )
    else:
        raise ValueError(f"Invalid feature source: {feat_src}")
    
    masks = sam2_model.predict_masks(boxes=input_boxes)
    
    # Sort masks by confidence descending so higher confidence masks get priority
    confidences = np.asarray(confidences)
    sorted_indices = np.argsort(confidences)[::-1]
    masks = masks[sorted_indices]
    class_ids = [class2idx[name] for name in labels]
    class_ids = np.array(class_ids)[sorted_indices]
    
    # common_class_ids = class_mapping.map_kitti_id_2_common_id(kitti_class_ids)
    
    # Create empty semantic map and apply masks
    sem2d = np.zeros((H, W))
    for class_id, mask in zip(class_ids, masks):
        if class_id == 255:
            continue
        # Use logical AND with inverse of existing mask to prevent overwrites
        sem2d = np.where(mask.astype(bool) & (sem2d == 0), class_id, sem2d)
    sem2d = sem2d.astype(np.uint8)
    # sem2d = np.expand_dims(sem2d, axis=0)  # Add channel dimension to get shape [1, H, W]
    return sem2d, boxes, confidences, labels
