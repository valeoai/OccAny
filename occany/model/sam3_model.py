try:
    from sam3.model.sam3_image_processor import Sam3Processor
except ImportError:
    print("Assuming sam3 is in third_party")
    import sys
    sys.path.append("third_party/sam3")
    from sam3.model.sam3_image_processor import Sam3Processor

from sam3.model import box_ops
import logging
import torch
from sam3 import build_sam3_image_model
from sam3.model.data_misc import FindStage, interpolate
from typing import Dict, List, Union


class Sam3ProcessorWrapper(Sam3Processor):
    """
    A wrapper for the Sam3Processor that allows for overriding and extending its functionality.
    """
    def __init__(self, model, resolution=1008, device="cuda", confidence_threshold=0.5):
        super().__init__(model, resolution, device, confidence_threshold)

    @torch.inference_mode()
    def forward(self, input_image, original_height, original_width):
        """
        Forward pass to extract image embeddings from SAM3 backbone.
        
        Args:
            input_image: Tensor of shape [B, 3, H, W] (already preprocessed)
            max_bs: Maximum batch size (unused, kept for API compatibility)
            
        Returns:
            image_embed: Final image embedding [B, C, H, W]
            feat_s1: High-res feature level 1 [B, C, H, W]
            feat_s0: High-res feature level 0 [B, C, H, W]
        """
        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size BxCxHxW, got {input_image.shape}"
        
        logging.info("Computing image embeddings for the provided image...")
        
        # backbone_out contains:
        # - "backbone_fpn": list of multi-scale feature maps [feat_s0, feat_s1, feat_s2]
        #   shapes: [B, 256, 288, 288], [B, 256, 144, 144], [B, 256, 72, 72]
        # - "vision_features": final visual feature map [B, 256, 72, 72] identical to backbone_fpn[2]
        backbone_out = self.model.backbone.forward_image(input_image)
        assert torch.equal(backbone_out["vision_features"], backbone_out["backbone_fpn"][2]), "vision_features and backbone_fpn[2] should be the same"
        
        state = {
            "backbone_out": backbone_out,
            "original_height": original_height,
            "original_width": original_width
        }
        
        return state

    @torch.no_grad()
    def forward_distill(self, input_image):
        """
        Forward pass for distillation that mirrors `SAM3Head` output ordering.

        Args:
            input_image: Tensor of shape [B, 3, H, W] (already preprocessed)
            
        Returns:
            Tuple(feat_s0, feat_s1, feat_s2, pre_neck_feat):
                - feat_s0, feat_s1, feat_s2: multi-scale features from the neck
                - pre_neck_feat: [B, 1024, H, W] ViT backbone output before neck convs
        """
        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size BxCxHxW, got {input_image.shape}"
        
        logging.info("Computing image embeddings for the provided image...")
        
        # Run the vision backbone only once, extracting both trunk output and neck output
        vision_backbone = self.model.backbone.vision_backbone
        
        # Step 1: Run the trunk (ViT) to get pre-neck features
        trunk_out = vision_backbone.trunk(input_image)
        pre_neck_feat = trunk_out[-1]  # [B, 1024, H, W] - the ViT output before neck convs
        
        # Step 2: Run the neck convs on the trunk output (same as neck.forward but without re-running trunk)
        sam3_out = []
        for conv in vision_backbone.convs:
            sam3_x_out = conv(pre_neck_feat)
            sam3_out.append(sam3_x_out)
        
        # Apply scalp (same as SAM3VLBackbone with scalp=1)
        scalp = self.model.backbone.scalp
        if scalp > 0:
            sam3_out = sam3_out[:-scalp]
        
        feat_s0, feat_s1, feat_s2 = sam3_out
        
        # Return (feat_s0, feat_s1, feat_s2, pre_neck_feat) to match SAM3Head output
        # All 4 features are returned as a tuple for distillation
        return (feat_s0, feat_s1, feat_s2, pre_neck_feat)

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict, image_id=0):
        """Sets the text prompt and run the inference"""

        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        # Ensure geometric_prompt has the correct batch size
        if "geometric_prompt" not in state or \
           (getattr(state["geometric_prompt"], "box_embeddings", None) is not None and \
            state["geometric_prompt"].box_embeddings.shape[1] != 1):
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        find_stage = FindStage(
            img_ids=torch.tensor([image_id], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        return self._forward_grounding(state, find_stage)

    @torch.inference_mode()
    def predict_batched(self, prompts: List[str], state: Dict, image_ids: Union[int, List[int]] = 0):
        """Predict masks for multiple text prompts in a single forward pass."""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before predict_batched")

        # Normalize image_ids to list
        if isinstance(image_ids, int):
            image_ids = [image_ids]

        # Get text embeddings for all prompts at once
        text_outputs = self.model.backbone.forward_text(prompts, device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        
        num_prompts = len(prompts)
        num_images = len(image_ids)
        num_tasks = num_prompts * num_images

        # Ensure geometric_prompt has the correct batch size for the number of tasks
        if "geometric_prompt" not in state or \
           (getattr(state["geometric_prompt"], "box_embeddings", None) is not None and \
            state["geometric_prompt"].box_embeddings.shape[1] != num_tasks):
            state["geometric_prompt"] = self.model._get_dummy_prompt(num_prompts=num_tasks)

        # Create find_stage with text_ids for each prompt
        # We want to run every prompt on every image.
        # img_ids: [img0, img0, ..., img0, img1, img1, ..., img1] (each repeated P times)
        # text_ids: [0, 1, ..., P-1, 0, 1, ..., P-1] (repeated V times)
        img_ids_tensor = torch.tensor(image_ids, device=self.device, dtype=torch.long)
        img_ids = torch.repeat_interleave(img_ids_tensor, num_prompts)
        
        text_ids_base = torch.arange(num_prompts, device=self.device, dtype=torch.long)
        text_ids = text_ids_base.repeat(num_images)
        
        find_stage = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        return self._forward_grounding(state, find_stage)

    @torch.inference_mode()
    def _forward_grounding(self, state: Dict, find_stage: FindStage):
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        # Detect if we have multiple classes (batched prompts)
        # out_probs shape: [N_tasks, N_queries]
        if len(out_probs.shape) == 2 and out_probs.shape[1] > 1:
            # Batched case
            keep = out_probs > self.confidence_threshold
            
            if not keep.any():
                # No predictions above threshold
                return {
                    "masks_logits": [],
                    "boxes": [],
                    "scores": [],
                    "masks": [],
                    "prompt_indices": []
                }
            
            # Extract kept predictions using boolean indexing
            out_probs_kept = out_probs[keep]
            out_masks_kept = out_masks[keep]
            out_bbox_kept = out_bbox[keep]
            # prompt_indices corresponds to the task index for each kept prediction
            prompt_indices = torch.nonzero(keep, as_tuple=True)[0]
            
            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox_kept)

            img_h = state["original_height"]
            img_w = state["original_width"]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
            boxes = boxes * scale_fct[None, :]

            out_masks_kept = interpolate(
                out_masks_kept.unsqueeze(1),
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            
            return {
                "masks_logits": out_masks_kept,
                "boxes": boxes,
                "scores": out_probs_kept,
                "masks": out_masks_kept > 0.5,
                "prompt_indices": prompt_indices
            }
        else:
            # Single prompt case (backward compatibility)
            keep = out_probs > self.confidence_threshold
            out_probs = out_probs[keep]
            out_masks = out_masks[keep]
            out_bbox = out_bbox[keep]

            # convert to [x0, y0, x1, y1] format
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

            img_h = state["original_height"]
            img_w = state["original_width"]
            scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
            boxes = boxes * scale_fct[None, :]

            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (img_h, img_w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()
            return {
                "masks_logits": out_masks,
                "boxes": boxes,
                "scores": out_probs,
                "masks": out_masks > 0.5,
                "prompt_indices": torch.zeros(len(out_probs), dtype=torch.long, device=self.device)
            }


class Sam3ModelManager:
    _instance = None

    def __new__(cls, resolution=1008, confidence_threshold=0.5):
        if cls._instance is None:
            cls._instance = super(Sam3ModelManager, cls).__new__(cls)
            cls._instance._initialize(resolution, confidence_threshold)
        return cls._instance

    def _initialize(self, resolution, confidence_threshold):
        self.resolution = resolution
        self.confidence_threshold = confidence_threshold
        self._sam3 = None

    def get_sam3(self, device="cuda"):
        if self._sam3 is None:
            logging.info("Loading SAM3 image model...")
           
            model = build_sam3_image_model(eval_mode=True)
            model.to(device)
            for param in model.parameters():
                param.requires_grad_(False)
            self._sam3 = Sam3ProcessorWrapper(
                model=model,
                resolution=self.resolution,
                device=device,
                confidence_threshold=self.confidence_threshold,
            )
        return self._sam3