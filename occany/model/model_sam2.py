# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# DUSt3R model class
# --------------------------------------------------------
from copy import deepcopy
from collections import OrderedDict
import torch
from packaging import version
import huggingface_hub
import torch.nn as nn
from dust3r.utils.misc import transpose_to_landscape
from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from croco.models.blocks import Mlp
from croco.models.pos_embed import RoPE1D
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


from typing import Union
from PIL import Image
import numpy as np


import logging
inf = float('inf')

hf_version_number = huggingface_hub.__version__
assert version.parse(hf_version_number) >= version.parse("0.22.0"), "Outdated huggingface_hub version, please reinstall requirements.txt"

def load_model(model_path, device, verbose=True):
    if verbose:
        print('... loading model from', model_path)
    ckpt = torch.load(model_path, map_location='cpu')
    args = ckpt['args'].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
    if 'landscape_only' not in args:
        args = args[:-1] + ', landscape_only=False)'
    else:
        args = args.replace(" ", "").replace('landscape_only=True', 'landscape_only=False')
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt['model'], strict=False)
    if verbose:
        print(s)
    return net.to(device)




class SAM2(nn.Module):
    def __init__(self, 
                #  model_id,
                 model_cfg,
                 sam2_checkpoint,
                 device, 
                 load_video_model=False,
                 image_size=512):
        super().__init__()
        self.load_video_model = load_video_model
        self.image_predictor = self.get_sam2_image_predictor(model_cfg, sam2_checkpoint, device, image_size=image_size)
        if load_video_model:
            self.video_predictor = self.get_sam2_video_predictor(model_cfg, sam2_checkpoint, device, image_size=image_size)
        
    def get_sam2_image_predictor(self, 
                                 config_name, ckpt_path, 
                                 device, 
                                 apply_postprocessing=True, 
                                 hydra_overrides_extra=[],
                                 use_high_res_features_in_sam=True, 
                                 image_size=512):
        # config_name, ckpt_path = _hf_download(model_id)

        
        hydra_overrides = [
            f"++model.image_size={image_size}",
        ]
        
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
        
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.use_high_res_features_in_sam={}".format(use_high_res_features_in_sam),
            ]
        hydra_overrides.extend(hydra_overrides_extra)
        # Read config and init model
        cfg = compose(config_name=config_name, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path)

        model = model.to(device)
        model.eval()
        self.model = model
        return SAM2ImagePredictorOccany(self.model)

    def get_sam2_video_predictor(self, config_name, ckpt_path, device, 
                                 hydra_overrides_extra=[],
                                 apply_postprocessing=True, image_size=512):
        # config_name, ckpt_path = _hf_download(model_id)\
            
        
        hydra_overrides = [
            "++model._target_=occany.model.model_sam2.SAM2VideoPredictorOccany",
            f"++model.image_size={image_size}",
        ]
        if apply_postprocessing:
            hydra_overrides_extra = hydra_overrides_extra.copy()
            hydra_overrides_extra += [
                # dynamically fall back to multi-mask if the single mask is not stable
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
                "++model.fill_hole_area=8",
            ]
        hydra_overrides.extend(hydra_overrides_extra)

        # Read config and init model
        cfg = compose(config_name=config_name, overrides=hydra_overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)
        _load_checkpoint(model, ckpt_path)
        model = model.to(device)
        model.eval()
        return model
    
    def set_image_features(self, high_res_feats, image_embed, img_hws):
        self.image_predictor._orig_hw = img_hws
        self.image_predictor._is_image_set = True
        self.image_predictor._is_batch = True
        self.image_predictor._features = {
            "high_res_feats": high_res_feats,
            "image_embed": image_embed
        }

    def predict_masks(self, point_coords=None, point_labels=None, boxes=None):
        assert point_coords is not None or boxes is not None, "either point_coords or boxes must be provided"
        masks, scores, logits = self.image_predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            box=boxes,
                            multimask_output=False,
                        )
        # convert the shape to (n, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        return masks

    def forward(self, input_image, max_bs=8):
        # image: 370, 1226, 3
        # input_image = self._transforms(image)
        # input_image: 3, 1024, 1024

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        bs = input_image.shape[0]

        if max_bs is None or max_bs <= 0 or bs <= max_bs:
            return self._forward_impl(input_image)

        image_embeds = []
        feat_s1_list = []
        feat_s0_list = []
        for start in range(0, bs, max_bs):
            chunk = input_image[start:start + max_bs]
            image_embed, feat_s1, feat_s0 = self._forward_impl(chunk)
            image_embeds.append(image_embed)
            feat_s1_list.append(feat_s1)
            feat_s0_list.append(feat_s0)

        return (
            torch.cat(image_embeds, dim=0),
            torch.cat(feat_s1_list, dim=0),
            torch.cat(feat_s0_list, dim=0),
        )

    def _forward_impl(self, input_image):
        backbone_out = self.model.forward_image(input_image)

        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        bs = input_image.shape[0]

        feats = [
            feat.permute(1, 2, 0).view(bs, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self.image_predictor._bb_feat_sizes[::-1])
        ][::-1]
        image_embed = feats[-1]
        feat_s1 = feats[-2]
        feat_s0 = feats[-3]

        return image_embed, feat_s1, feat_s0



def _load_checkpoint(model, ckpt_path):
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(sd)
        if missing_keys:
            logging.error(missing_keys)
            raise RuntimeError()
        if unexpected_keys:
            logging.error(unexpected_keys)
            raise RuntimeError()
        logging.info("Loaded checkpoint sucessfully")
        
def build_sam2_video_predictor(
    config_file,
    ckpt_path=None,
    device="cuda",
    mode="eval",
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    video_predictor="SAM2VideoPredictor",
    **kwargs,
):
    hydra_overrides = [
        f"++model._target_=sam2.sam2_video_predictor.{video_predictor}",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides.extend(hydra_overrides_extra)

    # Read config and init model
    cfg = compose(config_name=config_file, overrides=hydra_overrides)
    OmegaConf.resolve(cfg)
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, ckpt_path)
    model = model.to(device)
    if mode == "eval":
        model.eval()
    return model




class SAM2VideoPredictorOccany(SAM2VideoPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.inference_mode()
    def init_state(self, image_feats=None,
                   video_path=None,
                   video_height=None,
                   video_width=None,
                   offload_video_to_cpu=False,
                   offload_state_to_cpu=False,
                   async_loading_frames=False):
        compute_device = self.device
        inference_state = {}
        if video_path is not None:
            images, video_height, video_width = load_video_frames(
                video_path=video_path,
                image_size=self.image_size,
                offload_video_to_cpu=offload_video_to_cpu,
                async_loading_frames=async_loading_frames,
                compute_device=compute_device,
            )
            inference_state["images"] = images
            inference_state["num_frames"] = len(images)
        if image_feats is not None:
            inference_state["image_feats"] = image_feats
            inference_state["num_frames"] = len(image_feats)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        return inference_state


    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Compute the image features on a given frame."""
        # if "images" in inference_state:
        #     # Retrieve correct image features
        #     (
        #         _,
        #         _,
        #         current_vision_feats,
        #         current_vision_pos_embeds,
        #         feat_sizes,
        #     ) = self._get_image_feature(inference_state, frame_idx, batch_size)
        if "image_feats" in inference_state:
            my_vision_feat_dict = inference_state["image_feats"][frame_idx]

            # current_vision_feats[0].shape
            # torch.Size([65536, 1, 32])
            # current_vision_feats[1].shape
            # torch.Size([16384, 1, 64])
            # current_vision_feats[2].shape
            # torch.Size([4096, 1, 256])
            my_vision_feats = [
                my_vision_feat_dict['high_res_feats'][0], 
                my_vision_feat_dict['high_res_feats'][1], 
                my_vision_feat_dict['image_embed']
            ]
        
            feat_sizes = [(x.shape[-2], x.shape[-1]) for x in my_vision_feats]
            
            # Expand features to match batch_size (number of objects being tracked)
            my_vision_feats = [x.expand(batch_size, -1, -1, -1) for x in my_vision_feats]
            
            # Compute and expand positional embeddings to match batch_size
            my_vision_pos_embeds = [self.image_encoder.neck.position_encoding(x) for x in my_vision_feats]
            my_vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in my_vision_pos_embeds]

            my_vision_feats = [x.flatten(2).permute(2, 0, 1) for x in my_vision_feats]
            if self.directly_add_no_mem_embed:
                my_vision_feats[2] = my_vision_feats[2] - self.no_mem_embed
        
            current_vision_feats = my_vision_feats
            current_vision_pos_embeds = my_vision_pos_embeds

            return None, None, current_vision_feats, current_vision_pos_embeds, feat_sizes

        # Look up in the cache first
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Cache miss -- we will run inference on a single image
            device = inference_state["device"]
            image = inference_state["images"][frame_idx].to(device).float().unsqueeze(0)
           
            backbone_out = self.forward_image(image)
            # Cache the most recent frame's feature (for repeated interactions with
            # a frame; we can use an LRU cache for more frames in the future).
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                batch_size, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
        run_mem_encoder=True,
    ):
        """Propagate the input points across frames to track in the entire video.
        
        Args:
            run_mem_encoder: Whether to encode predicted masks into memory for future frames.
                            Set to False for read-only inference without updating the memory bank.
        """
        from tqdm import tqdm
        
        self.propagate_in_video_preflight(inference_state)

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(output_dict["cond_frame_outputs"])
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
            if start_frame_idx > 0:
                processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
            else:
                processing_order = []  # skip reverse tracking if starting from frame 0
        else:
            end_frame_idx = min(
                start_frame_idx + max_frame_num_to_track, num_frames - 1
            )
            processing_order = range(start_frame_idx, end_frame_idx + 1)
            
        for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=run_mem_encoder,
                )
                output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            yield frame_idx, obj_ids, video_res_masks


class SAM2ImagePredictorOccany(SAM2ImagePredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hires_size = self.model.image_size // 4
        self._bb_feat_sizes = [[hires_size // (2**k)]*2 for k in range(3)]
  
        
    
    @torch.no_grad()
    def get_image_features_with_pos_enc(
        self,
        image: Union[np.ndarray, Image],
    ) -> None:
        """
        Calculates the image embeddings for the provided image, allowing
        masks to be predicted with the 'predict' method.

        Arguments:
          image (np.ndarray or PIL Image): The input image to embed in RGB format. The image should be in HWC format if np.ndarray, or WHC format if PIL Image
          with pixel values in [0, 255].
          image_format (str): The color format of the image, in ['RGB', 'BGR'].
        """
        self.reset_predictor()
        # Transform the image to the form expected by the model
        if isinstance(image, np.ndarray):
            logging.info("For numpy array image, we assume (HxWxC) format")
            self._orig_hw = [image.shape[:2]]
        elif isinstance(image, Image):
            w, h = image.size
            self._orig_hw = [(h, w)]
        else:
            raise NotImplementedError("Image format not supported")

        # image: 370, 1226, 3
        input_image = self._transforms(image)
        # input_image: 3, 1024, 1024
        input_image = input_image[None, ...].to(self.device)

        assert (
            len(input_image.shape) == 4 and input_image.shape[1] == 3
        ), f"input_image must be of size 1x3xHxW, got {input_image.shape}"
        logging.info("Computing image embeddings for the provided image...")
        backbone_out = self.model.forward_image(input_image)
        _, vision_feats, vision_pos_embeds, _ = self.model._prepare_backbone_features(backbone_out)
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        
        
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1], "vision_pos_embeds": vision_pos_embeds}
