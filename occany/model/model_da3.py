from einops import rearrange

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import os
import numpy as np
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse
from depth_anything_3.utils.ray_utils import get_extrinsic_from_camray
from depth_anything_3.model.utils.transform import pose_encoding_to_extri_intri
from dust3r.utils.geometry import geotrf
from occany.utils.helpers import convert_depth_to_point_cloud, intrinsics_c2w_to_raymap
from occany.model.must3r_blocks.head import SAM3Head
from occany.model.raymap_encoder_da3 import RaymapEncoderDA3
from occany.da3_inference import create_gen_conditioning, concat_preds
from tqdm import tqdm


logger = logging.getLogger(__name__)


class DA3Wrapper(DepthAnything3):
    def __init__(self, img_size=518, projection_features='pts3d_local,pts3d,rgb,conf,sam3', **kwargs):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.projection_features = projection_features
        self.head_sam = None  # Will be initialized by init_sam3_head if SAM3 distillation is enabled
        
        # Gen encoder is lazily initialized
        self.gen_input_encoder = None
        self.slice_layer_idx = None  # Layer index after which recon tokens are sliced in gen mode
        
        # Remove unused cam_enc and cam_dec modules to save memory
        self._remove_unused_modules()

    def _remove_unused_modules(self):
        """Remove unused cam_enc and cam_dec modules to save ~59M parameters."""
        if hasattr(self.model, 'cam_enc') and self.model.cam_enc is not None:
            del self.model.cam_enc
            self.model.cam_enc = None
        if hasattr(self.model, 'cam_dec') and self.model.cam_dec is not None:
            del self.model.cam_dec
            self.model.cam_dec = None

    def init_gen_encoders(self):
        """Initialize raymap encoder for generation mode."""
        device = self._get_model_device()
        self.gen_input_encoder = RaymapEncoderDA3(
            img_size=(self.img_size, self.img_size), patch_size=14, embed_dim=1024,
            depth=0, num_heads=16, projection_features=self.projection_features,
        ).to(device)
        return self

    def set_slice_layer(self, layer_idx):
        """Set layer index where recon-token slicing is applied in gen mode."""
        self.slice_layer_idx = layer_idx
        print(f"Set slice_layer_idx to {layer_idx}")
        return self

    def set_alt_start(self, alt_start: int):
        """Change alt_start for backbone (camera_token already exists)."""
        self.model.backbone.pretrained.alt_start = alt_start
        return self

    def init_sam3_head(self, img_size=518, embed_dim=256, patch_size=14, device=None, use_dpt_proj=False):
        """
        Initialize SAM3Head for distillation.
        
        Args:
            img_size: Input image size (default: 518 for DA3)
            embed_dim: SAM3 feature embedding dimension (default: 256)
            patch_size: Patch size for DinoV2 backbone (default: 14)
            device: Device to place the head on
            use_dpt_proj: Use DPT-style projection instead of Mlp (default: False)
            
        Returns:
            self for method chaining
        """
        print('Initializing SAM3Head for distillation...')
        print(f'  - img_size: {img_size}')
        print(f'  - embed_dim: {embed_dim}')
        print(f'  - patch_size: {patch_size}')
        print(f'  - use_dpt_proj: {use_dpt_proj}')
        
        if use_dpt_proj:
            # DPT mode: 4 separate layer dimensions (each layer is 2048 with cat_token=True)
            input_dims = (2048, 2048, 2048, 2048)
            print(f'  - input_dims: {input_dims}')
            
            self.head_sam = SAM3Head(
                input_dim=None,        # Not used in DPT mode
                input_dims=input_dims,
                img_size=img_size,
                embed_dim=embed_dim,
                patch_size=patch_size,
                use_dpt_proj=True,
            )
        else:
            # Current behavior: concatenate all 4 layers
            # We use 4 layers concatenated (11, 15, 19, 23)
            backbone_embed_dim = 2048 * 4
            print(f'  - backbone_embed_dim: {backbone_embed_dim}')
            
            self.head_sam = SAM3Head(
                input_dim=backbone_embed_dim,
                img_size=img_size,
                embed_dim=embed_dim,
                patch_size=patch_size,
                use_dpt_proj=False,
            )
        
        if device is not None:
            self.head_sam = self.head_sam.to(device)
        
        return self
    
    def forward_sam_features(self, backbone_features, img_shape):
        """Forward pass through SAM3Head to get features for distillation."""
        if self.head_sam is None:
            raise RuntimeError("SAM3Head not initialized. Call init_sam3_head() first.")
        
        # Extract main features from (feat, cam_token) tuples
        feats = [f[0] if isinstance(f, (list, tuple)) else f for f in backbone_features]
        B, T, N, C = feats[0].shape
        
        if self.head_sam.use_dpt_proj:
            # DPT mode: pass features separately as list
            # Flatten B and T for each layer: (B, T, N, C) -> (BT, N, C)
            feats_flat = [f.flatten(0, 1) for f in feats]  # List of 4 tensors, each [BT, N, C]
            sam_outputs = self.head_sam(feats_flat, img_shape)
        else:
            # Current Mlp mode: concatenate all features
            concat_feat = torch.cat(feats, dim=-1)  # (B, T, N, C*4)
            # SAM3Head expects list of [BT, N, C]
            sam_outputs = self.head_sam([concat_feat.flatten(0, 1)], img_shape)
        
        # Restore view dimension: (BT, ...) -> (B, T, ...)
        return tuple(out.reshape(B, T, *out.shape[1:]) for out in sam_outputs)

    
    def apply_lora(self, lora_r=8, lora_alpha=8, lora_dropout=0.1, lora_target_modules='qkv', fine_tune_layers=None):
        """
        Apply LoRA (Low-Rank Adaptation) to the DinoV2 backbone.
        
        Args:
            lora_r: LoRA rank (dimension of the low-rank matrices)
            lora_alpha: LoRA alpha (scaling factor)
            lora_dropout: LoRA dropout rate
            lora_target_modules: Comma-separated string or list of target module names
            fine_tune_layers: Optional list of layer indices (0-23) to fully fine-tune 
                              instead of using LoRA. If None, all layers use LoRA.
        
        Returns:
            self for method chaining
        """
        from peft import LoraConfig, get_peft_model
        
        # Parse target modules from comma-separated string if needed
        if isinstance(lora_target_modules, str):
            target_modules = [m.strip() for m in lora_target_modules.split(',')]
        else:
            target_modules = list(lora_target_modules)
        
        print('Applying LoRA adapters to backbone...')
        print(f'  - LoRA rank (r): {lora_r}')
        print(f'  - LoRA alpha: {lora_alpha}')
        print(f'  - LoRA dropout: {lora_dropout}')
        print(f'  - Target modules: {target_modules}')
        
        # DinoV2-L has 24 transformer blocks (depth=24)
        total_layers = 24
        lora_layers = None
        
        if fine_tune_layers is not None:
            fine_tune_layers = [int(i) for i in fine_tune_layers]
            # Layers for LoRA = all layers NOT in fine_tune_layers
            lora_layers = [i for i in range(total_layers) if i not in fine_tune_layers]
            print(f'  - Selective LoRA: layers {lora_layers} will use LoRA, layers {fine_tune_layers} will be fully fine-tuned.')
        else:
            print(f'  - LoRA will be applied to all {total_layers} layers.')
            fine_tune_layers = []

        # Create LoRA configuration for the DinoV2 backbone
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            layers_to_transform=lora_layers,
            bias="none",
        )
        
        # Apply LoRA to the backbone
        self.model.backbone = get_peft_model(self.model.backbone, lora_config)
        self.model.backbone.print_trainable_parameters()
        
        # NOTE: Head trainability is now controlled by --freeze_head flag in training script
        # Do NOT unconditionally unfreeze the head here
        
        # If specific layers were withheld from LoRA, unfreeze them for full fine-tuning
        if fine_tune_layers:
            print(f'Unfreezing layers {fine_tune_layers} for full fine-tuning...')
            # DinoV2 blocks are in self.model.backbone.base_model.model.pretrained.blocks
            # After PEFT wrapping, the hierarchy is: peft_model.base_model.model
            for layer_idx in fine_tune_layers:
                block = self.model.backbone.base_model.model.pretrained.blocks[layer_idx]
                for param in block.parameters():
                    param.requires_grad = True
        
        print('LoRA adapters applied to backbone. Specified layers and DualDPT head remain trainable.')
        return self
    
    def init_aux_branch(self, n_layers=6):
        """
        Initialize aux branch by duplicating the last n layers.
        The aux branch is frozen and takes input from layer (depth - n - 1).
        
        Args:
            n_layers: Number of layers to duplicate (default: 6, i.e., layers 18-23)
        
        Returns:
            self for method chaining
        """
        import copy
        self.aux_n_layers = n_layers
        self.aux_input_layer_idx = 24 - n_layers - 1  # Layer before last n (e.g., 17 for n=6)
        
        # Get the pretrained backbone blocks
        backbone_blocks = self.model.backbone.pretrained.blocks
        
        # Duplicate last n layers
        self.aux_blocks = nn.ModuleList([
            copy.deepcopy(backbone_blocks[24 - n_layers + i])
            for i in range(n_layers)
        ])
        
        # Duplicate the head for aux predictions
        self.aux_head = copy.deepcopy(self.model.head)
        
        # Freeze all aux parameters
        for param in self.aux_blocks.parameters():
            param.requires_grad = False
        for param in self.aux_head.parameters():
            param.requires_grad = False
        
        print(f'Initialized aux branch with {n_layers} frozen layers (input from layer {self.aux_input_layer_idx})')
        return self
    
    def inference_batch_aux(self, aux_tokens, h, w, device_type, main_feats=None):
        """
        Run aux branch inference using intermediate tokens.
        Mimics vision_transformer.py _get_intermediate_layers_not_chunked processing.
        
        The aux branch processes tokens starting from aux_input_layer_idx.
        It collects outputs at specific layer indices matching the main backbone's out_layers.
        
        For DA3-LARGE with out_layers=[11, 15, 19, 23] and aux_branch starting at layer 17 (n_layers=6):
        - Layers 11, 15 are BEFORE aux branch - passed via main_feats from main backbone
        - Layer 19 = aux block index 1 (17+1+1=19)
        - Layer 23 = aux block index 5 (17+1+5=23)
        
        Args:
            aux_tokens: (B, S, N_total, C) raw intermediate tokens from backbone (at aux_input_layer_idx)
                        NOTE: These tokens INCLUDE cls/register tokens (raw export from backbone)
            h, w: image height and width
            device_type: device type for autocast
            main_feats: Features from main backbone for layers BEFORE aux branch (required)
                        List of (feat, cam_token) tuples for out_layers before aux_input_layer_idx
            
        Returns:
            Dict with aux branch predictions (pointmap, depth, etc.)
        """
        if not hasattr(self, 'aux_blocks'):
            raise RuntimeError("Aux branch not initialized. Call init_aux_branch() first.")
        
        # Get backbone parameters for compatibility
        backbone = self.model.backbone.pretrained
        
        # aux_tokens is now a tuple (x, local_x) from raw backbone export
        # Both include cls/register tokens
        # Shape: (B, S, 1 + num_register_tokens + N_patches, C)
        x, local_x = aux_tokens
        B, S, N_total, C = x.shape
        
        # Prepare RoPE positional encodings
        pos, pos_nodiff = backbone._prepare_rope(B, S, h, w, x.device)
        
        # Get out_layers from backbone config (e.g., [11, 15, 19, 23])
        out_layers = self.model.backbone.out_layers
        
        # Collect outputs at specific layer indices
        # For layers within aux block range, collect actual outputs
        # For layers before aux range, we'll handle differently
        aux_outputs = []
        
        for i, blk in enumerate(self.aux_blocks):
            layer_idx = self.aux_input_layer_idx + 1 + i  # Actual layer index
            
            # Determine positional encoding
            if layer_idx < backbone.rope_start or backbone.rope is None:
                g_pos, l_pos = None, None
            else:
                g_pos = pos_nodiff
                l_pos = pos
            
            # Apply attention pattern
            if backbone.alt_start != -1 and layer_idx >= backbone.alt_start and layer_idx % 2 == 1:
                x = self._process_attention_aux(x, blk, "global", pos=g_pos)
            else:
                x = self._process_attention_aux(x, blk, "local", pos=l_pos)
                local_x = x
            
            # Collect output if this layer is in out_layers
            if layer_idx in out_layers:
                out_x = torch.cat([local_x, x], dim=-1) if backbone.cat_token else x
                # Extract patch tokens only (remove cls and register)
                patch_x = out_x[..., 1 + backbone.num_register_tokens:, :]
                aux_outputs.append((patch_x, x[:, :, 0]))  # (features, camera_token)
        
        # For out_layers before aux range, use main_feats from the main backbone
        # For DA3-LARGE out_layers=[11, 15, 19, 23] with aux starting at 17:
        # - Layer 11, 15 are before aux, layer 19, 23 are in aux
        out_layers_before_aux = [l for l in out_layers if l <= self.aux_input_layer_idx]
        
        # Track how many outputs came from aux branch (need normalization)
        num_aux_branch_outputs = len(aux_outputs)
        
        # Prepend main_feats for layers before aux branch
        if main_feats is not None:
            # main_feats should have features for each layer in out_layers_before_aux
            assert len(main_feats) >= len(out_layers_before_aux), \
                f"main_feats has {len(main_feats)} items but need {len(out_layers_before_aux)} for layers {out_layers_before_aux}"
            # Insert main_feats at the beginning in order
            for i in range(len(out_layers_before_aux) - 1, -1, -1):
                aux_outputs.insert(0, main_feats[i])
        elif len(out_layers_before_aux) > 0:
            raise RuntimeError(
                f"main_feats required for layers {out_layers_before_aux} which are before aux branch (aux_input_layer_idx={self.aux_input_layer_idx})"
            )
        
        # Apply final layer norm ONLY to aux branch outputs (main_feats are already normalized)
        # main_feats are the first len(out_layers_before_aux) items, aux_outputs are the rest
        num_main_feats = len(out_layers_before_aux)
        feats = []
        for idx, (out_x, cam_token) in enumerate(aux_outputs):
            if idx < num_main_feats:
                # main_feats are already normalized from get_intermediate_layers
                feats.append((out_x, cam_token))
            else:
                # Aux branch outputs need normalization
                # out_x shape: (B, S, N, C) or (B, S, N, 2C) with cat_token
                if backbone.cat_token:
                    # Split, norm the second half, recombine
                    first_half = out_x[..., :backbone.embed_dim]
                    second_half = backbone.norm(out_x[..., backbone.embed_dim:])
                    normed = torch.cat([first_half, second_half], dim=-1)
                else:
                    normed = backbone.norm(out_x)
                feats.append((normed, cam_token))
        
        return self._process_depth_output(
            feats=feats,
            h=h, w=w,
            device_type=device_type,
            pose_from_depth_ray=False,
            pose_from_cam_dec=False,
            point_from_depth_and_pose=False,
            head=self.aux_head,  # Use frozen aux_head
        )
    
    def _process_attention_aux(self, x, block, attn_type="global", pos=None, attn_mask=None):
        """
        Process attention in aux blocks. Mimics vision_transformer.py process_attention.
        """
        b, s, n = x.shape[:3]
        
        if attn_type == "local":
            x = rearrange(x, "b s n c -> (b s) n c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> (b s) n c")
        elif attn_type == "global":
            x = rearrange(x, "b s n c -> b (s n) c")
            if pos is not None:
                pos = rearrange(pos, "b s n c -> b (s n) c")
        else:
            raise ValueError(f"Invalid attention type: {attn_type}")
        
        x = block(x, pos=pos, attn_mask=attn_mask)
        

        if attn_type == "local":
            x = rearrange(x, "(b s) n c -> b s n c", b=b, s=s)
        elif attn_type == "global":
            x = rearrange(x, "b (s n) c -> b s n c", b=b, s=s)
        
        return x

    def forward(self, images=None, recon_output=None, img_views=None, gen_views=None, **kwargs):
        if images is not None:
            # Reconstruction mode
            return self.inference_batch(images, **kwargs)
        elif recon_output is not None:
            # Generation mode
            return self.forward_gen(recon_output, img_views, gen_views, **kwargs)
        else:
            raise ValueError("Either 'images' or 'recon_output' must be provided")


    def forward_gen(
        self,
        recon_output,
        img_views,
        gen_views,
        projection_features=None,
        pose_from_depth_ray=False,
        point_from_depth_and_pose=False,
        return_loss_stats=False,
        # gen_batch_size=16,
        gen_batch_size=8,
        keep_sam_feats=True,
        keep_aux_feats=True,
        **kwargs
    ):
        if self.gen_input_encoder is None:
            raise RuntimeError("Gen encoder not initialized. Call init_gen_encoders() first.")

        if projection_features is None:
            projection_features = self.projection_features

        device = self._get_model_device()
        B = img_views[0]['img'].shape[0]
        nimgs = len(img_views)
        n_gen_views = len(gen_views)
        
        # Get image dimensions
        _, _, H, W = img_views[0]['img'].shape
        
        # Extract reconstruction outputs
        depth_conf = recon_output.get('depth_conf')
        pointmap = recon_output.get('pointmap')
        c2w = recon_output.get('c2w')
        intrinsics = recon_output.get('intrinsics')
        sam_feats = recon_output.get('sam_feats')
        
        # Build projection features
        projection_feature_list = [f.strip() for f in projection_features.split(',')]
        
        pts3d = pointmap.detach()  # (B, T, H, W, 3)
        conf = depth_conf.detach()  # (B, T, H, W)
        
        # Get RGB from views
        rgb = torch.stack([v['img'] for v in img_views], dim=1).to(device)
        rgb = rgb.permute(0, 1, 3, 4, 2)  # (B, T, H, W, 3)
        
        # Compute pts3d_local (in camera frame)
        # Use detached c2w to avoid gradient flow through frozen recon outputs
        if 'pts3d_local' in projection_feature_list:
            w2c = affine_inverse(c2w.detach())
            pts3d_flat = pts3d.reshape(B, nimgs, -1, 3)
            pts3d_local = geotrf(w2c, pts3d_flat)
            pts3d_local = pts3d_local.reshape(B, nimgs, H, W, 3)
        
        # Compute estimated focal from intrinsics (detach to avoid gradient flow)
        if intrinsics is not None:
            focal = intrinsics.detach()[:, :, 0, 0].mean(dim=1)  # (B,)
        else:
            focal = torch.ones(B, device=device) * 500.0  # Fallback
        
        # Build pts_features for projection
        feature_parts = []
        if 'pts3d_local' in projection_feature_list:
            feature_parts.append(pts3d_local)
        if 'pts3d' in projection_feature_list:
            feature_parts.append(pts3d)
        if 'rgb' in projection_feature_list:
            feature_parts.append(rgb)
        if 'conf' in projection_feature_list:
            feature_parts.append((conf.unsqueeze(-1) - 1.0))
        
        # Handle SAM3 features if enabled (3 scales, each 256 channels)
        if 'sam3' in projection_feature_list:
            if sam_feats is None:
                raise RuntimeError(
                    "projection_features includes 'sam3' but recon_output['sam_feats'] is None. "
                    "Initialize SAM3Head before loading the checkpoint, or remove 'sam3' "
                    "from projection_features for this run."
                )
            if len(sam_feats) < 3:
                raise RuntimeError(
                    f"projection_features includes 'sam3' but received only {len(sam_feats)} "
                    "SAM feature levels; expected at least 3."
                )

            for i in range(3):
                sam_feat = sam_feats[i]
                sam_feat_resized = F.interpolate(
                    sam_feat.reshape(B * nimgs, -1, sam_feat.shape[3], sam_feat.shape[4]),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                sam_feat_resized = sam_feat_resized.reshape(B, nimgs, -1, H, W)
                sam_feat_resized = sam_feat_resized.permute(0, 1, 3, 4, 2)
                feature_parts.append(sam_feat_resized)
        pts_features = torch.cat(feature_parts, dim=-1) if feature_parts else None
        del feature_parts  # Free memory
        
        # Get gen view camera poses
        gen_c2w = torch.stack([v['camera_pose'] for v in gen_views], dim=1).to(device)
        
        # Build gen_intrinsics
        cx, cy = W / 2.0, H / 2.0
        gen_intrinsics = torch.zeros((B, n_gen_views, 3, 3), device=device, dtype=focal.dtype)
        gen_intrinsics[:, :, 0, 0] = focal.unsqueeze(1)  # fx
        gen_intrinsics[:, :, 1, 1] = focal.unsqueeze(1)  # fy
        gen_intrinsics[:, :, 0, 2] = cx
        gen_intrinsics[:, :, 1, 2] = cy
        gen_intrinsics[:, :, 2, 2] = 1.0
        
        # Prepare context for gen views
        recon_cond_features = pts_features  # (B, nimgs, H, W, C)
        
        true_shape_recon = torch.stack([torch.tensor(v['true_shape']) for v in img_views], dim=1).to(device)
        recon_full_raymap = intrinsics_c2w_to_raymap(intrinsics[:, :1].detach(), c2w[:, :1].detach(), H, W, device=device)
        
        if gen_batch_size >= n_gen_views:
            gen_batch_size = n_gen_views

        gen_outputs = []
        gen_ranges = range(0, n_gen_views, gen_batch_size)
        if len(gen_ranges) > 1:
            gen_ranges = tqdm(gen_ranges, desc="Processing gen views")
        for i in gen_ranges:
            i_end = min(i + gen_batch_size, n_gen_views)
            gen_views_batch = gen_views[i:i_end]
            n_gen_views_batch = len(gen_views_batch)
            
            # 1. Compute raymap for this batch
            gen_intrinsics_batch = gen_intrinsics[:, i:i_end]
            gen_c2w_batch = gen_c2w[:, i:i_end]
            gen_raymap_batch = intrinsics_c2w_to_raymap(gen_intrinsics_batch, gen_c2w_batch, H, W, device=device)
            
            # 2. Create conditioning features via projection for current batch
            gen_cond_features_batch = create_gen_conditioning(
                pts3d, pts_features, focal, gen_c2w_batch,
                return_projected_pts3d=False,
                gen_views=gen_views_batch,
                projection_features=projection_features,
            )
            
            # 3. Concatenate recon + gen conditioning features
            cond_features_batch = torch.cat([recon_cond_features, gen_cond_features_batch], dim=1)
            n_total_views_batch = cond_features_batch.shape[1]
            
            # 4. Prepare for raymap encoder
            cond_features_flat_batch = cond_features_batch.permute(0, 1, 4, 2, 3)
            cond_features_flat_batch = cond_features_flat_batch.reshape(B * n_total_views_batch, -1, H, W)
            gen_input_dtype = next(self.gen_input_encoder.parameters()).dtype
            if cond_features_flat_batch.dtype != gen_input_dtype:
                cond_features_flat_batch = cond_features_flat_batch.to(gen_input_dtype)
            
            true_shape_gen_batch = torch.stack([torch.tensor(v['true_shape']) for v in gen_views_batch], dim=1).to(device)
            true_shape_batch = torch.cat([true_shape_recon, true_shape_gen_batch], dim=1)
            true_shape_flat_batch = true_shape_batch.reshape(B * n_total_views_batch, 2)
            
            # 5. Encode batch
            patch_tokens_batch, _ = self.gen_input_encoder(cond_features_flat_batch.detach(), true_shape_flat_batch)
            
            # Reshape
            N_patches, embed_dim = patch_tokens_batch.shape[1], patch_tokens_batch.shape[2]
            patch_tokens_batch = patch_tokens_batch.view(B, n_total_views_batch, N_patches, embed_dim)
            
            # 6. Full raymap for this batch
            full_raymap_batch = torch.cat([recon_full_raymap, gen_raymap_batch], dim=1)
            
             # 7. Run inference for this batch
            gen_output_batch = self.inference_batch_gen(
                patch_tokens_batch,
                H=H, W=W,
                gen_raymap=full_raymap_batch,
                n_gen_views=n_gen_views_batch,
                export_feat_layers=kwargs.get('export_feat_layers', None),
                return_aux_feats=keep_aux_feats,
            )

            if not keep_sam_feats:
                gen_output_batch.pop('sam_feats', None)
             
            # Add gen_c2w_batch to the output
            gen_output_batch['c2w'] = gen_c2w_batch
            
            gen_outputs.append(gen_output_batch)
            
            # Free batch memory
            del gen_cond_features_batch, cond_features_batch, cond_features_flat_batch
            del patch_tokens_batch, gen_raymap_batch, full_raymap_batch
            if i + gen_batch_size < n_gen_views:
                torch.cuda.empty_cache()
        
        del pts_features, recon_cond_features  # Free remaining large tensors
        
        # Remove keys not needed for gen output
        for gen_out in gen_outputs:
            for k in ["ray_conf", "c2w", "intrinsics"]:
                gen_out.pop(k, None)

        # Merge results from all batches
        if len(gen_outputs) > 1:
            gen_output = concat_preds(*gen_outputs)
        else:
            gen_output = gen_outputs[0]
       
        return gen_output
        
    def _process_ray_pose_estimation(
        self, ray: torch.Tensor, ray_conf: torch.Tensor, height: int, width: int
    ) -> Dict[str, torch.Tensor]:
        """Process ray pose estimation if ray pose decoder is available."""
        pred_extrinsic, pred_focal_lengths, pred_principal_points = get_extrinsic_from_camray(
            ray,
            ray_conf.clone(),  # Clone to prevent in-place modification in compute_optimal_rotation_intrinsics_batch
            ray.shape[-3],
            ray.shape[-2],
        )
        # pred_extrinsic = affine_inverse(pred_extrinsic) # w2c -> c2w
        pred_extrinsic = pred_extrinsic[:, :, :3, :]
        pred_intrinsic = torch.eye(3, 3)[None, None].repeat(pred_extrinsic.shape[0], pred_extrinsic.shape[1], 1, 1).clone().to(pred_extrinsic.device)
        pred_intrinsic[:, :, 0, 0] = pred_focal_lengths[:, :, 0] / 2 * width
        pred_intrinsic[:, :, 1, 1] = pred_focal_lengths[:, :, 1] / 2 * height
        pred_intrinsic[:, :, 0, 2] = pred_principal_points[:, :, 0] * width * 0.5
        pred_intrinsic[:, :, 1, 2] = pred_principal_points[:, :, 1] * height * 0.5
        # del output.ray
        # del output.ray_conf
        # output.extrinsics = pred_extrinsic
        # output.intrinsics = pred_intrinsic
        return pred_extrinsic, pred_intrinsic
        
    def _process_camera_estimation(
        self, feats: list[torch.Tensor], H: int, W: int, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Process camera pose estimation if camera decoder is available."""
   
        pose_enc = self.cam_dec(feats[-1][1])
        # Remove ray information as it's not needed for pose estimation
        # if "ray" in output:
        #     del output.ray
        # if "ray_conf" in output:
        #     del output.ray_conf

        # Convert pose encoding to extrinsics and intrinsics
        c2w, intrinsics = pose_encoding_to_extri_intri(pose_enc, (H, W))
        # output.extrinsics = affine_inverse(c2w)
        # output.intrinsics = ixt

        return c2w, intrinsics

    def _process_depth_output(
        self,
        feats,
        h,
        w,
        device_type,
        gen_raymap=None,
        pose_from_depth_ray=False,
        pose_from_cam_dec=False,
        point_from_depth_and_pose=False,
        images=None,
        head=None,  # Optional: use specific head (e.g., frozen aux_head)
    ):
        """
        Process depth head output and compute pointmap, pose, and SAM features.
        
        Args:
            feats: Features from backbone
            h, w: Image height and width
            device_type: Device type for autocast (e.g., 'cuda')
            pose_from_depth_ray: Whether to estimate pose from depth and ray
            pose_from_cam_dec: Whether to estimate pose from camera decoder
            
        Returns:
            Dict with pointmap, depth, depth_conf, ray, ray_conf, c2w, intrinsics, sam_feats
        """
        # Use provided head or default to main head
        depth_head = head if head is not None else self.model.head
        output = depth_head(feats, h, w, patch_start_idx=0)

        default_scale = 20
        # Extract depth and raymap from raw output
        depth = output["depth"]  # (B, T, H_proc, W_proc)
        depth = depth * default_scale
        depth_conf = output["depth_conf"]  # (B, T, H_proc, W_proc)

        if gen_raymap is not None:
            # Use provided gen_raymap (ground-truth or from camera poses)
            ray = gen_raymap
            ray_conf = None  # No confidence when using provided raymap
        else:
            # Use predicted raymap from the model
            ray = output["ray"]  # (B, T, H_proc, W_proc, 6) with [ray_dirs(3), ray_origins(3)]
            ray_conf = output["ray_conf"]  # (B, T, H_proc, W_proc)
            
            # Apply default scaling to ray origins
            ray[..., 3:] = ray[..., 3:] * default_scale
        
        c2w = None
        intrinsics = None
        if pose_from_depth_ray:
            # Cast to float32 to avoid torch.inverse() error with bf16/fp16
            with torch.autocast(device_type=device_type, enabled=False):
                c2w, intrinsics = self._process_ray_pose_estimation(ray.float(), ray_conf.float(), h, w)
    
        if pose_from_cam_dec:
            c2w, intrinsics = self._process_camera_estimation(feats, h, w, output)
         
        if point_from_depth_and_pose:
            assert intrinsics is not None and c2w is not None, "Intrinsics and c2w must be estimated to compute pointmap from depth and pose"
            pointmap = convert_depth_to_point_cloud(depth, intrinsics, c2w)
        else:    
            # DA3 paper: "We do not normalize d, so its magnitude preserves the projection scale.
            # Thus, a 3D point in world coordinates is simply P = t + D(u, v) · d"
            # where t = ray origin, D = depth, d = unnormalized ray direction
            pointmap = depth.unsqueeze(-1) * ray[..., :3] + ray[..., 3:]
            
        # Compute SAM features inside forward pass (important for DDP to avoid "marked ready twice" error)
        sam_feats = None
        if self.head_sam is not None:
            sam_feats = self.forward_sam_features(feats, (h, w))

        save_outputs = False
        if save_outputs:
            # Auto-incrementing folder logic for /scratch/project/eu-25-92/ssc_output/da3/
            base_output_dir = "/scratch/project/eu-25-92/ssc_output/da3"
            os.makedirs(base_output_dir, exist_ok=True)
            
            # Find next available folder number (00000, 00001, ...)
            folder_idx = 0
            # while os.path.exists(os.path.join(base_output_dir, f"{folder_idx:05d}")):
            #     folder_idx += 1
            
            output_folder = os.path.join(base_output_dir, f"{folder_idx:05d}")
            os.makedirs(output_folder, exist_ok=True)
            
            # Prepare data for saving
            # Extract raw data (no scaling yet)
            batch_idx = 0
            pts3d_render = pointmap[batch_idx].detach().cpu()  # (T, H, W, 3)
            conf_render = depth_conf[batch_idx].detach().cpu()  # (T, H, W)
            gt_depths = depth[batch_idx].detach().cpu()  # (T, H, W)
            
            # Intrinsics and c2w
            if intrinsics is not None and c2w is not None:
                focal = torch.stack([intrinsics[batch_idx, :, 0, 0], intrinsics[batch_idx, :, 1, 1]], dim=-1).detach().cpu()  # (T, 2)
                c2w_save = c2w[batch_idx].detach().cpu()  # (T, 3, 4) camera-to-world
                
                # Apply scaling factor of 30 to scene coordinates
                # SCALE_FACTOR = 30
                # pts3d_render = pts3d_render * SCALE_FACTOR  # Scale 3D points
                # c2w_save[:, :, 3] = c2w_save[:, :, 3] * SCALE_FACTOR  # Scale translation component
                
                # Create save dictionary
                save_dict = {
                    "pts3d": pts3d_render.numpy(),
                    "conf": conf_render.numpy(),
                    "gt_depths": gt_depths.numpy(),
                    "focal": focal[:, 0].numpy(),
                    "c2w": c2w_save.numpy(),
                }
                
                # Add colors if images are available
                if images is not None:
                    # images: (B, T, C, H, W) -> extract batch_idx and convert to (T, H, W, 3)
                    colors = images[batch_idx].detach().cpu()  # (T, C, H, W)
                    colors = colors.permute(0, 2, 3, 1)  # (T, H, W, C)
                    # ImageNet un-normalization to [0, 1], then map to [-1, 1]
                    mean = torch.tensor([0.485, 0.456, 0.406])
                    std = torch.tensor([0.229, 0.224, 0.225])
                    colors = (colors * std + mean) * 2.0 - 1.0
                    colors = colors.clamp(-1, 1)
                    save_dict["colors"] = colors.numpy()
                
                # Save to .npy file
                save_path = os.path.join(output_folder, "pts3d_render.npy")
                np.save(save_path, save_dict)
                print(f"Saved pts3d_render.npy to {save_path}")
              

        return {
            "pointmap": pointmap,
            "depth": depth,
            "depth_conf": depth_conf,
            "ray": ray,
            "ray_conf": ray_conf,
            "c2w": c2w,
            "intrinsics": intrinsics,
            "sam_feats": sam_feats,
        }

    def inference_batch(
        self,
        images,
        export_feat_layers=None,
        process_res=518,
        process_res_method="upper_bound_resize",
        save_outputs=False,
        save_prefix="da3",
        pose_from_depth_ray=False,
        pose_from_cam_dec=False,
        point_from_depth_and_pose=False,
    ):
        b, t, c, h, w = images.size()

        device = self._get_model_device()

        # Prepare export_feat_layers
        if export_feat_layers is None:
            export_feat_layers = list(self.model.backbone.out_layers)
        else:
            export_feat_layers = list(export_feat_layers)
        
        # Add aux input layer to export_feat_layers if aux branch is enabled
        if hasattr(self, 'aux_input_layer_idx'):
            if self.aux_input_layer_idx not in export_feat_layers:
                export_feat_layers.append(self.aux_input_layer_idx)

        # Keep export layer order consistent with backbone traversal order.
        export_feat_layers = sorted(set(export_feat_layers))

        # Call the underlying model directly to get raw output including ray field
        # before any post-processing deletes it.
        # The DualDPT head outputs 'ray' (B, T, H, W, 6) with [ray_dirs(3), ray_origins(3)]
        autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
       
        # NOTE: the current inference_batch_aux is only correct with ref_view_strategy="first" for now as
        # it use the token after reference reordering without reorder to the original view.
        ref_view_strategy = "first"
        assert ref_view_strategy == "first", \
            "inference_batch_aux requires ref_view_strategy='first' because aux tokens are not restored to original order"
        
        device_type = images.device.type
        feats, aux_feats = self.model.backbone(
            images,
            cam_token=None,
            export_feat_layers=export_feat_layers,
            ref_view_strategy=ref_view_strategy
        )

        output = self._process_depth_output(
            feats=feats,
            h=h,
            w=w,
            device_type=device_type,
            pose_from_depth_ray=pose_from_depth_ray,
            pose_from_cam_dec=pose_from_cam_dec,
            point_from_depth_and_pose=point_from_depth_and_pose,
            images=images,
        )
        
        # Run aux branch if initialized
        if hasattr(self, 'aux_blocks') and hasattr(self, 'aux_input_layer_idx'):
            # Get aux intermediate tokens from aux_feats (at aux_input_layer_idx)
            aux_idx = export_feat_layers.index(self.aux_input_layer_idx)
            # aux_feats[aux_idx] is (processed_feat, raw_state) - use raw_state for aux branch
            aux_feat_tuple = aux_feats[aux_idx]
            aux_intermediate = aux_feat_tuple[1]  # raw_state = (x, local_x)
            
            # Get out_layers and identify which are before aux branch
            out_layers = self.model.backbone.out_layers
            out_layers_before_aux = [l for l in out_layers if l <= self.aux_input_layer_idx]
           
            # Extract main backbone features for layers before aux branch
            # feats is indexed by out_layers order, so we take the first N items
            main_feats_for_aux = feats[:len(out_layers_before_aux)]
            
            with torch.no_grad():
                output['aux_outputs'] = self.inference_batch_aux(
                    aux_intermediate, h, w, device_type, main_feats=main_feats_for_aux
                )

        if aux_feats is not None:
            output["aux_feats"] = aux_feats

        return output
    
    def inference_batch_gen(
        self,
        patch_tokens,
        H, W,
        gen_raymap=None,
        n_gen_views=None,
        export_feat_layers=None,
        pose_from_depth_ray=False,
        point_from_depth_and_pose=False,
        return_aux_feats=True,
    ):
        """
        Inference for gen views using pre-computed patch tokens from RaymapEncoderDA3.
        
        Args:
            patch_tokens: (B, T, N, embed_dim) pre-computed tokens from RaymapEncoderDA3
            H, W: original image dimensions
            gen_raymap: optional raymap for all views (recon + gen)
            n_gen_views: number of gen views to process in the depth head (last n views)
            export_feat_layers: layers to export features from
            pose_from_depth_ray: whether to estimate pose from depth and ray
            
        Returns:
            Dict with pointmap, depth, depth_conf, ray, ray_conf
        """
        b, t, n, _ = patch_tokens.shape

        if export_feat_layers is None:
            export_feat_layers = list(self.model.backbone.out_layers)
        else:
            export_feat_layers = list(export_feat_layers)
        
        # Pass patch tokens through DinoV2 backbone using is_gen=True
        # This uses the forward method which is compatible with PEFT-wrapped models
        feats, aux_feats = self.model.backbone(
            patch_tokens,
            is_gen=True,
            H=H,
            W=W,
            n_gen_views=n_gen_views,
            slice_layer_idx=self.slice_layer_idx,
            export_feat_layers=export_feat_layers,
            ref_view_strategy="first"
        )
        
        # If n_gen_views is provided, slice feats and gen_raymap to only include gen views.
        # This avoids processing recon views in the depth head.
        if n_gen_views is not None and n_gen_views < t:
            # feats is a list of (feat, camera_token) tuples
            # Slice each element in the tuple along the view dimension (dim 1)
            feats = [
                (f[:, -n_gen_views:], c[:, -n_gen_views:])
                for f, c in feats
            ]
            if gen_raymap is not None:
                gen_raymap = gen_raymap[:, -n_gen_views:]
            
            for i, feat in enumerate(aux_feats):
                processed_feat, raw_state = feat
                processed_feat = processed_feat[:, -n_gen_views:]
                raw_state = tuple(state[:, -n_gen_views:] for state in raw_state)
                aux_feats[i] = (processed_feat, raw_state)
        
        device_type = patch_tokens.device.type
        output = self._process_depth_output(
            feats=feats,
            gen_raymap=gen_raymap,
            h=H,
            w=W,
            device_type=device_type,
            pose_from_depth_ray=pose_from_depth_ray,
            pose_from_cam_dec=False,
            point_from_depth_and_pose=point_from_depth_and_pose,
        )

        if aux_feats is not None and return_aux_feats:
            output["aux_feats"] = aux_feats

        return output
