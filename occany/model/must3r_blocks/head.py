# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn
from enum import Enum
from typing import List, Tuple
from occany.model.must3r_blocks.image import unpatchify
from occany.model.must3r_blocks.geometry import apply_exp_to_norm

import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import Mlp
import torch.nn.functional as F

# Import DPT utilities for DPTProj
from depth_anything_3.model.dpt import _make_fusion_block, _make_scratch


class ActivationType(Enum):
    NORM_EXP = "norm_exp"
    LINEAR = "linear"


def apply_activation(xyz, activation):
    if isinstance(activation, str):
        activation = ActivationType(activation)
    if activation == ActivationType.NORM_EXP:
        return apply_exp_to_norm(xyz, dim=-1)
    elif activation == ActivationType.LINEAR:
        return xyz
    else:
        raise ValueError(f"Unknown activation: {activation}")


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        x = head(decout, (H, W))
        return x

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return head(decout, (W, H)).swapaxes(1, 2)

        # batch is a mix of both portrait & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = head(selout(is_portrait), (W, H)).swapaxes(1, 2)

        x = l_result.new(B, *l_result.shape[1:])
        x[is_landscape] = l_result
        x[is_portrait] = p_result
        return x

    return wrapper_yes if activate else wrapper_no


class LinearHead(nn.Module):
    def __init__(self, embed_dim, output_dim, patch_size, use_mlp=False):
        super().__init__()
        self.patch_size = patch_size
        if use_mlp:
            self.proj = Mlp(in_features=embed_dim, out_features=output_dim)
        else:
            self.proj = nn.Linear(embed_dim, output_dim, bias=True)

    def forward(self, feats, img_shape):
        x = self.proj(feats[-1])
        x = unpatchify(x, self.patch_size, img_shape).permute(0, 2, 3, 1)
        return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class DPTProj(nn.Module):
    """
    DPT-style projection with multi-level feature pyramid fusion.
    Based on DualDPT architecture from Depth-Anything-3.
    
    Takes 4 intermediate feature levels and produces fused output via pyramid fusion.
    
    Architecture:
        1. Unpatchify each of 4 input features to spatial format
        2. Project each via 1x1 conv: input_dims[i] -> out_channels[i]
        3. Resize to common scale (x4, x2, x1, /2 relative to patch grid)
        4. Apply scratch layers (layer1_rn, ..., layer4_rn)
        5. Pyramid fusion (refinenet4 -> refinenet3 -> refinenet2 -> refinenet1)
        6. Final 1x1 conv -> out_dim
    """
    def __init__(
        self,
        input_dims: Tuple[int, int, int, int],  # Dims for each of 4 levels (e.g., (2048, 2048, 2048, 2048))
        out_dim: int,                            # Final output dimension (e.g., 1024 for SAM3)
        features: int = 256,                     # Intermediate feature dimension
        out_channels: Tuple[int, int, int, int] = (256, 512, 1024, 1024),  # Per-level channel counts
        patch_size: int = 14,                    # Patch size for unpatchify
    ):
        super().__init__()
        self.patch_size = patch_size
        self.features = features
        self.intermediate_layer_idx = (0, 1, 2, 3)  # Fixed 4 levels
        
        # Per-level 1x1 projections
        self.projects = nn.ModuleList([
            nn.Conv2d(input_dims[i], out_channels[i], kernel_size=1, stride=1, padding=0)
            for i in range(4)
        ])
        
        # Resize layers to align to common scale (relative to patch grid)
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),  # x4
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),  # x2
            nn.Identity(),                                                                              # x1
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),          # /2
        ])
        
        # Scratch: stage adapters
        self.scratch = _make_scratch(list(out_channels), features, expand=False)
        
        # Main fusion chain (4 refinenet blocks)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)
        
        # Final projection to out_dim
        self.final_proj = nn.Conv2d(features, out_dim, kernel_size=1, stride=1, padding=0)
    
    def forward(self, feats: List[torch.Tensor], img_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            feats: List of 4 feature tensors, each [B, N, C]
            img_shape: (H, W) of input image
            
        Returns:
            Fused features: [B, out_dim, H_feat, W_feat]
        """
        B = feats[0].shape[0]
        ph, pw = img_shape[0] // self.patch_size, img_shape[1] // self.patch_size
        
        # 1. Unpatchify + project + resize each level
        resized_feats = []
        for stage_idx in range(4):
            # Unpatchify to spatial: [B, N, C] -> [B, C, ph, pw]
            x = feats[stage_idx].permute(0, 2, 1).reshape(B, -1, ph, pw)
            # Project channels
            x = self.projects[stage_idx](x)
            # Resize to common scale
            x = self.resize_layers[stage_idx](x)
            resized_feats.append(x)
        
        # 2. Apply scratch layers
        l1, l2, l3, l4 = resized_feats
        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)
        
        # 3. Pyramid fusion (top-down: 4 -> 3 -> 2 -> 1)
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        
        # 4. Final projection
        out = self.final_proj(out)
        
        return out


class SAMHead(nn.Module):
    def __init__(self, input_dim, img_size=512, embed_dim=256, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.proj = Mlp(in_features=input_dim, out_features=embed_dim)
        self.img_size = img_size
        self.up1 = nn.Sequential(
            # First upsampling block
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=3, padding=1),
            LayerNorm2d(embed_dim // 4),
            nn.GELU(),

        )

        self.up2 = nn.Sequential(
            # Second upsampling block
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(embed_dim // 4, embed_dim // 8, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, feats, img_shape):
        x = unpatchify(feats[-1], 1, (img_shape[0]//self.patch_size, img_shape[1]//self.patch_size))# .permute(0, 2, 3, 1)
        x = F.interpolate(x, size=(self.img_size//self.patch_size, self.img_size//self.patch_size), mode='bilinear', align_corners=False) # 32x32 for SAM2 image size of 512
        image_embed = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_s1 = self.up1(image_embed)
        feat_s0 = self.up2(feat_s1)
        return image_embed, feat_s1, feat_s0


class IdentityTrunk(nn.Module):
    """Identity trunk that passes through features, used with Sam3DualViTDetNeck."""
    def __init__(self, channel_dim):
        super().__init__()
        # Sam3DualViTDetNeck expects trunk.channel_list[-1]
        self.channel_list = [channel_dim]
    
    def forward(self, x):
        # Return as list (neck expects list and takes [-1])
        return [x]


class SAM3Head(nn.Module):
    """
    Head to produce multi-scale feature maps matching SAM3's backbone.forward_image() output.
    Uses Sam3DualViTDetNeck directly with scale_factors=[4.0, 2.0, 1.0] (scalp=1 removes 0.5x).
    
    SAM3 backbone_fpn shapes (for 1008x1008 input):
      - feat_s0: [B, 256, 288, 288] (4x upsampled, high-res)
      - feat_s1: [B, 256, 144, 144] (2x upsampled, mid-res)
      - feat_s2: [B, 256, 72, 72]   (1x, low-res, same as vision_features)
    
    Output dict matches SAM3VLBackbone.forward_image():
      {
          "vision_features": feat_s2,           # lowest-res feature
          "vision_pos_enc": [pos_s0, pos_s1, pos_s2],  # position encodings
          "backbone_fpn": [feat_s0, feat_s1, feat_s2], # high-res to low-res
          "sam2_backbone_out": None,
      }
    """
    def __init__(self, input_dim=None, input_dims=None, img_size=518, embed_dim=256, patch_size=16, scalp=1, use_dpt_proj=False):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.scalp = scalp
        self.use_dpt_proj = use_dpt_proj  # Store for forward()
        
        # Project decoder features to the dimension expected by neck (1024 for SAM3's ViT)
        # Sam3DualViTDetNeck expects trunk output dim, which it divides for different scales
        self.trunk_dim = 1024  # SAM3's ViT embed_dim
        
        if use_dpt_proj:
            # DPT mode: multi-level pyramid fusion
            assert input_dims is not None, "input_dims (tuple of 4) required for use_dpt_proj=True"
            print(f'[INFO] SAM3Head using DPTProj with input_dims={input_dims}')
            self.proj = DPTProj(
                input_dims=input_dims,
                out_dim=self.trunk_dim,
                features=256,
                patch_size=patch_size,
            )
        else:
            # Current Mlp mode: single concatenated feature
            assert input_dim is not None, "input_dim required for use_dpt_proj=False"
            print(f'[INFO] SAM3Head using Mlp with input_dim={input_dim}')
            self.proj = Mlp(in_features=input_dim, out_features=self.trunk_dim)
        
        # Base feature size (1x scale): img_size // 14 for SAM3's patch_size=14
        self.feat_base_size = img_size // 14
        
        # Position encoding (same config as SAM3)
        from sam3.model_builder import _create_position_encoding
        position_encoding = _create_position_encoding(precompute_resolution=1008) # Match SAM3's precompute_resolution
        
        # Identity trunk - just passes through features
        trunk = IdentityTrunk(channel_dim=self.trunk_dim)
        
        # Use Sam3DualViTDetNeck directly
        from sam3.model.necks import Sam3DualViTDetNeck
        self.neck = Sam3DualViTDetNeck(
            trunk=trunk,
            position_encoding=position_encoding,
            d_model=embed_dim,
            scale_factors=(4.0, 2.0, 1.0, 0.5),  # Same as SAM3
            add_sam2_neck=False,
        )

    def forward(self, feats, img_shape):
        """
        Returns multi-scale features matching SAM3's backbone.forward_image() output,
        plus the pre-neck features for distillation.
        
        Args:
            feats:
                - If use_dpt_proj=False: List with one element feats[-1] of shape [B, N, C_concat]
                - If use_dpt_proj=True: List of 4 elements, each [B, N, C_i]
            img_shape: (H, W) of the input image
            
        Returns:
            Tuple of (feat_s0, feat_s1, feat_s2, pre_neck_feat):
                - feat_s0: [B, 256, H0, W0] high-res features
                - feat_s1: [B, 256, H1, W1] mid-res features  
                - feat_s2: [B, 256, H2, W2] low-res features
                - pre_neck_feat: [B, 1024, H, W] features before the neck (for distillation)
        """
    
        if self.use_dpt_proj:
            # DPT mode: feats is list of 4 features [B, N, C]
            # DPTProj handles unpatchify + pyramid fusion internally
            pre_neck_feat = self.proj(feats, img_shape)  # [B, trunk_dim, H_base, W_base]
            
            # Resize to base feature size if needed
            if pre_neck_feat.shape[2] != self.feat_base_size or pre_neck_feat.shape[3] != self.feat_base_size:
                pre_neck_feat = F.interpolate(
                    pre_neck_feat, 
                    size=(self.feat_base_size, self.feat_base_size), 
                    mode='bilinear', 
                    align_corners=False
                )
        else:
            # Current Mlp mode: feats[-1] is concatenated feature
            # Unpatchify decoder features to spatial format
            x = unpatchify(feats[-1], 1, (img_shape[0]//self.patch_size, img_shape[1]//self.patch_size))
            
            # Resize to base feature size (1x scale)
            x = F.interpolate(x, size=(self.feat_base_size, self.feat_base_size), mode='bilinear', align_corners=False)
            
            # Project to trunk_dim (8192 -> 1024)
            pre_neck_feat = self.proj(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # [B, 1024, H, W]
        
        # Forward through Sam3DualViTDetNeck
        sam3_features, sam3_pos, _, _ = self.neck(pre_neck_feat)
        
        # Apply scalp (remove lowest resolution features, same as SAM3VLBackbone with scalp=1)
        if self.scalp > 0:
            sam3_features = sam3_features[:-self.scalp]
            sam3_pos = sam3_pos[:-self.scalp]
        
        # Return (feat_s0, feat_s1, feat_s2, pre_neck_feat) as a tuple
        # [B, 256, 148, 148], [B, 256, 74, 74], [B, 256, 37, 37], [B, 1024, 37, 37]
        return sam3_features[0], sam3_features[1], sam3_features[2], pre_neck_feat
        # Return dict matching SAM3VLBackbone.forward_image() output
        # return {
        #     "vision_features": sam3_features[-1],  # lowest-res in remaining features
        #     "vision_pos_enc": sam3_pos,
        #     "backbone_fpn": sam3_features,
        #     "sam2_backbone_out": None,
        # } 