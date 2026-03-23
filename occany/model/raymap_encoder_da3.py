# Copyright (C) 2025-present. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# RaymapEncoderDA3: Converts projected point features to patch tokens
# for DA3's DinoV2 backbone
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial

from dust3r.patch_embed import get_patch_embed
from occany.model.must3r_blocks.pos_embed import get_pos_embed
from occany.model.must3r_blocks.layers import BaseTransformer, TimestepEmbedder, DiTBlock


class RaymapEncoderDA3(BaseTransformer):
    """
    Converts projected point features to patch tokens compatible with
    DA3's DinoV2 transformer blocks.
    
    Input: Projected features (B, T, H, W, C_proj) where C_proj = sum of feature channels
    Output: Patch tokens (B*T, N, embed_dim=1024) matching DinoV2-L internal format
    
    The output tokens can be passed directly to DinoV2 transformer blocks after
    adding cls token and position encoding.
    """
    
    def __init__(
        self,
        img_size=(518, 518),        # DA3 input size  
        patch_size=14,              # DinoV2-L patch size
        embed_dim=1024,             # DinoV2-L embed dim
        output_embed_dim=1024,      # Output embed dim (same as DinoV2)
        depth=6,                    # Encoder transformer depth
        num_heads=16,               # Number of attention heads
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        use_time_cond=True,
        patch_embed='PatchEmbedDust3R',
        pos_embed='RoPE100',
        projection_features='pts3d_local,pts3d,rgb,conf',
    ):
        super(RaymapEncoderDA3, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.patch_size = patch_size
        
        # Parse projection features
        self.projection_features = [f.strip() for f in projection_features.split(',')]
        
        # Set up patch embeddings for each feature type
        self._init_patch_embeds(patch_embed, img_size, patch_size, embed_dim)
        
        self.max_seq_len = max(img_size) // patch_size
        self.rope = get_pos_embed(pos_embed)
        
        # Timestep embedder and transformer blocks (only when depth > 0)
        if self.depth > 0:
            self.t_embedder = TimestepEmbedder(embed_dim)
            nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
            
            # Transformer blocks with DiT-style conditioning
            self.blocks_enc = nn.ModuleList([
                DiTBlock(embed_dim, num_heads, pos_embed=self.rope, 
                         mlp_ratio=mlp_ratio, use_time_cond=use_time_cond) 
                for _ in range(depth)
            ])
        else:
            self.t_embedder = None
            self.blocks_enc = nn.ModuleList([])
        
        # Optional projection if output dim differs
        if output_embed_dim != embed_dim:
            self.proj = nn.Linear(embed_dim, output_embed_dim, bias=True)
        else:
            self.proj = nn.Identity()
        
        self.norm_enc = norm_layer(output_embed_dim)
        self.initialize_weights()
    
    def _init_patch_embeds(self, patch_embed_name, img_size, patch_size, embed_dim):
        """Initialize patch embeddings for each projection feature type."""
        per_input_dim = 256  # Intermediate dim per feature type
        
        num_feature_embeds = 0
        first_patch_embed = None
        
        if 'pts3d_local' in self.projection_features:
            self.patch_embed_pts3d_local = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=3
            )
            num_feature_embeds += 1
            first_patch_embed = self.patch_embed_pts3d_local
        
        if 'pts3d' in self.projection_features:
            self.patch_embed_pts3d = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=3
            )
            num_feature_embeds += 1
            if first_patch_embed is None:
                first_patch_embed = self.patch_embed_pts3d
        
        if 'rgb' in self.projection_features:
            self.patch_embed_rgb = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=3
            )
            num_feature_embeds += 1
            if first_patch_embed is None:
                first_patch_embed = self.patch_embed_rgb
        
        if 'conf' in self.projection_features:
            self.patch_embed_conf = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=1
            )
            num_feature_embeds += 1
            if first_patch_embed is None:
                first_patch_embed = self.patch_embed_conf
        
        if 'raymap' in self.projection_features:
            self.patch_embed_raymap = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=6
            )
            num_feature_embeds += 1
            if first_patch_embed is None:
                first_patch_embed = self.patch_embed_raymap
        
        # SAM3 features (256 + 256 + 256 channels from 3 scales)
        if 'sam3' in self.projection_features:
            self.patch_embed_sam3_s0 = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=256
            )
            self.patch_embed_sam3_s1 = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=256
            )
            self.patch_embed_sam3_s2 = get_patch_embed(
                patch_embed_name, img_size, patch_size, per_input_dim, in_chans=256
            )
            num_feature_embeds += 3
            if first_patch_embed is None:
                first_patch_embed = self.patch_embed_sam3_s0
        
        # Projection to combine all feature embeddings into embed_dim
        # if self.depth > 0:
        self.patch_embed_proj = nn.Linear(per_input_dim * num_feature_embeds, self.embed_dim)
        # else:
        #     # Use Mlp for more capacity when no transformer blocks
        #     self.patch_embed_proj = Mlp(
        #         in_features=per_input_dim * num_feature_embeds,
        #         hidden_features=self.embed_dim * 4,
        #         out_features=self.embed_dim,
        #         act_layer=nn.GELU
        #     )
        self.grid_size = first_patch_embed.grid_size
        
    def _get_channel_split_sizes(self):
        """Get channel split sizes based on enabled projection features."""
        split_sizes = []
        if 'pts3d_local' in self.projection_features:
            split_sizes.append(3)
        if 'pts3d' in self.projection_features:
            split_sizes.append(3)
        if 'rgb' in self.projection_features:
            split_sizes.append(3)
        if 'conf' in self.projection_features:
            split_sizes.append(1)
        if 'raymap' in self.projection_features:
            split_sizes.append(6)
        if 'sam3' in self.projection_features:
            split_sizes.extend([256, 256, 256])
        return split_sizes
    
    def forward(self, projected_features, true_shape, timesteps=None):
        """
        Forward pass through the raymap encoder.
        
        Args:
            projected_features: (B*T, C, H, W) projected point features in channel-first format
                               where C = sum of enabled feature channels
            true_shape: (B*T, 2) tensor of (H, W) for each view
            timesteps: (B, T) optional timestep tensor for temporal conditioning
            
        Returns:
            patch_tokens: (B*T, N, embed_dim) patch tokens for DinoV2
            pos: (B*T, N, 2) position encoding for each patch
        """
        BT = projected_features.shape[0]
        
        # Split input by feature type
        split_sizes = self._get_channel_split_sizes()
        inputs = projected_features.split(split_sizes, dim=1)
        
        x_list = []
        pos = None
        input_idx = 0
        
        if 'pts3d_local' in self.projection_features:
            x_pts3d_local, pos = self.patch_embed_pts3d_local(inputs[input_idx], true_shape=true_shape)
            x_list.append(x_pts3d_local)
            input_idx += 1
        
        if 'pts3d' in self.projection_features:
            x_pts3d, pos_tmp = self.patch_embed_pts3d(inputs[input_idx], true_shape=true_shape)
            x_list.append(x_pts3d)
            if pos is None:
                pos = pos_tmp
            input_idx += 1
        
        if 'rgb' in self.projection_features:
            x_rgb, pos_tmp = self.patch_embed_rgb(inputs[input_idx], true_shape=true_shape)
            x_list.append(x_rgb)
            if pos is None:
                pos = pos_tmp
            input_idx += 1
        
        if 'conf' in self.projection_features:
            x_conf, pos_tmp = self.patch_embed_conf(inputs[input_idx], true_shape=true_shape)
            x_list.append(x_conf)
            if pos is None:
                pos = pos_tmp
            input_idx += 1
        
        if 'raymap' in self.projection_features:
            x_raymap, pos_tmp = self.patch_embed_raymap(inputs[input_idx], true_shape=true_shape)
            x_list.append(x_raymap)
            if pos is None:
                pos = pos_tmp
            input_idx += 1
        
        if 'sam3' in self.projection_features:
            x_sam3_s0, pos_tmp = self.patch_embed_sam3_s0(inputs[input_idx], true_shape=true_shape)
            x_sam3_s1, _ = self.patch_embed_sam3_s1(inputs[input_idx + 1], true_shape=true_shape)
            x_sam3_s2, _ = self.patch_embed_sam3_s2(inputs[input_idx + 2], true_shape=true_shape)
            x_list.extend([x_sam3_s0, x_sam3_s1, x_sam3_s2])
            if pos is None:
                pos = pos_tmp
        
        # Concatenate and project to embed_dim
        x = torch.cat(x_list, dim=2)  # (BT, N, per_input_dim * num_features)
        x = self.patch_embed_proj(x)  # (BT, N, embed_dim)
        
        # Process through encoder blocks (only when depth > 0)
        if self.depth > 0:
            # Compute time embedding if provided
            if timesteps is not None:
                t_emb = self.t_embedder(timesteps.reshape(-1))
                # Expand t_emb to match token sequence
                t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # (BT, N, embed_dim)
            else:
                t_emb = torch.zeros_like(x)
            
            for blk in self.blocks_enc:
                x = blk(x, pos, t_emb)
        
        x = self.proj(x)
        x = self.norm_enc(x)
        
        return x, pos
