import torch
from packaging import version
import huggingface_hub
import torch.nn as nn
from dust3r.utils.misc import transpose_to_landscape
from dust3r.patch_embed import get_patch_embed

from dust3r.utils.misc import is_symmetrized, interleave, freeze_all_params
from croco.models.blocks import Mlp
from occany.model.pos_embed import RoPE1D


from occany.model.must3r_blocks.pos_embed import get_pos_embed
from occany.model.must3r_blocks.layers import Block, BaseTransformer, CachedDecoderBlock, get_current_dtype, TimestepEmbedder, DiTBlock
from occany.model.must3r_blocks.head import ActivationType, SAMHead, SAM3Head, LinearHead, transpose_to_landscape, apply_activation
from occany.model.must3r_blocks.feedback_mechanism import init_feedback_layers, create_feedback_layers, run_feedback_layers
from occany.model.must3r_blocks.dropout import MemoryDropoutSelector, TemporaryMemoryDropoutSelector
from functools import partial

    
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


class Dust3rEncoder(BaseTransformer):
    def __init__(self,
                 img_size=(224, 224),           # input image size
                 patch_size=16,          # patch_size
                 embed_dim=1024,      # encoder feature dimension
                 depth=24,           # encoder depth
                 num_heads=16,       # encoder number of heads in the transformer block
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 patch_embed='PatchEmbedDust3R',
                 pos_embed='RoPE100'):
        super(Dust3rEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth

        self.set_patch_embed(patch_embed, img_size, patch_size, embed_dim)

        self.max_seq_len = max(img_size) // patch_size
        self.grid_size = self.patch_embed.grid_size
        self.rope = get_pos_embed(pos_embed)

        self.blocks_enc = nn.ModuleList([
            Block(embed_dim, num_heads, pos_embed=self.rope, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm_enc = norm_layer(embed_dim)
        self.initialize_weights()

    def set_patch_embed(self, patch_embed_name='PatchEmbedDust3R', img_size=224, patch_size=16, patch_embed_dim=768):
        self.patch_size = patch_size
        assert self.embed_dim == patch_embed_dim
        self.patch_embed = get_patch_embed(patch_embed_name, img_size, patch_size, patch_embed_dim, in_chans=3)
        self.grid_size = self.patch_embed.grid_size

    @torch.autocast("cuda", dtype=torch.float32)
    def forward(self, img, true_shape):
        # img: [80, 3, 80, 224])
        x, pos = self.patch_embed(img, true_shape=true_shape)
        
        # x: [80, 70, 1024]
        for blk in self.blocks_enc:
            x = blk(x, pos)
        x = self.norm_enc(x)
        return x, pos


class RaymapEncoderDiT(BaseTransformer):
    def __init__(self,
                 img_size=(224, 224),           # input image size
                 patch_size=16,          # patch_size
                 embed_dim=768,      # encoder feature dimension
                 output_embed_dim=1024,
                 depth=6,           # encoder depth
                 num_heads=16,       # encoder number of heads in the transformer block
                 mlp_ratio=4,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 use_time_cond=True,
                #  in_chans=6,
                 patch_embed='PatchEmbedDust3R',
                 pos_embed='RoPE100',
                 use_raymap_only_conditioning=False,
                 projection_features='pts3d_local,pts3d,rgb,conf,sam'):
        super(RaymapEncoderDiT, self).__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        # self.in_chans = in_chans
        self.use_raymap_only_conditioning = use_raymap_only_conditioning
        # Parse projection features
        self.projection_features = [f.strip() for f in projection_features.split(',')]
        
        self.set_patch_embed(patch_embed, img_size, patch_size, embed_dim, use_raymap_only_conditioning, self.projection_features)

        self.max_seq_len = max(img_size) // patch_size
        # self.grid_size = self.patch_embed.grid_size
        self.rope = get_pos_embed(pos_embed)

        self.t_embedder = TimestepEmbedder(embed_dim)
        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        # self.blocks_enc = nn.ModuleList([
        #     Block(embed_dim, num_heads, pos_embed=self.rope, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
        #     for i in range(depth)])
        self.blocks_enc = nn.ModuleList([
            DiTBlock(embed_dim, num_heads, pos_embed=self.rope, mlp_ratio=mlp_ratio, use_time_cond=use_time_cond) for _ in range(depth)
        ])
        # Zero-out adaLN modulation layers in DiT blocks:
        # for block in self.blocks_enc:
        #     nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
        #     nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        if output_embed_dim != embed_dim:
            self.proj_mem = nn.Linear(output_embed_dim, embed_dim, bias=True)
            self.proj = nn.Linear(embed_dim, output_embed_dim, bias=True)
        else:
            self.proj = nn.Identity()
        

        self.norm_enc = norm_layer(output_embed_dim)
        self.initialize_weights()

    def set_patch_embed(self, patch_embed_name='PatchEmbedDust3R', 
                        # in_chans=6,
                        img_size=224, patch_size=16, patch_embed_dim=768, 
                        use_raymap_only_conditioning=False,
                        projection_features=None):
        self.patch_size = patch_size
        assert self.embed_dim == patch_embed_dim
        per_input_patch_embed_dim = 256
        
        if projection_features is None:
            projection_features = ['pts3d_local', 'pts3d', 'rgb', 'conf', 'sam']
        self.projection_features = projection_features
        if use_raymap_only_conditioning:
            # Dedicated patch embed for raymap-only conditioning (6 channels)
            self.patch_embed_raymap = get_patch_embed(patch_embed_name, 
                                            img_size, patch_size, patch_embed_dim, 
                                            in_chans=6)
            self.grid_size = self.patch_embed_raymap.grid_size
        else:
            # Patch embeds for projected point features conditioning
            # Only create patch embeds for features that are enabled
            num_feature_embeds = 0
            first_patch_embed = None
            
            if 'pts3d_local' in projection_features:
                self.patch_embed_pts3d_local = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=3)
                num_feature_embeds += 1
                first_patch_embed = self.patch_embed_pts3d_local
            
            if 'raymap' in projection_features:
                # Raymap has 6 channels (origin + direction)
                self.patch_embed_raymap = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=6)
                num_feature_embeds += 1
                if first_patch_embed is None:
                    first_patch_embed = self.patch_embed_raymap
            
            if 'pts3d' in projection_features:
                self.patch_embed_pts3d = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=3)
                num_feature_embeds += 1
                if first_patch_embed is None:
                    first_patch_embed = self.patch_embed_pts3d
            
            if 'rgb' in projection_features:
                self.patch_embed_rgb = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=3)
                num_feature_embeds += 1
                if first_patch_embed is None:
                    first_patch_embed = self.patch_embed_rgb
            
            if 'conf' in projection_features:
                self.patch_embed_conf = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=1)
                num_feature_embeds += 1
                if first_patch_embed is None:
                    first_patch_embed = self.patch_embed_conf
            
            if 'sam' in projection_features:
                self.patch_embed_sam_256 = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=256)
                self.patch_embed_sam_64 = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=64)
                self.patch_embed_sam_32 = get_patch_embed(patch_embed_name, 
                                                img_size, patch_size, per_input_patch_embed_dim, 
                                                in_chans=32)
                num_feature_embeds += 3  # sam_256, sam_64, sam_32
                if first_patch_embed is None:
                    first_patch_embed = self.patch_embed_sam_256

            if 'sam3' in projection_features:
                self.patch_embed_sam3_s0 = get_patch_embed(
                    patch_embed_name,
                    img_size,
                    patch_size,
                    per_input_patch_embed_dim,
                    in_chans=256,
                )
                self.patch_embed_sam3_s1 = get_patch_embed(
                    patch_embed_name,
                    img_size,
                    patch_size,
                    per_input_patch_embed_dim,
                    in_chans=256,
                )
                self.patch_embed_sam3_s2 = get_patch_embed(
                    patch_embed_name,
                    img_size,
                    patch_size,
                    per_input_patch_embed_dim,
                    in_chans=256,
                )
                num_feature_embeds += 3  # sam3_s0, sam3_s1, sam3_s2
                if first_patch_embed is None:
                    first_patch_embed = self.patch_embed_sam3_s0
            
           
            self.patch_embed_proj = nn.Linear(per_input_patch_embed_dim * num_feature_embeds, patch_embed_dim)
            
            self.grid_size = first_patch_embed.grid_size

    @torch.autocast("cuda", dtype=torch.float32)
    def forward(self, raymap, true_shape, mem, mem_raymap, mem_pos, 
                timesteps, mem_timesteps):
        # Shapes
        # - raymap: (B*nimgs, 6, H, W)
        # - true_shape: (B*nimgs, 2)
        # - mem: (B, Nm, Dm)
        # - mem_raymap: (B, nimgs_mem_raymap, 6, H, W)

        # Encode input raymaps
        # x: (B*nimgs, N, D), pos: (B*nimgs, N, D)
        
        mem = self.proj_mem(mem)
        B, D = mem.shape[0], mem.shape[-1]
        
        # Use dedicated raymap patch embed when using raymap-only conditioning
        if self.use_raymap_only_conditioning:
            x, pos = self.patch_embed_raymap(raymap, true_shape=true_shape)
        else:
            # Split input based on enabled projection features
            # Input channels (by feature key order):
            # pts3d_local(3), raymap(6), pts3d(3), rgb(3), conf(1), sam(256,64,32), sam3(256,256,256)
            split_sizes = []
            if 'pts3d_local' in self.projection_features:
                split_sizes.append(3)
            if 'raymap' in self.projection_features:
                split_sizes.append(6)
            if 'pts3d' in self.projection_features:
                split_sizes.append(3)
            if 'rgb' in self.projection_features:
                split_sizes.append(3)
            if 'conf' in self.projection_features:
                split_sizes.append(1)
            if 'sam' in self.projection_features:
                split_sizes.extend([256, 64, 32])
            if 'sam3' in self.projection_features:
                split_sizes.extend([256, 256, 256])
        
            inputs = raymap.split(split_sizes, dim=1)
           
            x_list = []
            pos = None
            input_idx = 0
            
            if 'pts3d_local' in self.projection_features:
                # print("using pts3d_local")
                x_pts3d_local, pos = self.patch_embed_pts3d_local(inputs[input_idx], true_shape=true_shape)
                x_list.append(x_pts3d_local)
                input_idx += 1
            
            if 'raymap' in self.projection_features:
                # print("using raymap")
                x_raymap, pos_tmp = self.patch_embed_raymap(inputs[input_idx], true_shape=true_shape)
                x_list.append(x_raymap)
                if pos is None:
                    pos = pos_tmp
                input_idx += 1
            
            if 'pts3d' in self.projection_features:
                # print("using pts3d")
                x_pts3d, pos_tmp = self.patch_embed_pts3d(inputs[input_idx], true_shape=true_shape)
                x_list.append(x_pts3d)
                if pos is None:
                    pos = pos_tmp
                input_idx += 1
            
            if 'rgb' in self.projection_features:
                # print("using rgb")
                x_rgb, pos_tmp = self.patch_embed_rgb(inputs[input_idx], true_shape=true_shape)
                x_list.append(x_rgb)
                if pos is None:
                    pos = pos_tmp
                input_idx += 1
            
            if 'conf' in self.projection_features:
                # print("using conf")
                x_conf, pos_tmp = self.patch_embed_conf(inputs[input_idx], true_shape=true_shape)
                x_list.append(x_conf)
                if pos is None:
                    pos = pos_tmp
                input_idx += 1
            
            if 'sam' in self.projection_features:
                # print("using sam")
                x_sam_256, pos_tmp = self.patch_embed_sam_256(inputs[input_idx], true_shape=true_shape)
                x_sam_64, _ = self.patch_embed_sam_64(inputs[input_idx + 1], true_shape=true_shape)
                x_sam_32, _ = self.patch_embed_sam_32(inputs[input_idx + 2], true_shape=true_shape)
                x_list.extend([x_sam_256, x_sam_64, x_sam_32])
                if pos is None:
                    pos = pos_tmp
                input_idx += 3

            if 'sam3' in self.projection_features:
                x_sam3_s0, pos_tmp = self.patch_embed_sam3_s0(inputs[input_idx], true_shape=true_shape)
                x_sam3_s1, _ = self.patch_embed_sam3_s1(inputs[input_idx + 1], true_shape=true_shape)
                x_sam3_s2, _ = self.patch_embed_sam3_s2(inputs[input_idx + 2], true_shape=true_shape)
                x_list.extend([x_sam3_s0, x_sam3_s1, x_sam3_s2])
                if pos is None:
                    pos = pos_tmp
            
            x = torch.cat(x_list, dim=2)
            x = self.patch_embed_proj(x)
        
        
        # Cache original sequence lengths to avoid recomputation
        n_tokens_per_img = x.shape[1]
        pos_seq_len = pos.shape[1]
        n_raymaps = x.shape[0] // B
        x = x.view(B, n_raymaps, n_tokens_per_img, D)
        pos = pos.view(B, n_raymaps, n_tokens_per_img, 2)
        
        
        x_and_mem = torch.cat([x, mem], dim=1)
        pos_and_mem_pos = torch.cat([pos, mem_pos], dim=1)
        x_and_mem_timesteps = torch.cat([timesteps, mem_timesteps], dim=1)
        
        t_emb = self.t_embedder(x_and_mem_timesteps.reshape(-1))
        t_emb = t_emb.view(B, -1, t_emb.shape[-1])
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, n_tokens_per_img, -1)

    
        x_and_mem = x_and_mem.view(B, -1, D)
        pos_and_mem_pos = pos_and_mem_pos.view(B, -1, 2)
      
        t_emb = t_emb.reshape(B, -1, t_emb.shape[-1])
   
        # Process through encoder blocks
        for blk in self.blocks_enc:
            x_and_mem = blk(x_and_mem, pos_and_mem_pos, t_emb)
       
        # Extract results using slicing (no copy needed)
        x_and_mem = x_and_mem.view(B, -1, n_tokens_per_img, D)
        pos_and_mem_pos = pos_and_mem_pos.view(B, -1, n_tokens_per_img, 2)
        x_ray_map = x_and_mem[:, :n_raymaps]
        pos_ray_map = pos_and_mem_pos[:, :n_raymaps]
        
        x = self.proj(x)
        x = self.norm_enc(x)
        return x, pos_ray_map



class Must3rDecoder (nn.Module):
    def __init__(self, 
        img_size=(224, 224),           # input image size
        distill_img_size=512,
        enc_embed_dim=1024,     # encoder feature dimension
        patch_size=16,          # encoder patch_size
        embed_dim=768,      # decoder feature dimension
        output_dim=1792,      # 16*16*7
        depth=12,           # decoder depth 
        num_heads=12,       # decoder number of heads in the transformer block
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,      # activation layer in the mlp
        pos_embed='RoPE100',
        landscape_only=True,
        head='Linear',
        feedback_type=None,
        memory_mode="norm_y",  # 3 choices, norm_y, kv and raw
        pointmaps_activation=ActivationType.NORM_EXP,
        freeze="encoder",
        pred_pose=True,
        use_ray_map=True,
        pred_rgb=True,
        pred_sam_features=False,
        sam_model="SAM2",
        ray_map_encoder_depth=2,
        use_multitask_token=False,
        block_type=CachedDecoderBlock):
        # self.desc_mode = desc_mode
        # self.two_confs = two_confs
        # self.desc_conf_mode = desc_conf_mode
        super().__init__()
        
        self.pointmaps_activation = pointmaps_activation
        print("Must3rDecoder.pointmaps_activation", self.pointmaps_activation)
        self.use_multitask_token = use_multitask_token
        self.pred_rgb = pred_rgb
        self.pred_sam_features = pred_sam_features
        self.sam_model = sam_model
        self.pred_pose = pred_pose
        self._init_projector(enc_embed_dim, embed_dim)
        self._init_pos_embed(img_size, patch_size, embed_dim, num_heads, pos_embed)
        self._init_blocks(block_type, embed_dim, depth, num_heads, mlp_ratio, norm_layer, act_layer,
                          memory_mode=memory_mode)
        self._init_feedback_mechanism(embed_dim, depth, feedback_type)
        self._init_head(enc_embed_dim, patch_size, embed_dim, distill_img_size, output_dim, depth, norm_layer, landscape_only, head)
        init_feedback_layers(self.feedback_type, self.feedback_layer)
        # if use_ray_map:
        #     self.raymap_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #     torch.nn.init.normal_(self.raymap_embed, std=.02)
        
        
    def _init_pose_predictor(self, embed_dim):
        self.head_pose = Mlp(in_features=embed_dim, 
                              hidden_features=embed_dim * 4, out_features=7)
        self.pose_token = nn.Parameter(
                torch.randn(1, 1, 1, embed_dim), requires_grad=True
            )
        nn.init.normal_(self.pose_token, std=1e-6)
        

    def _init_projector(self, enc_embed_dim, embed_dim):
        self.feat_embed_enc_to_dec = nn.Linear(enc_embed_dim, embed_dim, bias=True)
        self.image2_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.image2_embed, std=.02)

    def _init_pos_embed(self, img_size, patch_size, embed_dim, num_heads, pos_embed):
        self.max_seq_len = max(img_size) // patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.rope = get_pos_embed(pos_embed)
        

    def _init_blocks(self, block_type, embed_dim, depth, num_heads, mlp_ratio, norm_layer, act_layer, memory_mode):
        if isinstance(block_type, str):
            block_type = eval(block_type)
        self.depth = depth
        self.embed_dim = embed_dim
        self.memory_mode = memory_mode
        self.attn_num_heads = num_heads
        self.blocks_dec = nn.ModuleList([
            block_type(embed_dim, num_heads, self.rope, mlp_ratio, qkv_bias=True,
                       norm_layer=norm_layer, act_layer=act_layer, memory_mode=memory_mode)
            for i in range(depth)])
        
        self.inject_pose_token = nn.ModuleList([
            nn.Sequential(
                Mlp(in_features=embed_dim, out_features=embed_dim),
                # norm_layer(embed_dim)
            )
            for i in range(depth)
        ])

    def _init_feedback_mechanism(self, embed_dim, depth, feedback_type):
        self.feedback_type = feedback_type
        self.feedback_layer, self.feedback_norm = create_feedback_layers(embed_dim, depth, feedback_type)

    def _init_head(self, enc_embed_dim, patch_size, embed_dim, distill_img_size, output_dim, depth, norm_layer, landscape_only, head):
        self.norm_dec = norm_layer(embed_dim)
        if head == 'Linear':
            self.head_dec = LinearHead(embed_dim, output_dim, patch_size)
            if self.use_multitask_token:
                self.pts3d_task_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
            if self.pred_pose:
                self._init_pose_predictor(embed_dim)
            if self.pred_rgb:
                self.head_rgb = LinearHead(embed_dim, 16 * 16 * 3, patch_size, use_mlp=True)
                self._head_wrapper_rgb = transpose_to_landscape(self.head_rgb, activate=landscape_only)
                if self.use_multitask_token:
                    self.rgb_task_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
            if self.pred_sam_features:
                if self.sam_model == "SAM2":
                    self.head_sam = SAMHead(input_dim=embed_dim, img_size=distill_img_size, embed_dim=256, patch_size=patch_size)
                elif self.sam_model == "SAM3":
                    self.head_sam = SAM3Head(input_dim=embed_dim, img_size=distill_img_size, embed_dim=256, patch_size=patch_size)
                else:
                    raise ValueError(f"Unsupported sam_model: {self.sam_model}")
                self._head_wrapper_sam = transpose_to_landscape(self.head_sam, activate=landscape_only)
                if self.use_multitask_token:
                    self.sam_task_token = nn.Parameter(torch.randn(1, 1, embed_dim), requires_grad=True)
        else:
            raise ValueError(f'invalid head {head}')
        self._head_wrapper = transpose_to_landscape(self.head_dec, activate=landscape_only)

    def _compute_prediction_head(self, true_shape, B, nimgs, feats):
        feats[-1] = self.norm_dec(feats[-1])
        # decout = feats
        with torch.autocast("cuda", dtype=torch.float32):
            # decout = [tok.float() for tok in decout]
            last_feat = feats[-1]
            pose_feature, dense_feature = last_feat[:, 0, :].float(), last_feat[:, 1:, :].float()
            

            if self.use_multitask_token:
                decout_pts3d = [dense_feature + self.pts3d_task_token]
            else:
                decout_pts3d = [dense_feature]
            x = self._head_wrapper(decout_pts3d, true_shape.view(B * nimgs, *true_shape.shape[2:]))
            x = x.view(B, nimgs, *x.shape[1:])
                
            
            if self.pred_rgb:
                if self.use_multitask_token:
                    decout_rgb = [dense_feature + self.rgb_task_token]
                else:
                    decout_rgb = [dense_feature]
                x_rgb = self._head_wrapper_rgb(decout_rgb, true_shape.view(B * nimgs, *true_shape.shape[2:]))
                x_rgb = x_rgb.view(B, nimgs, *x_rgb.shape[1:])
                x = torch.cat([x, x_rgb], dim=-1)

            if self.pred_sam_features:
                if self.use_multitask_token:
                    decout_sam = [dense_feature + self.sam_task_token]
                else:
                    decout_sam = [dense_feature]
                sam_head_out = self._head_wrapper_sam(decout_sam, true_shape.view(B * nimgs, *true_shape.shape[2:]))
                if self.sam_model == "SAM3":
                    # SAM3Head returns (feat_s0, feat_s1, feat_s2, pre_neck_feat)
                    feat_s0, feat_s1, feat_s2, pre_neck_feat = sam_head_out
                    # Include pre_neck_feat as 4th element for distillation
                    sam_feats = [v.view(B, nimgs, *v.shape[1:]) for v in (feat_s0, feat_s1, feat_s2, pre_neck_feat)]
                else:
                    # SAMHead returns (image_embed, feat_s1, feat_s0)
                    image_embed, feat_s1, feat_s0 = sam_head_out
                    sam_feats = [v.view(B, nimgs, *v.shape[1:]) for v in (image_embed, feat_s1, feat_s0)]
            else:
                sam_feats = None
            if self.pred_pose:
                pose_out = self.head_pose(pose_feature)
                pose_out = pose_out.view(B, nimgs, -1)
            else:
                pose_out = None
        return x, pose_out, sam_feats
    
    def _get_empty_memory(self, device, current_dtype, B, mem_D):
        current_mem = [torch.zeros((B, 0, mem_D), dtype=current_dtype, device=device) for _ in range(self.depth)]
        current_mem_labels = torch.zeros((B, 0), dtype=torch.int64, device=device)
        mem_nimgs = 0
        mem_protected_imgs = 0
        mem_protected_tokens = 0
        return current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens
    
    def make_mem_mask(self, nimgs, N, Nm, device):
        if isinstance(nimgs, list):
            assert isinstance(N, list)
            tokens_images = [nimg * Ni for nimg, Ni in zip(nimgs, N)]

            Nt = sum(tokens_images)
            mem_masks = [torch.ones((nimg, Nm + Nt), dtype=torch.bool, device=device) for nimg in nimgs]
            offset = 0
            for i, (nimg, Ni) in enumerate(zip(nimgs, N)):
                for j in range(nimg):
                    mem_masks[i][j, Nm + offset + (j * Ni):Nm + offset + ((j + 1) * Ni)] = 0
                offset += nimg * Ni
            return mem_masks
        else:
            mem_mask = torch.ones((1, N), dtype=torch.bool, device=device)
            mem_mask = mem_mask.repeat(nimgs, 1)  # nimgs, N
            mem_mask = torch.block_diag(*mem_mask).view(nimgs, -1)  # nimgs, nimgs * N
            mem_mask = torch.concatenate([torch.zeros((nimgs, Nm), dtype=mem_mask.dtype, device=device),
                                          mem_mask], dim=1)  # nimgs, Nm + nimgs * N
            mem_mask = ~mem_mask
            return mem_mask
    
    
    def apply_time_embed(self, x, timesteps):
        if len(x.shape) == 3:
            Bxnimgs, N, D = x.shape
            x_t = x[:, :, None].clone().transpose(1,2)
            timesteps_t = timesteps.view(Bxnimgs)[..., None].expand(-1, N).contiguous()
            x_t = self.time_embed(x_t, timesteps_t)
            x = x_t.squeeze(1)
            return x
        else:
            B, nimgs, N, D = x.shape
            x_t = x.view(B * nimgs, N, D)[:, :, None].clone().transpose(1,2)
            timesteps_t = timesteps.view(B * nimgs)[..., None].expand(-1, N).contiguous()
            x_t = self.time_embed(x_t, timesteps_t)
            x = x_t.squeeze(1).view(B, nimgs, N, D)
            return x

    def forward(self, x, pos, true_shape, 
                current_mem=None,
                timesteps=None,
                is_raymap=False,
                render=False):
                # ray_map=None, ray_map_mask=None, 
                # x_view_0=None, img_view_0=None):
        if isinstance(x, list):
            raise NotImplementedError
            # multiple ar in this batch
            return self.forward_list(x, pos, true_shape, current_mem, render)

        # assert not self.pred_rgb or (ray_map is not None and ray_map_mask is not None), "ray_map and ray_map_mask must be provided if pred_rgb is True"

        current_dtype = get_current_dtype(x.dtype)
        B, nimgs, N, Denc = x.shape
        feats = [x.view(B * nimgs, N, Denc)]
        x = self.feat_embed_enc_to_dec(feats[0]).view(B, nimgs, N, -1)


        if self.pred_pose:
            # Add pose token and positional encoding
            pose_pos = torch.full((B, nimgs, 1, pos.shape[-1]), 0, device=pos.device, dtype=pos.dtype)
            pos = torch.cat([pose_pos, pos + 1], dim=2) # shift positions by 1 to account for the pose token
            x = torch.cat([self.pose_token.expand(B, nimgs, -1, -1), x], dim=2)
        
        B, nimgs, N, D = x.shape
        mem_D = 2 * D if self.memory_mode == "kv" else D
        assert not render or current_mem is not None

        if current_mem is None:
            assert not is_raymap, "only predict raymap in render mode with memory"
            # initialization
            x[:, 1:] = x[:, 1:] + self.image2_embed.to(current_dtype)
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = \
                self._get_empty_memory(x.device, current_dtype, B, mem_D)
        else:
            
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = current_mem
            x = x + self.image2_embed.to(current_dtype)  # not the reference image / memory
        x = x.view(B * nimgs, N, D)
        pos = pos.view(B * nimgs, N, 2)

        mem = []
        Nm = current_mem[0].shape[1]
        if not render and (Nm > 0 or nimgs > 1):
            # when updating the memory, do not let an image do CA with its own tokens
            # ignore this rule when initializing from only one image
            mem_mask = self.make_mem_mask(nimgs, N, Nm, x.device)
        else:
            mem_mask = None

        new_mem = []
        for i, (blk, current_mem_blk) in enumerate(zip(self.blocks_dec, current_mem)):
            if not render:
                # update the memory for this layer
                xmem = x.view(B, nimgs * N, D)
                new_mem.append(xmem)
                mem_i = torch.concatenate([current_mem_blk, blk.prepare_y(xmem)], dim=1)
            else:
                mem_i = current_mem_blk

            # mem is B, Nmi, D
            # we need B*nimgs, Nmi, D for CA
            if mem_mask is not None:
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1)[:, mem_mask].reshape(
                    B * nimgs, Nm + ((nimgs - 1)) * N, mem_D)
            else:
                Nmi = mem_i.shape[1]
                
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1).reshape(B * nimgs, Nmi, mem_D)

            
            x_pose_token = x[:, :1]
            # apply decoder
            # assert timesteps is not None, "timesteps must be provided"
            # x = self.apply_time_embed(x, timesteps)
            x = blk(x, mem_i, pos, None, inject_pose_token=self.inject_pose_token[i](x_pose_token))
            feats.append(x)
      
        if not render:
            # assert (Nm + nimgs * N) == mem[0].shape[1]
            new_mem = run_feedback_layers(self.feedback_layer, self.feedback_norm, new_mem)

            mem = []
            for i in range(len(new_mem)):
                new_mem_i = self.blocks_dec[i].prepare_y(new_mem[i])
                mem.append(torch.concatenate([current_mem[i], new_mem_i], dim=1))

          
            new_labels = torch.arange(nimgs, dtype=current_mem_labels.dtype, device=current_mem_labels.device).view(
                1, nimgs, 1).repeat(B, 1, N).view(B, N * nimgs) + mem_nimgs
            mem_labels = torch.concatenate([current_mem_labels, new_labels], dim=1)

            mem_nimgs = mem_nimgs + nimgs
            
            # Return updated memory state:
            # mem: List of memory tokens for each decoder layer
            # mem_labels: Labels indicating which image each token corresponds to
            # mem_nimgs: Total number of images processed so far
            # mem_nimgs: Duplicate of above (total images processed)
            # mem_labels.shape[1]: Total number of tokens in memory
            out = (mem, mem_labels, mem_nimgs, mem_nimgs, mem_labels.shape[1])
        else:
            out = (current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens)

        # apply prediction head
        x, pose_out, sam_feats = self._compute_prediction_head(true_shape, B, nimgs, feats)
        # return memory, pointmaps
        return out, x, pose_out, sam_feats

class CausalMust3rDecoder(Must3rDecoder):
    """
    Training class
    """

    def __init__(self,
                 protected_imgs=1,
                 mem_dropout=0.0,
                 dropout_mode='temporary',
                 use_xformers_mask=False,
                 use_mem_mask=False, **kv):
        super().__init__(**kv)
        self._init_dropout(protected_imgs, mem_dropout, dropout_mode)
        self.use_xformers_mask = use_xformers_mask
        self.use_mem_mask = use_mem_mask

    def _init_dropout(self, protected_imgs, mem_dropout, dropout_mode):
        self.protected_imgs = protected_imgs
        self.dropout_mode = dropout_mode
        if dropout_mode == 'permanent':
            self.mem_dropout = MemoryDropoutSelector(mem_dropout)
        elif dropout_mode == 'temporary':
            self.mem_dropout = TemporaryMemoryDropoutSelector(mem_dropout)
        else:
            raise ValueError(f'Invalid dropout mode = {dropout_mode}')

    def make_mem_mask(self, nimgs, N, Nm, device):
        mem_mask = torch.ones((1, N), dtype=torch.bool, device=device)
        mem_mask = mem_mask.repeat(nimgs, 1)  # nimgs, N
        mem_mask = torch.block_diag(*mem_mask).view(nimgs, -1)  # nimgs, nimgs * N
        mem_mask = torch.concatenate([torch.zeros((nimgs, Nm), dtype=mem_mask.dtype, device=device),
                                      mem_mask], dim=1)  # nimgs, Nm + nimgs * N
        mem_mask = ~mem_mask
        return mem_mask

    def make_attn_mask(self, x, B, nimgs, N, mem_nimgs, Nm, mem_not_sel, mem_labels, mem_mask):
        idx = torch.arange(nimgs, device=x.device).view(1, nimgs, 1) + mem_nimgs
        idx = idx.expand(B, -1, mem_labels.shape[-1])  # B, nimgs, Nmem

        mem_labels_view = mem_labels.view(B, 1, -1).expand(-1, nimgs, -1)  # B, nimgs, Nmem
        # do not attend tokens from the same image
        attn_mask = mem_labels_view != idx  # B, nimgs, Nmem

        # only attend tokens of the previous images
        if Nm == 0:  # exception for initialization, let the first image do CA with the second image
            idx = idx.clone()
            idx[:, 0] = idx[:, 0] + 2  # idx for img 0 will become 2
        attn_mask = attn_mask & (mem_labels_view < idx)

        if mem_not_sel is not None:
            # mask dropped out tokens
            for i in range(len(mem_not_sel) - 1):
                mem_not_sel_c = mem_not_sel[i]  # Nmem_out
                mem_not_sel_c = mem_not_sel_c.unsqueeze(0).expand(B, -1)
                attn_mask[:, i] = attn_mask[:, i].scatter(
                    dim=-1, index=mem_not_sel_c, src=torch.zeros_like(mem_not_sel_c, dtype=torch.bool))

        if mem_mask is not None:
            # use mem_mask on attn_mask
            mem_mask_attn = mem_mask.view(1, nimgs, Nm + nimgs * N)
            mem_mask_attn = mem_mask_attn.expand(B, -1, -1)
            attn_mask = attn_mask[mem_mask_attn]

        attn_mask = attn_mask.view(B, nimgs, 1, 1, -1)
        attn_mask = attn_mask.repeat(1, 1, self.attn_num_heads, N, 1)
        attn_mask = attn_mask.reshape(B * nimgs, self.attn_num_heads, N, -1)

        if self.use_xformers_mask:
            current_dtype = get_current_dtype(x.dtype)
            # xformers mask is in an additive mask in float
            # -torch.inf for ignored values, 0 for values we keep
            # you need to ensure memory is aligned by slicing a bigger tensor
            attn_mask = attn_mask.reshape(B * nimgs * self.attn_num_heads, N, -1)
            last_dim = attn_mask.shape[-1]
            last_dim = (last_dim + 7) // 8 * 8
            attn_mask_float = torch.full((B * nimgs * self.attn_num_heads, N, last_dim),
                                         -torch.inf, dtype=current_dtype, device=x.device
                                         )[:, :, :attn_mask.shape[-1]]
            attn_mask_float[attn_mask] = 0
            attn_mask = attn_mask_float
        return attn_mask

    def forward(self, x, pos, true_shape, current_mem=None, timesteps=None, render=False):
        current_dtype = get_current_dtype(x.dtype)
        # project encoder features to the correct dimension
        B, nimgs, N, Denc = x.shape
        feats = [x.view(B * nimgs, N, Denc)]
        x = self.feat_embed_enc_to_dec(feats[0]).view(B, nimgs, N, -1)

        if self.pred_pose:
            # Add pose token and positional encoding
            pose_pos = torch.full((B, nimgs, 1, pos.shape[-1]), 0, device=pos.device, dtype=pos.dtype)
            pos = torch.cat([pose_pos, pos + 1], dim=2) # shift positions by 1 to account for the pose token
            x = torch.cat([self.pose_token.expand(B, nimgs, 1, -1), x], dim=2)
        
        B, nimgs, N, D = x.shape
        mem_D = 2 * D if self.memory_mode == "kv" else D
        # render=True means we do not update the memory
        assert not render or current_mem is not None

        if current_mem is None:
            # initialization
            x[:, 1:] = x[:, 1:] + self.image2_embed.to(current_dtype)
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = \
                self._get_empty_memory(x.device, current_dtype, B, mem_D)
        else:
            current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = current_mem
            x = x + self.image2_embed.to(current_dtype)  # not the reference image / memory
            

        
        # protected tokens will not be dropped out
        if not render:
            current_mem_protected_imgs = mem_protected_imgs
            mem_protected_imgs = min(self.protected_imgs, current_mem_protected_imgs + nimgs)
            mem_protected_tokens = mem_protected_tokens + (mem_protected_imgs - current_mem_protected_imgs) * N

        x = x.view(B * nimgs, N, D)
        pos = pos.view(B * nimgs, N, 2)

        Nm = current_mem[0].shape[1]  # number of memory tokens at the previous step

        mem_sel = None
        mem_not_sel = None
        active_mem = current_mem
        if not render and self.mem_dropout.p > 0.0:
            # random token dropout, efficient for training
            mem_sel, mem_not_sel = self.mem_dropout(Nm, nimgs, N, protected=mem_protected_tokens, device=x.device)
        elif render and self.mem_dropout.p > 0.0 and self.dropout_mode == 'temporary':
            new_mem_tokens = 0
            mem_sel, mem_not_sel = self.mem_dropout(Nm, 1, new_mem_tokens, protected=mem_protected_tokens,
                                                    device=x.device)

            # dropout mem here
            active_mem = [mem_i[:, mem_sel[0]] for mem_i in current_mem]
            mem_sel, mem_not_sel = None, None
            Nm = active_mem[0].shape[1]  # number of memory tokens at the previous step

        if not render:
            # prepare labels for the new memory tokens
            new_labels = torch.arange(nimgs, dtype=current_mem_labels.dtype, device=current_mem_labels.device).view(
                1, nimgs, 1).repeat(B, 1, N).view(B, N * nimgs) + mem_nimgs
            mem_labels = torch.concatenate([current_mem_labels, new_labels], dim=1)
        else:
            mem_labels = current_mem_labels

        if mem_sel is not None and self.dropout_mode == 'permanent':
            # select the new memory labels after dropout
            mem_labels_out = mem_labels[:, mem_sel[-1]]
        else:
            mem_labels_out = mem_labels

        mem_mask = None
        attn_mask = None
        if not render and (Nm > 0 or nimgs > 1):
            # when updating the memory, do not let an image do CA with its own tokens
            # ignore this rule when initializing from only one image
            if self.use_mem_mask:
                # physically remove the self attending memory tokens
                mem_mask = self.make_mem_mask(nimgs, N, Nm, x.device)
            # create mask for the cross attention
            attn_mask = self.make_attn_mask(x, B, nimgs, N, mem_nimgs, Nm, mem_not_sel, mem_labels, mem_mask)
            
        new_mem = []
        for i, (blk, current_mem_blk) in enumerate(zip(self.blocks_dec, active_mem)):
            if not render:
                # update the memory for this layer
                xmem = x.view(B, nimgs * N, D)
                new_mem.append(xmem)
                mem_i = torch.concatenate([current_mem_blk, blk.prepare_y(xmem)], dim=1)
            else:
                mem_i = current_mem_blk

            # mem is B, Nmi, D
            # we need B*nimgs, Nmi, D for CA
            if mem_mask is not None:
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1)
                mem_i = mem_i[:, mem_mask]
                mem_i = mem_i.reshape(B * nimgs, Nm + ((nimgs - 1)) * N, mem_D)
            else:
                Nmi = mem_i.shape[1]
                mem_i = mem_i.unsqueeze(1).expand(-1, nimgs, -1, -1).reshape(B * nimgs, Nmi, mem_D)
            
            # apply decoder
            # x = blk(x, mem_i, pos, None, ca_attn_mask=attn_mask)
            x_pose_token = x[:, :1]
            x = blk(x, mem_i, pos, None, 
                ca_attn_mask=attn_mask,
                inject_pose_token=self.inject_pose_token[i](x_pose_token))
            feats.append(x)

        if not render:
            new_mem = run_feedback_layers(self.feedback_layer, self.feedback_norm, new_mem)
            mem = []
            for i in range(len(new_mem)):
                new_mem_i = self.blocks_dec[i].prepare_y(new_mem[i])
                mem.append(torch.concatenate([current_mem[i], new_mem_i], dim=1))
            if mem_sel is not None and self.dropout_mode == 'permanent':
                mem = [mem_i[:, mem_sel[-1]] for mem_i in mem]
            mem_nimgs = mem_nimgs + nimgs
            out = (mem, mem_labels_out, mem_nimgs, mem_protected_imgs, mem_protected_tokens)
        else:
            out = (current_mem, current_mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens)

        # apply prediction head
        x, pose_out = self._compute_prediction_head(true_shape, B, nimgs, feats)
        # return memory, pointmaps
        return out, x, pose_out


class Must3r (nn.Module):
    def __init__(self, img_size=(512, 512), enc_embed_dim=1024, 
                 feedback_type='single_mlp', memory_mode='kv', 
                 embed_dim=768,
                 freeze="encoder"):
        # self.desc_mode = desc_mode
        # self.two_confs = two_confs
        # self.desc_conf_mode = desc_conf_mode
        super().__init__()
        self.pointmaps_activation = ActivationType.NORM_EXP
        

        self.encoder = Dust3rEncoder()
        self.decoder = Must3rDecoderOriginal(img_size=img_size, enc_embed_dim=enc_embed_dim, 
                                     embed_dim=embed_dim, 
                                     feedback_type=feedback_type, memory_mode=memory_mode)
            
        self.head_pose = Mlp(in_features=embed_dim, 
                              hidden_features=embed_dim * 4, out_features=7)
   
        self.pose_token = nn.Parameter(
                torch.randn(1, 3, embed_dim), requires_grad=True
            )
        nn.init.normal_(self.pose_token, std=1e-6)

        self.time_embed = RoPE1D()
        # self.time_embed_torch = RoPE1D_torch()

        self.set_freeze(freeze)

 

    def from_pretrained(self, pretrained_model_name_or_path):
        ckpt = torch.load(pretrained_model_name_or_path, 
            weights_only=False,
            map_location="cpu")
        print("Loading MUSt3R checkpoint")
        encoder_state_dict = ckpt['encoder']
        incompatible_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
        print(incompatible_keys)
        # encoder_state_dict = {k.replace('blocks_enc.', '').replace('norm_enc', 'enc_norm'): v for k, v in encoder_state_dict.items()}
        incompatible_keys = self.decoder.load_state_dict(ckpt['decoder'], strict=False)
        print(incompatible_keys)
       
        del ckpt


    


    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        # to_be_frozen = {
        #     'none':     [],
        #     'mask':     [self.mask_token],
        #     'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        #     'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        # }
        to_be_frozen = {
            'none':     [],
            'encoder':  [self.encoder],
            'encoder_and_decoder': [self.encoder, self.decoder],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')


    def _decoder(self, f1, pos1, f2, pos2, t1, t2, shape1, shape2):
        f = torch.stack([f1, f2], dim=1)
        pos = torch.stack([pos1, pos2], dim=1)
        shape = torch.stack([shape1, shape2], dim=1)
      
        out, x = self.decoder(f, pos, shape)
       
        return out, x



    def _decoder_forecast(self, f1, pos1, f2, pos2, t1, t2, t_forecast):
        f_forecast, pos_forecast = (f2 + f1)/2, pos2 

        final_output = [(f1, f2, f_forecast)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder.feat_embed_enc_to_dec(f1) # B, n_patches, D
        f2 = self.decoder.feat_embed_enc_to_dec(f2) + self.decoder.image2_embed # B, n_patches, D
        f_forecast = self.decoder.feat_embed_enc_to_dec(f_forecast) # B, n_patches, D

        f_pose = self.pose_token.expand(f1.shape[0], -1, -1)
        # Create a special position for the pose token at (0,0,0) - outside the normal patch grid
        pose_pos = torch.full((f1.shape[0], 1, pos1.shape[-1]), 0, device=pos1.device, dtype=pos1.dtype)
        
        f1 = torch.cat([f_pose[:, 0:1], f1], dim=1)
        f2 = torch.cat([f_pose[:, 1:2], f2], dim=1)
        f_forecast = torch.cat([f_pose[:, 2:3], f_forecast], dim=1)
        
        # shift positions by 1 to account for the pose token
        pos1, pos2, pos_forecast = pos1 + 1, pos2 + 1, pos_forecast + 1 
        pos1 = torch.cat([pose_pos, pos1], dim=1)
        pos2 = torch.cat([pose_pos, pos2], dim=1)
        pos_forecast = torch.cat([pose_pos, pos_forecast], dim=1)
        

        final_output.append((f1, f2, f_forecast))
        for blk1 in self.decoder.blocks_dec:
            _f1, _f2, _f_forecast = final_output[-1]
    
            # img1 side
            _f1_cond = _f2
            _pos1_cond = pos2
            f1 = blk1(_f1, blk1.prepare_y(_f1_cond), pos1, _pos1_cond)

            # img2 side
            _f2_cond = _f1
            _pos2_cond = pos1
            f2 = blk1(_f2, blk1.prepare_y(_f2_cond), pos2, _pos2_cond)
            
            # forecast side
            t1_expanded = t1[:, None].expand(-1, _f1.shape[1]).contiguous()
            t2_expanded = t2[:, None].expand(-1, _f2.shape[1]).contiguous()
            t_forecast_expanded = t_forecast[:, None].expand(-1, _f_forecast.shape[1]).contiguous()

            _f1_time = self.time_embed(_f1[:, :, None].clone().transpose(1,2), t1_expanded)
            # _f1_time_torch = self.time_embed_torch(_f1[:, :, None].clone().transpose(1,2), t1_expanded)
            # print("max diff", (_f1_time - _f1_time_torch).abs().max().item())
            _f2_time = self.time_embed(_f2[:, :, None].clone().transpose(1,2), t2_expanded)
            # _f2_time_torch = self.time_embed_torch(_f2[:, :, None].clone().transpose(1,2), t2_expanded)
            # print("max diff", (_f2_time - _f2_time_torch).abs().max().item())
            _f_forecast_time = self.time_embed(_f_forecast[:, :, None].clone().transpose(1,2), t_forecast_expanded)
            # _f_forecast_time_torch = self.time_embed_torch(_f_forecast[:, :, None].clone().transpose(1,2), t_forecast_expanded)
            # print("max diff", (_f_forecast_time - _f_forecast_time_torch).abs().max().item())
            
            _f1_time = _f1_time.squeeze(1)
            _f2_time = _f2_time.squeeze(1)
            _f_forecast_time = _f_forecast_time.squeeze(1)

            _f_forecast_cond = torch.cat([_f1_time, _f2_time], dim=1)
            _pos_forecast_cond = torch.cat([pos1, pos2], dim=1)

            f_forecast = blk1(_f_forecast_time, blk1.prepare_y(_f_forecast_cond), pos_forecast, _pos_forecast_cond)

            # store the result
            final_output.append((f1, f2, f_forecast))
        
        
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.decoder.norm_dec, final_output[-1]))

        return zip(*final_output)

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos = self.encoder(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))

            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos = self.encoder(img1, true_shape1)
            out2, pos2 = self.encoder(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]


        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)


    def forward(self, view1, view2, t_forecast=None):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        t1, t2 = view1['timestep'], view2['timestep']
        shape1 = view1['true_shape']
        shape2 = view2['true_shape']
        # combine all ref images into object-centric representation
        # dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        _, pointmaps = self._decoder(feat1, pos1, feat2, pos2, t1, t2, shape1, shape2)
       
        pointmaps_activation = ActivationType.NORM_EXP
        res1 = self.postprocess(pointmaps[:, 0], pointmaps_activation=pointmaps_activation)
        res2 = self.postprocess(pointmaps[:, 1], pointmaps_activation=pointmaps_activation)
        
        # res1['pts3d'] /= 20
        # res2['pts3d'] /= 20
        
        # save_depth_as_colored_png(res1['pts3d'].cpu().numpy()[0][..., 2], 0, 'demo_data/pts3d_view1.png')
        # breakpoint()
        # save_depth_as_colored_png(res2['pts3d'].cpu().numpy()[0][..., 2], 0, 'demo_data/pts3d_view2.png')
        # res1['pose'] = self.head_pose(pose_feat0)
        # res2['pose'] = self.head_pose(pose_feat1)
        
        # pts1_with_conf = torch.cat([res1['pts3d'], res1['conf'][..., None]], dim=-1)
        # pts2_with_conf = torch.cat([res2['pts3d'], res2['conf'][..., None]], dim=-1)
        # save_pcd_as_txt("demo_data/pts3d_view1.txt", pts1_with_conf.cpu().numpy()[0].reshape(-1, 4))
        # save_pcd_as_txt("demo_data/pts3d_view2.txt", pts2_with_conf.cpu().numpy()[0].reshape(-1, 4))
      
        # feat1, feat2: B, 288, 1024
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2, None


    def forward_forecast(self, view1, view2, t_forecast=None):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        t1, t2 = view1['timestep'], view2['timestep']
        # combine all ref images into object-centric representation
        # dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        dec1, dec2, dec_forecast = self._decoder_forecast(feat1, pos1, feat2, pos2, t1, t2, t_forecast)
        pose_feat0, pose_feat1, pose_feat2 = dec1[-1][:, 0, :], dec2[-1][:, 0, :], dec_forecast[-1][:, 0, :]
        dec1, dec2, dec_forecast = list(dec1), list(dec2), list(dec_forecast)
        for i in range(1, len(dec1)):
            dec1[i] = dec1[i][:, 1:, :]
            dec2[i] = dec2[i][:, 1:, :]
            dec_forecast[i] = dec_forecast[i][:, 1:, :]

        with torch.autocast("cuda", dtype=torch.float32):
            x1 = self.decoder._head_wrapper([tok.float() for tok in dec1], shape1)
            x2 = self.decoder._head_wrapper([tok.float() for tok in dec2], shape2)
            x3 = self.decoder._head_wrapper([tok.float() for tok in dec_forecast], shape2)

            # res1_semantic = self._downstream_head('_semantic1', [tok.float() for tok in dec1], shape1)
            # res2_semantic = self._downstream_head('_semantic2', [tok.float() for tok in dec2], shape2)

        pointmaps_activation = ActivationType.NORM_EXP
        res1 = self.postprocess(x1, pointmaps_activation=pointmaps_activation)
        res2 = self.postprocess(x2, pointmaps_activation=pointmaps_activation)
        res3 = self.postprocess(x3, pointmaps_activation=pointmaps_activation)
        res1['pose'] = self.head_pose(pose_feat0)
        res2['pose'] = self.head_pose(pose_feat1)
        res3['pose'] = self.head_pose(pose_feat2)

     
        # feat1, feat2: B, 288, 1024
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        res3['pts3d_in_other_view'] = res3.pop('pts3d') # predict forecast's pts3d in view1's frame
        return res1, res2, res3

    # def postprocess(self, pointmaps, pointmaps_activation=ActivationType.NORM_EXP, compute_cam=False):
    #     out = {}
    #     channels = pointmaps.shape[-1]
    #     out['pts3d'] = pointmaps[..., :3]
    #     out['pts3d'] = apply_activation(out['pts3d'], activation=pointmaps_activation)
    #     if channels >= 6:
    #         out['pts3d_local'] = pointmaps[..., 3:6]
    #         out['pts3d_local'] = apply_activation(out['pts3d_local'], activation=pointmaps_activation)
    #     if channels == 4 or channels == 7:
    #         out['conf'] = 1.0 + pointmaps[..., -1].exp()

    #     if compute_cam:
    #         H, W = out['conf'].shape[-2:]
    #         pp = torch.tensor((W / 2, H / 2), device=out['pts3d'].device)
    #         focal = estimate_focal_knowing_depth(out['pts3d_local'], pp, focal_mode='weiszfeld')
    #         out['focal'] = focal

    #         batch_dims = out['pts3d'].shape[:-3]
    #         num_batch_dims = len(batch_dims)
    #         R, T = roma.rigid_points_registration(
    #             out['pts3d_local'].reshape(*batch_dims, -1, 3),
    #             out['pts3d'].reshape(*batch_dims, -1, 3),
    #             weights=out['conf'].reshape(*batch_dims, -1) - 1.0, compute_scaling=False)

    #         c2w = torch.eye(4, device=out['pts3d'].device)
    #         c2w = c2w.view(*([1] * num_batch_dims), 4, 4).repeat(*batch_dims, 1, 1)
    #         c2w[..., :3, :3] = R
    #         c2w[..., :3, 3] = T.view(*batch_dims, 3)
    #         out['c2w'] = c2w
    #     return out

class OccAnyMust3r (nn.Module):
    def __init__(self, img_size=(512, 512), enc_embed_dim=1024, 
                 feedback_type='single_mlp', memory_mode='kv', 
                 embed_dim=768,
                 freeze="encoder"):
        super().__init__()
        

        self.encoder = Dust3rEncoder()
        self.decoder = Must3rDecoder(img_size=img_size, enc_embed_dim=enc_embed_dim, 
                                     embed_dim=embed_dim, 
                                     feedback_type=feedback_type, memory_mode=memory_mode)
        self.pointmaps_activation = ActivationType.LINEAR
        print("self.pointmaps_activation", self.pointmaps_activation)
        self.head_pose = Mlp(in_features=embed_dim, 
                              hidden_features=embed_dim * 4, out_features=7)
   
        self.pose_token = nn.Parameter(
                torch.randn(1, 3, embed_dim), requires_grad=True
            )
        nn.init.normal_(self.pose_token, std=1e-6)

        self.time_embed = RoPE1D()
        self.time_embed_torch = RoPE1D_torch()

        self.set_freeze(freeze)

 

    def from_pretrained(self, pretrained_model_name_or_path, load_must3r=True):
        if load_must3r:
            ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
            print("Loading MUSt3R checkpoint")
            encoder_state_dict = ckpt['encoder']
            incompatible_keys = self.encoder.load_state_dict(encoder_state_dict, strict=False)
            print(incompatible_keys)
            # encoder_state_dict = {k.replace('blocks_enc.', '').replace('norm_enc', 'enc_norm'): v for k, v in encoder_state_dict.items()}
            incompatible_keys = self.decoder.load_state_dict(ckpt['decoder'], strict=False)
            print(incompatible_keys)
        
            del ckpt
        else:
            ckpt = torch.load(pretrained_model_name_or_path, map_location="cpu")
            state_dict = ckpt['model']
            self.load_state_dict(state_dict, strict=True)
            del ckpt                        


    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        # to_be_frozen = {
        #     'none':     [],
        #     'mask':     [self.mask_token],
        #     'encoder':  [self.mask_token, self.patch_embed, self.enc_blocks],
        #     'encoder_and_decoder': [self.mask_token, self.patch_embed, self.enc_blocks, self.dec_blocks, self.dec_blocks2],
        # }
        to_be_frozen = {
            'none':     [],
            'encoder':  [self.encoder],
            'encoder_and_decoder': [self.encoder, self.decoder],
        }
        freeze_all_params(to_be_frozen[freeze])
        print(f'Freezing {freeze} parameters')


    def _decoder(self, f1, pos1, f2, pos2, t1, t2, shape1, shape2):
        f = torch.stack([f1, f2], dim=1)
        pos = torch.stack([pos1, pos2], dim=1)
        shape = torch.stack([shape1, shape2], dim=1)
      
        out, x = self.decoder(f, pos, shape)
       
        return out, x



    def _decoder_forecast(self, f1, pos1, f2, pos2, t1, t2, t_forecast):
        f_forecast, pos_forecast = (f2 + f1)/2, pos2 

        final_output = [(f1, f2, f_forecast)]  # before projection
        original_D = f1.shape[-1]

        # project to decoder dim
        f1 = self.decoder.feat_embed_enc_to_dec(f1) # B, n_patches, D
        f2 = self.decoder.feat_embed_enc_to_dec(f2) + self.decoder.image2_embed # B, n_patches, D
        f_forecast = self.decoder.feat_embed_enc_to_dec(f_forecast) # B, n_patches, D

        f_pose = self.pose_token.expand(f1.shape[0], -1, -1)
        # Create a special position for the pose token at (0,0,0) - outside the normal patch grid
        pose_pos = torch.full((f1.shape[0], 1, pos1.shape[-1]), 0, device=pos1.device, dtype=pos1.dtype)
        
        f1 = torch.cat([f_pose[:, 0:1], f1], dim=1)
        f2 = torch.cat([f_pose[:, 1:2], f2], dim=1)
        f_forecast = torch.cat([f_pose[:, 2:3], f_forecast], dim=1)
        
        # shift positions by 1 to account for the pose token
        pos1, pos2, pos_forecast = pos1 + 1, pos2 + 1, pos_forecast + 1 
        pos1 = torch.cat([pose_pos, pos1], dim=1)
        pos2 = torch.cat([pose_pos, pos2], dim=1)
        pos_forecast = torch.cat([pose_pos, pos_forecast], dim=1)
        

        final_output.append((f1, f2, f_forecast))
        for blk1 in self.decoder.blocks_dec:
            _f1, _f2, _f_forecast = final_output[-1]
    
            # img1 side
            _f1_cond = _f2
            _pos1_cond = pos2
            f1 = blk1(_f1, blk1.prepare_y(_f1_cond), pos1, _pos1_cond)

            # img2 side
            _f2_cond = _f1
            _pos2_cond = pos1
            f2 = blk1(_f2, blk1.prepare_y(_f2_cond), pos2, _pos2_cond)
            
            # forecast side
            t1_expanded = t1[:, None].expand(-1, _f1.shape[1]).contiguous()
            t2_expanded = t2[:, None].expand(-1, _f2.shape[1]).contiguous()
            t_forecast_expanded = t_forecast[:, None].expand(-1, _f_forecast.shape[1]).contiguous()

            _f1_time = self.time_embed(_f1[:, :, None].clone().transpose(1,2), t1_expanded)
            # _f1_time_torch = self.time_embed_torch(_f1[:, :, None].clone().transpose(1,2), t1_expanded)
            # print("max diff", (_f1_time - _f1_time_torch).abs().max().item())
            _f2_time = self.time_embed(_f2[:, :, None].clone().transpose(1,2), t2_expanded)
            # _f2_time_torch = self.time_embed_torch(_f2[:, :, None].clone().transpose(1,2), t2_expanded)
            # print("max diff", (_f2_time - _f2_time_torch).abs().max().item())
            _f_forecast_time = self.time_embed(_f_forecast[:, :, None].clone().transpose(1,2), t_forecast_expanded)
            # _f_forecast_time_torch = self.time_embed_torch(_f_forecast[:, :, None].clone().transpose(1,2), t_forecast_expanded)
            # print("max diff", (_f_forecast_time - _f_forecast_time_torch).abs().max().item())
            
            _f1_time = _f1_time.squeeze(1)
            _f2_time = _f2_time.squeeze(1)
            _f_forecast_time = _f_forecast_time.squeeze(1)

            _f_forecast_cond = torch.cat([_f1_time, _f2_time], dim=1)
            _pos_forecast_cond = torch.cat([pos1, pos2], dim=1)

            f_forecast = blk1(_f_forecast_time, blk1.prepare_y(_f_forecast_cond), pos_forecast, _pos_forecast_cond)

            # store the result
            final_output.append((f1, f2, f_forecast))
        
        
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = tuple(map(self.decoder.norm_dec, final_output[-1]))

        return zip(*final_output)

    def _encode_image_pairs(self, img1, img2, true_shape1, true_shape2):
        if img1.shape[-2:] == img2.shape[-2:]:
            out, pos = self.encoder(torch.cat((img1, img2), dim=0),
                                             torch.cat((true_shape1, true_shape2), dim=0))

            out, out2 = out.chunk(2, dim=0)
            pos, pos2 = pos.chunk(2, dim=0)
        else:
            out, pos = self.encoder(img1, true_shape1)
            out2, pos2 = self.encoder(img2, true_shape2)
        return out, out2, pos, pos2

    def _encode_symmetrized(self, view1, view2):
        img1 = view1['img']
        img2 = view2['img']
        B = img1.shape[0]


        # Recover true_shape when available, otherwise assume that the img shape is the true one
        shape1 = view1.get('true_shape', torch.tensor(img1.shape[-2:])[None].repeat(B, 1))
        shape2 = view2.get('true_shape', torch.tensor(img2.shape[-2:])[None].repeat(B, 1))

        # warning! maybe the images have different portrait/landscape orientations
        if is_symmetrized(view1, view2):
            # computing half of forward pass!'
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1[::2], img2[::2], shape1[::2], shape2[::2])
            feat1, feat2 = interleave(feat1, feat2)
            pos1, pos2 = interleave(pos1, pos2)
        else:
            feat1, feat2, pos1, pos2 = self._encode_image_pairs(img1, img2, shape1, shape2)

        return (shape1, shape2), (feat1, feat2), (pos1, pos2)


    def forward(self, view1, view2, t_forecast=None):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        t1, t2 = view1['timestep'], view2['timestep']
        shape1 = view1['true_shape']
        shape2 = view2['true_shape']
        # combine all ref images into object-centric representation
        # dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        _, pointmaps = self._decoder(feat1, pos1, feat2, pos2, t1, t2, shape1, shape2)
       
        # pointmaps_activation = self.decoder.pointmaps_activation
        
        
        res1 = self.postprocess(pointmaps[:, 0], pointmaps_activation=self.pointmaps_activation)
        res2 = self.postprocess(pointmaps[:, 1], pointmaps_activation=self.pointmaps_activation)
        
        # res1['pts3d'] /= 20
        # res2['pts3d'] /= 20
        
        # save_depth_as_colored_png(res1['pts3d'].cpu().numpy()[0][..., 2], 0, 'demo_data/pts3d_view1.png')
        # breakpoint()
        # save_depth_as_colored_png(res2['pts3d'].cpu().numpy()[0][..., 2], 0, 'demo_data/pts3d_view2.png')
        # res1['pose'] = self.head_pose(pose_feat0)
        # res2['pose'] = self.head_pose(pose_feat1)
        
        # pts1_with_conf = torch.cat([res1['pts3d'], res1['conf'][..., None]], dim=-1)
        # pts2_with_conf = torch.cat([res2['pts3d'], res2['conf'][..., None]], dim=-1)
        # save_pcd_as_txt("demo_data/pts3d_view1.txt", pts1_with_conf.cpu().numpy()[0].reshape(-1, 4))
        # save_pcd_as_txt("demo_data/pts3d_view2.txt", pts2_with_conf.cpu().numpy()[0].reshape(-1, 4))
      
        # feat1, feat2: B, 288, 1024
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        return res1, res2


    def forward_forecast(self, view1, view2, t_forecast=None):
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(view1, view2)
        
        t1, t2 = view1['timestep'], view2['timestep']
        # combine all ref images into object-centric representation
        # dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        dec1, dec2, dec_forecast = self._decoder_forecast(feat1, pos1, feat2, pos2, t1, t2, t_forecast)
        pose_feat0, pose_feat1, pose_feat2 = dec1[-1][:, 0, :], dec2[-1][:, 0, :], dec_forecast[-1][:, 0, :]
        dec1, dec2, dec_forecast = list(dec1), list(dec2), list(dec_forecast)
        for i in range(1, len(dec1)):
            dec1[i] = dec1[i][:, 1:, :]
            dec2[i] = dec2[i][:, 1:, :]
            dec_forecast[i] = dec_forecast[i][:, 1:, :]

        with torch.autocast("cuda", dtype=torch.float32):
            x1 = self.decoder._head_wrapper([tok.float() for tok in dec1], shape1)
            x2 = self.decoder._head_wrapper([tok.float() for tok in dec2], shape2)
            x3 = self.decoder._head_wrapper([tok.float() for tok in dec_forecast], shape2)

            # res1_semantic = self._downstream_head('_semantic1', [tok.float() for tok in dec1], shape1)
            # res2_semantic = self._downstream_head('_semantic2', [tok.float() for tok in dec2], shape2)

        pointmaps_activation = ActivationType.NORM_EXP
        res1 = self.postprocess(x1, pointmaps_activation=pointmaps_activation)
        res2 = self.postprocess(x2, pointmaps_activation=pointmaps_activation)
        res3 = self.postprocess(x3, pointmaps_activation=pointmaps_activation)
        res1['pose'] = self.head_pose(pose_feat0)
        res2['pose'] = self.head_pose(pose_feat1)
        res3['pose'] = self.head_pose(pose_feat2)

     
        # feat1, feat2: B, 288, 1024
        res2['pts3d_in_other_view'] = res2.pop('pts3d')  # predict view2's pts3d in view1's frame
        res3['pts3d_in_other_view'] = res3.pop('pts3d') # predict forecast's pts3d in view1's frame
        return res1, res2, res3

    def postprocess(self, pointmaps, pointmaps_activation=ActivationType.NORM_EXP, compute_cam=False):
        out = {}
        channels = pointmaps.shape[-1]
        out['pts3d'] = pointmaps[..., :3]
        out['pts3d'] = apply_activation(out['pts3d'], activation=pointmaps_activation)
        if channels >= 6:
            out['pts3d_local'] = pointmaps[..., 3:6]
            out['pts3d_local'] = apply_activation(out['pts3d_local'], activation=pointmaps_activation)
        if channels == 4 or channels == 7:
            out['conf'] = 1.0 + pointmaps[..., -1].exp()

        if compute_cam:
            H, W = out['conf'].shape[-2:]
            pp = torch.tensor((W / 2, H / 2), device=out['pts3d'].device)
            focal = estimate_focal_knowing_depth(out['pts3d_local'], pp, focal_mode='weiszfeld')
            out['focal'] = focal

            batch_dims = out['pts3d'].shape[:-3]
            num_batch_dims = len(batch_dims)
            R, T = roma.rigid_points_registration(
                out['pts3d_local'].reshape(*batch_dims, -1, 3),
                out['pts3d'].reshape(*batch_dims, -1, 3),
                weights=out['conf'].reshape(*batch_dims, -1) - 1.0, compute_scaling=False)

            c2w = torch.eye(4, device=out['pts3d'].device)
            c2w = c2w.view(*([1] * num_batch_dims), 4, 4).repeat(*batch_dims, 1, 1)
            c2w[..., :3, :3] = R
            c2w[..., :3, 3] = T.view(*batch_dims, 3)
            out['c2w'] = c2w
        return out
