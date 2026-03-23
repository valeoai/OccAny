# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn
from occany.model.must3r_blocks.attention import Attention, CachedCrossAttention
# import must3r.tools.path_to_dust3r  # noqa
import dust3r.utils.path_to_croco  # noqa: F401
from models.blocks import Mlp, DropPath
import math

MEMORY_MODES = ['norm_y', 'kv', 'raw']

def get_current_dtype(default_dtype, verbose=False):
    current_dtype = default_dtype
    try:
        if torch.is_autocast_cpu_enabled():
            current_dtype = torch.get_autocast_cpu_dtype()
        elif torch.is_autocast_enabled():
            current_dtype = torch.get_autocast_gpu_dtype()
    except Exception:
        pass
    if verbose:
        print(current_dtype)
    return current_dtype


class BaseTransformer(nn.Module):
    def initialize_weights(self):
        # linears and layer norms
        self.apply(self._init_weights)
        self.apply(self._init_override)

    def _init_override(self, m):
        init_weight_override_fun = getattr(m, "_init_weight_override", None)
        if callable(init_weight_override_fun):
            init_weight_override_fun()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)


class Block(nn.Module):
    def __init__(self, dim, num_heads, pos_embed=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        # SA
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, pos_embed=pos_embed, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)
        # MLP
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, xpos=None):
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    


class CachedDecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, pos_embed=None, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, memory_mode="norm_y"):
        super().__init__()
        assert memory_mode in MEMORY_MODES
        self.memory_mode = memory_mode

        # SA
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, pos_embed=pos_embed, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                              proj_drop=drop)

        # CA
        self.norm2 = norm_layer(dim)
        self.norm_y = norm_layer(dim)
        self.cross_attn = CachedCrossAttention(dim, pos_embed=None, num_heads=num_heads, qkv_bias=qkv_bias,
                                               attn_drop=attn_drop, proj_drop=drop)

        # MLP
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def prepare_y(self, y):
        if self.memory_mode == 'raw':
            return y
        y_ = self.norm_y(y)
        if self.memory_mode == 'norm_y':
            return y_.to(y.dtype)
        k, v = self.cross_attn.prepare_kv(y_, y_)
        return torch.concatenate([k, v], dim=-1)

    def forward(self, x, y, xpos=None, ypos=None, ca_attn_mask=None, inject_pose_token=None):
        
        x = x + self.drop_path(self.attn(self.norm1(x), xpos))
        y_ = self.norm_y(y) if self.memory_mode == 'raw' else y
        if self.memory_mode == 'kv':
            key, value = torch.split(y_, x.shape[-1], dim=-1)
        else:
            key, value = self.cross_attn.prepare_kv(y_, y_)
        
        x = x + self.drop_path(self.cross_attn(self.norm2(x), key, value, xpos, ypos, ca_attn_mask))
        
        if inject_pose_token is not None:
            x[:, 1:] = x[:, 1:] + inject_pose_token
        
        x = x + self.drop_path(self.mlp(self.norm3(x)))
        return x

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, 
                 pos_embed=None,
                 mlp_ratio=4.0,
                 use_time_cond=True, 
                 **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, 
                              qkv_bias=True, pos_embed=pos_embed, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.use_time_cond = use_time_cond
        if self.use_time_cond:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        self.pos_embed = pos_embed

    def forward(self, x, pos, c):
        if self.use_time_cond:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=2)
            x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa), pos)
            x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            x = x + self.attn(self.norm1(x), pos)
            x = x + self.mlp(self.norm2(x))
        return x
    
    


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb