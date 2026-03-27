# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn as nn

try:
    import xformers.ops
    has_xformers = True
except Exception:
    has_xformers = False

try:
    from torch.nn.functional import scaled_dot_product_attention  # noqa
    has_scaled_dot_product_attention = True
except Exception:
    has_scaled_dot_product_attention = False


_use_memory_efficient_attention = False


def toggle_memory_efficient_attention(enabled: bool = True):
    global _use_memory_efficient_attention
    _use_memory_efficient_attention = enabled
    print(f"Memory efficient attention enabled: {enabled}")


def is_memory_efficient_attention_enabled():
    return _use_memory_efficient_attention


