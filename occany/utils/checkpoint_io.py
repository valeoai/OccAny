from __future__ import annotations

import torch
import torch.distributed as dist


def save_on_master(*args, **kwargs):

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        return torch.save(*args, **kwargs)

    return None
