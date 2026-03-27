import torch

from croco.utils.misc import get_rank, get_world_size
from dust3r.datasets import *  # noqa: F401,F403

# Evaluation datasets
from .kitti import KittiDataset  # noqa: F401
from .nuscenes import NuScenesDataset  # noqa: F401

# Training datasets
from .base_seq_dataset import BaseSeqDataset, BaseSeqDatasetMultiView  # noqa: F401
from .waymo import WaymoSeqMultiView  # noqa: F401
from .vkitti_pairs import VKittiSeqMultiView  # noqa: F401
from .ddad_pairs import DDADSeqMultiView  # noqa: F401
from .pandaset_pairs import PandasetSeqMultiView  # noqa: F401
from .once_pairs import OnceSeqMultiView  # noqa: F401
from .kitti_pairs import KittiSeqMultiView  # noqa: F401
from .nuscenes_pairs import Occ3dNuscenesSeqMultiView  # noqa: F401


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(
            batch_size,
            shuffle=shuffle,
            world_size=world_size,
            rank=rank,
            drop_last=drop_last,
        )
    except (AttributeError, NotImplementedError):
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=shuffle,
                drop_last=drop_last,
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    return torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )
