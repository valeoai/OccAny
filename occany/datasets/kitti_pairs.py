# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class KittiSeqMultiView(BaseSeqDatasetMultiView):
    """Dataset of outdoor street scenes for multiview training."""

    def __init__(self, *args, KITTI_PREPROCESSED_ROOT, seq_pkl_name='kitti_seq_video.pkl', **kwargs):
        super().__init__(*args, ROOT=KITTI_PREPROCESSED_ROOT, seq_pkl_name=seq_pkl_name, **kwargs)
        self.is_metric_scale = True

        val_scenes = ("08",)
        if self.split is None:
            return
        if self.split == 'train':
            self.select_scene(val_scenes, opposite=True)
        elif self.split in ('val', 'vis'):
            self.select_scene(val_scenes)
        else:
            raise ValueError(f'bad {self.split=}')
