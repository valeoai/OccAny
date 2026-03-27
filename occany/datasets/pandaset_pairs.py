# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class PandasetSeqMultiView(BaseSeqDatasetMultiView):
    def __init__(self, *args, PANDASET_PREPROCESSED_ROOT, seq_pkl_name='pandaset_seq_video.pkl', **kwargs):
        super().__init__(*args, ROOT=PANDASET_PREPROCESSED_ROOT, seq_pkl_name=seq_pkl_name, **kwargs)
        self.is_metric_scale = True
        val_scenes = ("084", "040")

        if self.split is None:
            return
        if self.split == 'train':
            self.select_scene(val_scenes, opposite=True)
        elif self.split == 'val':
            self.select_scene(val_scenes)
        else:
            raise ValueError(f'bad {self.split=}')
