# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed WayMo
# dataset at https://github.com/waymo-research/waymo-open-dataset
# See datasets_preprocess/preprocess_waymo.py
# --------------------------------------------------------
from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class WaymoSeqMultiView(BaseSeqDatasetMultiView):
    """Dataset of outdoor street scenes for multiview training."""

    def __init__(self, *args, split, ROOT, seq_pkl_name='seq_sub1_stride9.pkl', **kwargs):
        super().__init__(*args, ROOT=ROOT, seq_pkl_name=seq_pkl_name, **kwargs)
        self.split = split
        self.is_metric_scale = True

        val_scenes = (
            "segment-18331713844982117868_2920_900_2940_900_with_camera_labels.tfrecord",
            "segment-9465500459680839281_1100_000_1120_000_with_camera_labels.tfrecord",
        )
        if self.split is None:
            return
        if self.split == 'train':
            self.select_scene(val_scenes, opposite=True)
        elif self.split == 'val':
            self.select_scene(val_scenes)
        elif self.split == "debug":
            debug_scenes = [
                'segment-2590213596097851051_460_000_480_000_with_camera_labels.tfrecord',
                'segment-15795616688853411272_1245_000_1265_000_with_camera_labels.tfrecord',
                'segment-11004685739714500220_2300_000_2320_000_with_camera_labels.tfrecord',
                'segment-2151482270865536784_900_000_920_000_with_camera_labels.tfrecord',
            ]
            self.select_scene(debug_scenes)
        else:
            raise ValueError(f'bad {self.split=}')
