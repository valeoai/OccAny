# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Dataloader for preprocessed Occ3D-nuScenes
# Following the structure from https://github.com/Tsinghua-MARS-Lab/Occ3D
# --------------------------------------------------------
import json
import os

from occany.datasets.base_seq_dataset import BaseSeqDatasetMultiView


class Occ3dNuscenesSeqMultiView(BaseSeqDatasetMultiView):
    """ Dataset of outdoor street scenes from Occ3D-nuScenes, 6 surround cameras
    """

    VIS_SCENES = [
        'scene-0108','scene-0794', 
        'scene-0625', 
    ]

    def __init__(self, *args,
                 NUSCENES_PREPROCESSED_ROOT,
                 seq_pkl_name='seq_surround_surround.pkl', **kwargs):

        super().__init__(*args, ROOT=NUSCENES_PREPROCESSED_ROOT,
                         seq_pkl_name=seq_pkl_name, **kwargs)
        self.is_metric_scale = True
        # self.img_ext = ".jpg"  # Use default from base class (will be overridden by .npz loading)

        if self.split is None:
            return

        if self.split not in ('train', 'val', 'vis'):
            raise ValueError(f"bad split: {self.split}")

        if self.split == 'vis':
            self.select_scene(self.VIS_SCENES)
            return

        split_scenes = self._load_split_scenes(NUSCENES_PREPROCESSED_ROOT, self.split)
        self.select_scene(split_scenes)

    @staticmethod
    def _resolve_annotations_path(preprocessed_root):
        candidates = [
            os.path.join(os.environ.get('NUSCENES_ROOT', ''), 'annotations.json'),
            os.path.join(preprocessed_root, 'annotations.json'),
        ]
        if 'DSDIR' in os.environ:
            candidates.append(os.path.join(os.environ['DSDIR'], 'Occ3D-nuScenes', 'annotations.json'))

        for path in candidates:
            if os.path.exists(path):
                return path

        raise FileNotFoundError(
            'annotations.json not found in preprocessed root or default Occ3D-nuScenes locations'
        )

    @classmethod
    def _load_split_scenes(cls, preprocessed_root, split):
        annotations_path = cls._resolve_annotations_path(preprocessed_root)
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)

        split_key = f'{split}_split'
        if split_key not in annotations:
            raise ValueError(f"Split '{split}' not found in annotations.json")

        return annotations[split_key]


