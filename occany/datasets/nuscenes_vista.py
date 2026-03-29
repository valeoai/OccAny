"""NuScenes Vista dataset for trajectory evaluation.

Loads video sequences and ground-truth trajectories from the Vista NuScenes
annotation JSON files produced by the Vista project.
"""

import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NuscenesVistaDataset(Dataset):
    """Dataset that loads NuScenes image sequences with Vista GT trajectories."""

    def __init__(
        self,
        data_root,
        num_frames=25,
        img_size=(294, 518),
        crop_img_size=None,
        anno_file="vista_nuscenes_anno/nuScenes_val.json",
    ):
        self.data_root = data_root
        self.num_frames = num_frames
        self.img_size = tuple(img_size)
        self.crop_img_size = tuple(crop_img_size) if crop_img_size is not None else self.img_size

        with open(anno_file, "r") as f:
            payload = json.load(f)

        if isinstance(payload, list):
            self.samples = payload
        elif isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            self.samples = payload["samples"]
        else:
            raise ValueError(
                f"Unsupported annotation format in {anno_file}. "
                "Expected a list or a dict with a 'samples' list."
            )

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in annotation file: {anno_file}")

        self.data_root = self._resolve_data_root(self.data_root, self.samples)

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.img_size, antialias=True),
                transforms.CenterCrop(self.crop_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @staticmethod
    def _resolve_data_root(data_root, all_samples):
        if len(all_samples) == 0:
            return data_root

        first_sample = all_samples[0]
        first_frames = first_sample.get("frames", [])
        if len(first_frames) == 0:
            return data_root

        first_rel_path = first_frames[0]
        if os.path.exists(os.path.join(data_root, first_rel_path)):
            return data_root
        if os.path.exists(first_rel_path):
            return ""
        return data_root

    @staticmethod
    def _interpolate_trajectory(traj_xy, num_frames):
        if len(traj_xy) == 0:
            return np.zeros((num_frames, 2), dtype=np.float32)
        if len(traj_xy) == num_frames:
            return traj_xy.astype(np.float32)

        src_indices = np.arange(len(traj_xy), dtype=np.float32) * 5.0
        tgt_indices = np.arange(num_frames, dtype=np.float32)

        traj_full = np.zeros((num_frames, 2), dtype=np.float32)
        for axis in range(2):
            values = traj_xy[:, axis].astype(np.float32, copy=False)
            interp_values = np.interp(tgt_indices, src_indices, values)

            if len(values) > 1:
                delta_src = float(src_indices[-1] - src_indices[-2])
                if delta_src <= 0.0:
                    delta_src = 1.0
                end_slope = float(values[-1] - values[-2]) / delta_src
                right_mask = tgt_indices > src_indices[-1]
                interp_values[right_mask] = values[-1] + (tgt_indices[right_mask] - src_indices[-1]) * end_slope

            traj_full[:, axis] = interp_values
        return traj_full

    @classmethod
    def _extract_gt_trajectory(cls, sample_dict, num_frames):
        raw_traj = np.asarray(sample_dict["traj"], dtype=np.float32).reshape(-1, 2)

        if raw_traj.shape[0] > 1:
            traj_sparse = np.concatenate([np.zeros((1, 2), dtype=np.float32), raw_traj[1:]], axis=0)
        else:
            traj_sparse = np.zeros((1, 2), dtype=np.float32)

        return cls._interpolate_trajectory(traj_sparse, num_frames)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dict = self.samples[idx]
        frame_entries = sample_dict["frames"]
        if len(frame_entries) == 0:
            raise ValueError(f"Sample {idx} has no frames")

        path_list = []
        last_index = len(frame_entries) - 1
        for frame_idx in range(self.num_frames):
            src_idx = min(frame_idx, last_index)
            frame_rel_path = frame_entries[src_idx]
            image_path = (
                frame_rel_path
                if os.path.isabs(frame_rel_path)
                else os.path.join(self.data_root, frame_rel_path)
            )
            if not os.path.exists(image_path):
                raise FileNotFoundError(image_path)
            path_list.append(image_path)

        img_seq = []
        for each_path in path_list:
            with Image.open(each_path) as image:
                img_seq.append(self.transform(image.convert("RGB")))
        images = torch.stack(img_seq)

        gt_traj = self._extract_gt_trajectory(sample_dict, self.num_frames)

        return {
            "images": images,
            "gt_trajectory": torch.from_numpy(gt_traj),
        }
