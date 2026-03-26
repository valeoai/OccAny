import os
from typing import Any, Dict, Optional

from occany.datasets.kitti import KittiDataset, collate_kitti_identity, collate_kitti_metric
from occany.datasets.nuscenes import NuScenesDataset, collate_nuscenes_identity, collate_nuscenes_metric


def _build_nuscenes_scene_token_time_index(scene_info: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    if not scene_info:
        return {}

    remaining_tokens = set(scene_info.keys())
    token_to_time_index: Dict[str, int] = {}
    time_index = 0

    while remaining_tokens:
        pointed_tokens = set()
        for token in remaining_tokens:
            next_token = scene_info[token].get("next")
            if next_token in remaining_tokens:
                pointed_tokens.add(next_token)

        start_tokens = sorted(token for token in remaining_tokens if token not in pointed_tokens)
        if not start_tokens:
            start_tokens = [sorted(remaining_tokens)[0]]

        for start_token in start_tokens:
            if start_token not in remaining_tokens:
                continue

            token = start_token
            while token in remaining_tokens:
                token_to_time_index[token] = time_index
                time_index += 1
                remaining_tokens.remove(token)

                next_token = scene_info[token].get("next")
                if next_token in (None, "", "EOF") or next_token not in remaining_tokens:
                    break
                token = next_token

    return token_to_time_index


def build_nuscenes_vis_time_index_map(dataset: Any) -> Dict[str, Dict[str, int]]:
    annotations = getattr(dataset, "annotations", None)
    if annotations is None:
        return {}

    scene_infos = annotations.get("scene_infos", {})
    if not isinstance(scene_infos, dict):
        return {}

    scene_filter = set(getattr(dataset, "scenes", []))
    scene_time_index_map: Dict[str, Dict[str, int]] = {}
    for scene_name, scene_info in scene_infos.items():
        if scene_filter and scene_name not in scene_filter:
            continue
        if not isinstance(scene_info, dict):
            continue
        scene_time_index_map[scene_name] = _build_nuscenes_scene_token_time_index(scene_info)

    return scene_time_index_map


def prepare_eval_setting(
    dataset: str,
    setting: str,
    boxes_folder: Optional[str] = None,
    output_resolution=None,
    image_size: int = 512,
    novel_view_rgb_path: Optional[str] = None,
    process_id: int = 0,
    num_worlds: int = 1,
    allowed_scenes=None,
    sam3_resolution: int = 1008,
    base_model: str = "must3r",
    split: str = "val",
    kitti_root: Optional[str] = None,
    nuscenes_root: Optional[str] = None,
):
    if output_resolution is None:
        if dataset == 'kitti':
            if base_model == 'da3':
                output_resolution = (518, 168)
            else:
                output_resolution = (512, 160)
        elif dataset == 'nuscenes':
            if base_model == 'da3':
                output_resolution = (518, 294)
            else:
                output_resolution = (512, 288)
        else:
            raise ValueError(f"Dataset {dataset} not supported")

    project_root = os.environ.get('PROJECT', '.')
    scratch_root = os.environ.get('SCRATCH', os.path.join(project_root, 'eval_output'))

    kitti_root = kitti_root or os.path.join(project_root, 'data/kitti')
    kitti_artifacts_root = os.path.join(scratch_root, 'data/kitti_processed')
    nuscenes_root = nuscenes_root or os.path.join(project_root, 'data/nuscenes')
    nuscenes_artifacts_root = os.path.join(scratch_root, 'data/nuscenes_processed')

    if dataset == 'kitti':
        assert setting in ['5frames', '1frame'], f"Setting {setting} is not supported for KITTI"
        if setting == '5frames':
            video_length = 10
            recon_view_idx = [0, 2, 4, 6, 8]
        else:
            video_length = 1
            recon_view_idx = [0]

        boxes_dir = None
        if boxes_folder is not None:
            boxes_dir = os.path.join(kitti_artifacts_root, boxes_folder)
        dataset_obj = KittiDataset(
            split=split,
            boxes_dir=boxes_dir,
            output_resolution=output_resolution,
            semkitti_root=kitti_root,
            kittiodo_root=kitti_root,
            remap_lut_path=os.path.join(os.path.dirname(__file__), 'semantic_kitti.yaml'),
            video_length=video_length,
            video_interval=5,
            frame_interval=5,
            novel_view_rgb_path=novel_view_rgb_path,
            pid=process_id,
            world=num_worlds,
            sam3_resolution=sam3_resolution,
            base_model=base_model,
        )
        collate_fn = collate_kitti_identity
    elif dataset == 'nuscenes':
        if setting == 'surround':
            use_surround_label = True
            frame_interval = 1
            video_length = 1
            recon_view_idx = list(range(6))
            camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        elif setting == '5frames':
            use_surround_label = False
            camera_names = ['CAM_FRONT']
            video_length = 5
            frame_interval = 2
            recon_view_idx = list(range(5))
        else:
            use_surround_label = False
            camera_names = ['CAM_FRONT']
            frame_interval = 1
            video_length = 1
            recon_view_idx = [0]

        boxes_dir = None
        if boxes_folder is not None:
            boxes_dir = os.path.join(nuscenes_artifacts_root, boxes_folder)
        dataset_obj = NuScenesDataset(
            split=split,
            use_surround_label=use_surround_label,
            root=nuscenes_root,
            output_resolution=output_resolution,
            boxes_dir=boxes_dir,
            apply_camera_mask=True,
            apply_lidar_mask=False,
            pid=process_id,
            world=num_worlds,
            video_length=video_length,
            camera_names=camera_names,
            frame_interval=frame_interval,
            novel_view_rgb_path=novel_view_rgb_path,
            allowed_scenes=allowed_scenes,
            base_model=base_model,
        )
        collate_fn = collate_nuscenes_identity
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    return dataset_obj, collate_fn, recon_view_idx


def prepare_metric_eval_setting(
    dataset: str,
    setting: str,
    process_id: int = 0,
    num_worlds: int = 1,
    split: str = "val",
    kitti_root: Optional[str] = None,
    nuscenes_root: Optional[str] = None,
):
    project_root = os.environ.get('PROJECT', '.')
    kitti_root = kitti_root or os.path.join(project_root, 'data/kitti')
    nuscenes_root = nuscenes_root or os.path.join(project_root, 'data/nuscenes')

    if dataset == 'kitti':
        assert setting in ['10frames', '5frames', '1frame'], f"Setting {setting} is not supported for KITTI"
        if setting == '10frames':
            video_length = 10
            recon_view_idx = list(range(10))
        elif setting == '5frames':
            video_length = 10
            recon_view_idx = [0, 2, 4, 6, 8]
        else:
            video_length = 1
            recon_view_idx = [0]

        dataset_obj = KittiDataset(
            split='val' if split == 'vis' else split,
            semkitti_root=kitti_root,
            kittiodo_root=kitti_root,
            remap_lut_path=os.path.join(os.path.dirname(__file__), 'semantic_kitti.yaml'),
            video_length=video_length,
            video_interval=5,
            frame_interval=5,
            pid=process_id,
            world=num_worlds,
        )
        collate_fn = collate_kitti_metric
    elif dataset == 'nuscenes':
        if setting == 'surround':
            use_surround_label = True
            frame_interval = 1
            video_length = 1
            recon_view_idx = list(range(6))
            camera_names = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
        elif setting == '5frames':
            use_surround_label = False
            camera_names = ['CAM_FRONT']
            video_length = 5
            frame_interval = 2
            recon_view_idx = list(range(5))
        else:
            use_surround_label = False
            camera_names = ['CAM_FRONT']
            frame_interval = 1
            video_length = 1
            recon_view_idx = [0]

        dataset_obj = NuScenesDataset(
            split=split,
            use_surround_label=use_surround_label,
            root=nuscenes_root,
            apply_camera_mask=True,
            apply_lidar_mask=False,
            pid=process_id,
            world=num_worlds,
            video_length=video_length,
            camera_names=camera_names,
            frame_interval=frame_interval,
        )
        collate_fn = collate_nuscenes_metric
    else:
        raise ValueError(f"Dataset {dataset} not supported")

    return dataset_obj, collate_fn, recon_view_idx
