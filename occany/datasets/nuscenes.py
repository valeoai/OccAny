import torch
import os
import json
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from tqdm import tqdm
from occany.utils.helpers import crop_resize_if_necessary
from occany.utils.image_util import GroundingDinoImgNorm, ImgNorm, get_SAM2_transforms, get_SAM3_transforms
from depth_anything_3.utils.io.input_processor import InputProcessor
from torchvision.transforms.functional import to_tensor
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes




def collate_nuscenes_identity(batch):
    """
    Collate function for nuScenes dataset

    Args:
        batch: List of dataset items
    """
    data = {
        'imgs': torch.stack([item['imgs'] for item in batch]),
        "gdino_imgs": torch.stack([item['gdino_imgs'] for item in batch]),
        'sam2_imgs': torch.stack([item['sam2_imgs'] for item in batch]),
        'sam3_imgs': torch.stack([item['sam3_imgs'] for item in batch]),
        'begin_frame_token': [item['begin_frame_token'] for item in batch],
        'scene_name': [item['scene_name'] for item in batch],
        'gt_depths': torch.stack([torch.from_numpy(item['gt_depths']) for item in batch]),
        'cam_k_resized': torch.stack([torch.from_numpy(item['cam_k_resized']) for item in batch]),
        'cam_poses_in_cam0': torch.stack([torch.from_numpy(item['cam_poses_in_cam0']) for item in batch]),
        'cam0_to_ego': torch.stack([torch.from_numpy(item['cam0_to_ego']) for item in batch]),
        'camera_masks': torch.stack([torch.from_numpy(item['camera_masks']) for item in batch]),
        'voxel_label': torch.stack([torch.from_numpy(item['voxel_label']) for item in batch]),
        'voxel_mask_camera': torch.stack([torch.from_numpy(item['voxel_mask_camera']) for item in batch]),
        'voxel_mask_lidar': torch.stack([torch.from_numpy(item['voxel_mask_lidar']) for item in batch]),
        'image_paths': [item['image_paths'] for item in batch],
        'box_dicts': [item['box_dicts'] for item in batch],
        'camera_names': [item['camera_names'] for item in batch],
        'lidar_origin': [torch.from_numpy(item['lidar_origin']) for item in batch],  # List of [T, 3] tensors
        'lidar_points': [torch.from_numpy(item['lidar_points']) for item in batch],  # List of [N, 3] tensors (varying N)
    }

    # Handle novel_view_rgbs if present
    if any(item.get('novel_view_rgbs') is not None for item in batch):
        data['novel_view_rgbs'] = torch.stack([item.get('novel_view_rgbs') for item in batch])

    return data


def collate_nuscenes_metric(batch):
    return {
        'begin_frame_token': [item['begin_frame_token'] for item in batch],
        'scene_name': [item['scene_name'] for item in batch],
        'voxel_label': torch.stack([torch.from_numpy(item['voxel_label']) for item in batch]),
    }


class NuScenesDataset(Dataset):
    """
    Dataset for Occ3D nuScenes data
    Following the structure from https://github.com/Tsinghua-MARS-Lab/Occ3D
    """
    def __init__(
        self,
        split,
        root='/path/to/Occ3D-nuScenes',
        color_jitter=None,
        fliplr=0.0,
        output_resolution=(512, 288),
        boxes_dir=None,
        apply_camera_mask=True,
        apply_lidar_mask=False,
        video_length=10,
        camera_names=None,
        use_surround_label=False,
        frame_interval=1,
        pid=0,
        world=1,
        allowed_scenes=None,
        novel_view_rgb_path=None,
        base_model="must3r",
    ):
        """
        Args:
            split: 'train', 'val', 'test'
            root: Path to Occ3D-nuScenes dataset root
            color_jitter: Color jitter parameters
            fliplr: Probability of horizontal flip
            output_resolution: Output image resolution (W, H)
            boxes_dir: Directory containing bounding box predictions
            apply_camera_mask: Whether to apply camera visibility mask
            apply_lidar_mask: Whether to apply lidar visibility mask
            video_length: Number of temporal frames to use (1 = single frame, >1 = temporal)
            frame_interval: Interval between frames (1 = consecutive, 2 = skip 1 frame, etc.)
            camera_names: List of camera names to use. Options:
                - None: defaults to ['CAM_FRONT']
                - ['CAM_FRONT']: Single front camera
                - ['CAM_FRONT', 'CAM_BACK']: Multiple cameras
                - ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
                   'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']: All 6 cameras
            pid: Process ID for distributed processing
            world: Total number of processes for distributed processing
            allowed_scenes: List of allowed scene tokens in format 'scene-XXXX_token'. If None, all scenes are used.
            
        Examples:
            # Single camera, temporal frames (default)
            dataset = NuScenesDataset(split='train', video_length=10)
            # → 1 camera × 10 frames = 10 images
            
            # All 6 cameras, single frame
            dataset = NuScenesDataset(split='train', 
                camera_names=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                             'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                video_length=1)
            # → 6 cameras × 1 frame = 6 images
            
            # Front and back cameras, 10 temporal frames each
            dataset = NuScenesDataset(split='train',
                camera_names=['CAM_FRONT', 'CAM_BACK'],
                video_length=10)
            # → 2 cameras × 10 frames = 20 images
        """
        super().__init__()
        self.base_model = base_model
        self.use_surround_label = use_surround_label
        self.root = root
        self.split = split
        self.allowed_scenes = allowed_scenes
        self.n_classes = 18  # Occ3D has 18 classes (0-16 + free=17)
        self.other_class = 0
        self.fliplr = fliplr
        self.boxes_dir = boxes_dir
        self.output_resolution = output_resolution
        self.novel_view_rgb_path = novel_view_rgb_path
        self.apply_camera_mask = apply_camera_mask
        self.apply_lidar_mask = apply_lidar_mask
        
        # Set camera names - default to single front camera
        if camera_names is None:
            self.camera_names = ['CAM_FRONT']
        else:
            self.camera_names = camera_names

        if use_surround_label:
            assert len(self.camera_names) == 6, "Surround label requires 6 cameras"
        
        self.video_length = video_length
        self.frame_interval = frame_interval
        sam2_output_resolution = min(1024, max(output_resolution))
        self.SAM2_transforms = get_SAM2_transforms(resolution=sam2_output_resolution)
        self.sam3_output_resolution = 1008  # Default SAM3 resolution
        self.SAM3_transforms = get_SAM3_transforms(resolution=self.sam3_output_resolution)
        
        # GaussTR
        self.OCC3D_CATEGORIES = (
            ['other'],
            ['barrier', 'concrete barrier', 'metal barrier', 'water barrier'],
            ['bicycle', 'bicyclist'],
            ['bus'],
            ['car'],
            ['crane'],
            ['motorcycle', 'motorcyclist'],
            ['pedestrian', 'adult', 'child'],
            ['cone'],
            ['trailer'],
            ['truck'],
            ['road'],
            ['traffic island', 'rail track', 'lake', 'river'],
            ['sidewalk'],
            ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
            ['building', 'wall', 'guard rail', 'fence', 'pole', 'drainage', 'hydrant', 'street sign', 'traffic light'],
            ['tree', 'bush'],
            ['sky', 'empty'],
        )
        
        # PROMPT attribute for SAM-based semantic segmentation
        # Follows the same pattern as KittiDataset.PROMPT
        self.PROMPT = list(self.OCC3D_CATEGORIES)
        
        # LOSC
        # self.OCC3D_CATEGORIES = (
        #     ['barrier', 'barricade'],
        #     ['bicycle', 'bicyclist'],
        #     ['bus'],
        #     ['car'],
        #     ['bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck'],
        #     ['motorcycle', 'motorcyclist'],
        #     ['person', 'pedestrian', 'adult', 'child'],
        #     ['traffic cone'],
        #     ['trailer', 'sem trailer', 'cargo container', 'shipping container', 'freight container'],
        #     ['truck'],
        #     ['road'],
        #     ['traffic island', 'rail track', 'lake', 'river'],
        #     ['sidewalk'],
        #     ['terrain', 'grass', 'grassland', 'lawn', 'meadow', 'turf', 'sod'],
        #     ['building', 'wall', 'guard rail', 'fence', 'pole', 'drainage',
        #     'hydrant', 'street sign', 'traffic light', 'awning'],
        #     ['tree', 'trunk', 'tree trunk', 'bush', 'shrub', 'plant', 'flower', 'woods'],
        #     ['sky', 'empty'],
        # )
        self.CLASS_NAMES = [
            "other",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction_vehicle",
            "motorcycle",
            "pedestrian",
            "traffic_cone",
            "trailer",
            "truck",
            "driveable_surface",
            "other_flat",
            "sidewalk",
            "terrain",
            "manmade",
            "vegetation",
            "free",
        ]
        
        # Mapping to superclasses
        self.MAPPING = [
            0, # "other",
            2, # "barrier",
            3, # "bicycle",
            3, # "bus",
            3, # "car",
            3, # "construction_vehicle",
            3, # "motorcycle",
            5, # "pedestrian",
            6, # "traffic_cone",
            3, # "trailer",
            3, # "truck",
            1, # "driveable_surface",
            1, # "other_flat",
            1, # "sidewalk",
            4, # "terrain",
            2, # "manmade",
            4, # "vegetation",
            7 # "free",
        ]
        
        self.SUPERCLASS_NAMES = [
            "other",
            "ground", # 1
            "structure", # 2
            "vehicle", # 3
            "nature", # 4
            "human", # 5
            "object", # 6
            "empty", # 7
         ]
        
        self.superclass_empty_class = 7  # "empty" is at index 7 in SUPERCLASS_NAMES
        
        
        self.COLORS = np.array([
            [0, 0, 0, 255],
            [112, 128, 144, 255],
            [220, 20, 60, 255],
            [255, 127, 80, 255],
            [255, 158, 0, 255],
            [233, 150, 70, 255],
            [255, 61, 99, 255],
            [0, 0, 230, 255],
            [47, 79, 79, 255],
            [255, 140, 0, 255],
            [255, 98, 70, 255],
            [0, 207, 191, 255],
            [175, 0, 75, 255],
            [75, 0, 75, 255],
            [112, 180, 60, 255],
            [222, 184, 135, 255],
            [0, 175, 0, 255],
            [135, 206, 235, 255], # sky, empty
        ])
        self.empty_class = 17
        
        
        
        
        # Occ3D voxel configuration
        self.voxel_size = 0.4  # 0.4m for nuScenes
        self.scene_size = (40.0, 40.0, 6.4)  # meters (X, Y, Z range)
        self.occ_size = [200, 200, 16]  # voxel grid size
        self.pc_range = [-40.0, -40.0, -1.0, 40.0, 40.0, 3.0]  # [x_min, y_min, z_min, x_max, y_max, z_max]
        self.voxel_origin = np.array([self.pc_range[0], self.pc_range[1], self.pc_range[2]])

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        
        # Load annotations
        anno_path = os.path.join(root, 'annotations.json') 
        
        assert os.path.exists(anno_path), f"Annotations file not found at {anno_path}"
        
        with open(anno_path, 'r') as f:
            self.annotations = json.load(f)
        
        print("Initializing NuScenes devkit...")
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=root, verbose=False)
        
        # Get scene list for this split.
        split_key = f'{split}_split'
        if split_key in self.annotations:
            self.scenes = self.annotations[split_key]
        else:
            # If split info not in annotations, use all scenes
            self.scenes = list(self.annotations['scene_infos'].keys())

        self.videos = []
        for scene_name in tqdm(self.scenes, desc=f"Loading {split} scenes"):
            if scene_name not in self.annotations['scene_infos']:
                continue
                
            scene_info = self.annotations['scene_infos'][scene_name]
            for frame_token, frame_info in scene_info.items():
                # Temporal mode: build a video sequence starting from this frame
                video_tokens = [frame_token]  # Include the starting frame
                cur_frame_token = frame_token
                
                # Collect subsequent frames with frame_interval
                while len(video_tokens) < self.video_length:
                    # Skip frames based on frame_interval
                    for _ in range(self.frame_interval):
                        next_token = scene_info[cur_frame_token]['next']
                        if next_token is None or next_token == 'EOF':
                            break
                        cur_frame_token = next_token
                    
                    # Check if we successfully moved to next frame
                    if next_token is None or next_token == 'EOF':
                        break
                    
                    video_tokens.append(cur_frame_token)
                
                # Only add if we have enough frames
                if len(video_tokens) == self.video_length:
                    # Filter by allowed scenes if specified
                    if self.allowed_scenes is not None:
                        scene_token = f"{scene_name}_{frame_token}"
                        if scene_token not in self.allowed_scenes:
                            continue
                    
                    self.videos.append({
                        'scene_name': scene_name,
                        'video_tokens': video_tokens,
                        'begin_frame_token': frame_token
                    })
        
        # Print dataset configuration
        n_cameras = len(self.camera_names)
        n_frames = self.video_length
        images_per_sample = n_cameras * n_frames
        print(f"Loaded {len(self.videos)} samples from {len(self.scenes)} scenes for {split} split")
        print(f"Configuration: {n_cameras} camera(s) × {n_frames} frame(s) = {images_per_sample} images per sample")
        print(f"Cameras: {self.camera_names}")
        print(f"Temporal mode: {self.video_length > 1}")
        
        # Filter videos based on pid and world for distributed processing
        if world > 1:
            self.videos = [video for i, video in enumerate(self.videos) if i % world == pid]
            print(f"Process {pid}/{world}: Loading {len(self.videos)} samples out of total dataset")


    def __len__(self):
        return len(self.videos)
    
    def quaternion_to_matrix(self, quaternion):
        """Convert quaternion to rotation matrix"""
        q = Quaternion(quaternion)
        return q.rotation_matrix
    
    def get_camera_pose(self, camera_info):
        """
        Get camera pose in world coordinates
        Returns 4x4 transformation matrix
        """
        # Camera extrinsic (sensor to ego)
        sensor_translation = np.array(camera_info['extrinsic']['translation'])
        sensor_rotation = self.quaternion_to_matrix(camera_info['extrinsic']['rotation'])
        
        # Ego pose (ego to world)
        ego_translation = np.array(camera_info['ego_pose']['translation'])
        ego_rotation = self.quaternion_to_matrix(camera_info['ego_pose']['rotation'])
        
        # Sensor to ego transform
        sensor_to_ego = np.eye(4)
        sensor_to_ego[:3, :3] = sensor_rotation
        sensor_to_ego[:3, 3] = sensor_translation
        
        # Ego to world transform
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = ego_rotation
        ego_to_world[:3, 3] = ego_translation
        
        # Camera to world
        cam_to_world = ego_to_world @ sensor_to_ego
        
        return cam_to_world
    
    def load_voxel_label(self, gt_path):
        """
        Load voxel labels from Occ3D format
        Returns: voxel_label, mask_camera, mask_lidar arrays of shape [X, Y, Z]
        """
        if not os.path.exists(gt_path):
            print(f"Warning: GT path not found: {gt_path}")
            return (np.zeros(self.occ_size, dtype=np.uint8), 
                    np.zeros(self.occ_size, dtype=np.uint8),
                    np.zeros(self.occ_size, dtype=np.uint8))
        
        data = np.load(gt_path)
        semantics = data['semantics']  # Shape should be [X, Y, Z]
        mask_camera = data['mask_camera']  # Shape should be [X, Y, Z]
        mask_lidar = data['mask_lidar']  # Shape should be [X, Y, Z]
   
        
        return semantics.astype(np.uint8), mask_camera.astype(np.uint8), mask_lidar.astype(np.uint8)
    
    def get_cam_to_ego(self, camera_info):
        """
        Get camera to ego transformation
        Returns 4x4 transformation matrix
        """
        sensor_translation = np.array(camera_info['extrinsic']['translation'])
        sensor_rotation = self.quaternion_to_matrix(camera_info['extrinsic']['rotation'])
        
        # Sensor to ego transform
        sensor_to_ego = np.eye(4)
        sensor_to_ego[:3, :3] = sensor_rotation
        sensor_to_ego[:3, 3] = sensor_translation
        
        return sensor_to_ego
    
    def get_ego_pose(self, camera_info):
        """
        Get ego pose (ego to world transformation)
        Returns 4x4 transformation matrix
        """
        ego_translation = np.array(camera_info['ego_pose']['translation'])
        ego_rotation = self.quaternion_to_matrix(camera_info['ego_pose']['rotation'])
        
        ego_to_world = np.eye(4)
        ego_to_world[:3, :3] = ego_rotation
        ego_to_world[:3, 3] = ego_translation
        
        return ego_to_world
    
    def load_lidar_pointcloud(self, scene_name, frame_token):
        """Load lidar point cloud for a given frame using NuScenes devkit.
        
        Loads the LIDAR_TOP point cloud from NuScenes .bin files.
        Points are returned in the lidar frame.
        
        Args:
            scene_name: Name of the scene (not used, kept for API compatibility)
            frame_token: Token of the frame (sample token in NuScenes)
            
        Returns:
            lidar_points: [N, 3] numpy array of lidar points in lidar frame
        """
        try:
            # frame_token is the sample token in NuScenes
            sample = self.nusc.get('sample', frame_token)
            
            # Get LIDAR_TOP sample_data token from the sample
            lidar_token = sample['data']['LIDAR_TOP']
            lidar_data = self.nusc.get('sample_data', lidar_token)
            
            # Construct the lidar file path
            lidar_path = os.path.join(self.root, lidar_data['filename'])
            
            if not os.path.exists(lidar_path):
                print(f"Warning: Lidar file not found: {lidar_path}")
                return np.zeros((0, 3), dtype=np.float32)
            
            # Load lidar point cloud from .bin file
            # NuScenes format: (x, y, z, intensity, ring_index) - 5 floats per point
            lidar_points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
            lidar_points = lidar_points[:, :3]  # Keep only x, y, z
            
            return lidar_points.astype(np.float32)
            
        except Exception as e:
            print(f"Warning: Failed to load lidar for {scene_name}/{frame_token}: {e}")
            return np.zeros((0, 3), dtype=np.float32)
    
    def get_lidar_calibration(self, frame_token):
        """Get LIDAR_TOP calibration (lidar to ego transform) for a given frame.
        
        Args:
            frame_token: Token of the frame (sample token in NuScenes)
            
        Returns:
            lidar_to_ego: 4x4 transformation matrix from lidar to ego frame
        """
        sample = self.nusc.get('sample', frame_token)
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        calibrated_sensor = self.nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
        
        lidar2ego_translation = np.array(calibrated_sensor['translation'])
        lidar2ego_rotation = Quaternion(calibrated_sensor['rotation']).rotation_matrix
        
        lidar_to_ego = np.eye(4)
        lidar_to_ego[:3, :3] = lidar2ego_rotation
        lidar_to_ego[:3, 3] = lidar2ego_translation
        return lidar_to_ego
    
    def get_lidar_to_cam_transform(self, camera_info, frame_token):
        """Get transformation from lidar frame to camera frame.
        
        This computes: cam_from_lidar = cam_from_ego @ ego_from_lidar
        
        Args:
            camera_info: Camera info dict containing extrinsic and ego_pose
            frame_token: Token of the frame (sample token in NuScenes)
            
        Returns:
            lidar_to_cam: 4x4 transformation matrix from lidar to camera frame
        """
        # Get lidar to ego transform from NuScenes calibration
        lidar_to_ego = self.get_lidar_calibration(frame_token)
        
        # Get camera extrinsic (camera to ego)
        cam_to_ego = self.get_cam_to_ego(camera_info)
        ego_to_cam = np.linalg.inv(cam_to_ego)
        
        # Compose: lidar -> ego -> camera
        lidar_to_cam = ego_to_cam @ lidar_to_ego
        
        return lidar_to_cam
    
    def project_lidar_to_depth(self, lidar_points, camera_info, intrinsics, img_width, img_height, frame_token):
        """Project lidar points to camera image plane to create depth map.
        
        Args:
            lidar_points: [N, 3] lidar points in lidar frame
            camera_info: Camera info dict containing extrinsic and ego_pose
            intrinsics: 3x3 camera intrinsics matrix
            img_width: Output image width
            img_height: Output image height
            frame_token: Token of the frame (sample token in NuScenes)
            
        Returns:
            depth_map: [H, W] depth map with projected lidar depths (z-depth in camera frame)
        """
        if len(lidar_points) == 0:
            return np.zeros((img_height, img_width), dtype=np.float32)
        
        # Get lidar to camera transformation
        lidar_to_cam = self.get_lidar_to_cam_transform(camera_info, frame_token)
        
        # Add homogeneous coordinate
        lidar_points_homo = np.concatenate([lidar_points, np.ones((len(lidar_points), 1))], axis=1)  # [N, 4]
        
        # Transform to camera frame
        points_cam = (lidar_to_cam @ lidar_points_homo.T).T  # [N, 4]
        points_cam = points_cam[:, :3]  # [N, 3]
        
        # Filter points behind camera (z > 0)
        valid_depth = points_cam[:, 2] > 0.1  # At least 0.1m in front
        points_cam = points_cam[valid_depth]
        
        if len(points_cam) == 0:
            return np.zeros((img_height, img_width), dtype=np.float32)
        
        # Project to image plane
        points_2d_homo = (intrinsics @ points_cam.T).T  # [N, 3]
        depths = points_2d_homo[:, 2]  # z-depth
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # [N, 2]
        
        # Round to pixel coordinates
        points_2d = np.round(points_2d).astype(np.int32)
        
        # Filter points within image bounds
        valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width)
        valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
        valid_points = valid_x & valid_y
        
        points_2d = points_2d[valid_points]
        depths = depths[valid_points]
        
        # Create depth map (keep minimum depth for each pixel - closest point)
        depth_map = np.zeros((img_height, img_width), dtype=np.float32)
        for (x, y), depth in zip(points_2d, depths):
            if depth_map[y, x] == 0 or depth < depth_map[y, x]:
                depth_map[y, x] = depth
        
        return depth_map
    

    def compute_lidar_origins(self, scene_name, video_tokens, ref_frame_token):
        """Compute lidar origins for RayIoU evaluation.

        This function mimics the logic of ``EgoPoseDataset`` in
        ``occany.occany.metrics.ray_metrics`` to generate a small set of
        virtual lidar origins for a given scene, expressed in the *reference
        frame's ego coordinate system*.

        High‑level procedure:

        1. Assume a fixed NuScenes LIDAR_TOP extrinsic ``lidar_to_ego``
           (translation [0.9858, 0.0, 1.8402], rotation ~ identity).
        2. Use the reference frame's CAM_FRONT ego pose to compute the
           reference lidar pose in the world and its inverse
           (world -> reference lidar).
        3. Iterate over all frames in the scene:
           - For each frame, use CAM_FRONT ego pose to get that frame's
             lidar pose in the world.
           - Express the current lidar origin in the *reference lidar frame*
             by composing with the world -> reference_lidar transform.
           - For the reference frame itself, this origin is [0, 0, 0].
        4. Convert each origin from the reference lidar frame to the
           reference ego frame using the same ``lidar_to_ego`` extrinsic.
        5. Keep only origins inside a spatial window ``|x| < 39``,
           ``|y| < 39`` (same heuristic as in the original RayIoU code).
        6. If more than 8 origins remain, subsample them to at most 8
           evenly spaced in time.
        7. If no origin survives the spatial filter, fall back to a single
           default origin equal to ``lidar2ego_translation``.

        Args:
            scene_name: Name of the scene.
            video_tokens: List of frame tokens used for this sample
                (not directly used here, but kept for API symmetry).
            ref_frame_token: Frame token used as the reference frame for
                the coordinate system (usually the first frame of the
                video).

        Returns:
            A NumPy array of shape ``[T, 3]`` containing the selected lidar
            origins in the reference ego frame, where ``T <= 8``. The
            collate function adds a batch dimension to make it ``[1, T, 3]``
            before passing it to ``main_rayiou``.
        """
        scene_info = self.annotations['scene_infos'][scene_name]
        
        # Get lidar to ego transform from NuScenes calibration (using reference frame)
        lidar_to_ego = self.get_lidar_calibration(ref_frame_token)
        
        # Get reference frame's ego pose (using CAM_FRONT as reference for ego pose)
        ref_frame_info = scene_info[ref_frame_token]
        ref_cam_info = ref_frame_info['camera_sensor']['CAM_FRONT']
        ref_ego_to_world = self.get_ego_pose(ref_cam_info)
        
        # Compute global pose of reference lidar
        ref_lidar_to_world = ref_ego_to_world @ lidar_to_ego
        ref_world_to_lidar = np.linalg.inv(ref_lidar_to_world)
        
        # Collect all frame tokens in the scene
        all_frame_tokens = list(scene_info.keys())
        
        output_origin_list = []
        ref_index = all_frame_tokens.index(ref_frame_token) if ref_frame_token in all_frame_tokens else 0
        
        for curr_index, frame_token in enumerate(all_frame_tokens):
            frame_info = scene_info[frame_token]
            if 'CAM_FRONT' not in frame_info['camera_sensor']:
                continue
                
            curr_cam_info = frame_info['camera_sensor']['CAM_FRONT']
            
            if curr_index == ref_index:
                # Reference frame: origin is at [0, 0, 0] in lidar frame
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # Transform from current lidar frame to reference lidar frame
                curr_ego_to_world = self.get_ego_pose(curr_cam_info)
                curr_lidar_to_world = curr_ego_to_world @ lidar_to_ego
                ref_from_curr = ref_world_to_lidar @ curr_lidar_to_world
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)
            
            # Transform to ego coordinates
            origin_tf_pad = np.ones(4)
            origin_tf_pad[:3] = origin_tf
            origin_tf = (lidar_to_ego[:3] @ origin_tf_pad).astype(np.float32)
            
            # Filter origins within spatial range (same as ray_metrics.py)
            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)
        
        # Select up to 8 origins evenly spaced
        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]
        
        if len(output_origin_list) == 0:
            # Fallback: use fixed lidar origin (translation from lidar_to_ego)
            output_origin_list = [lidar_to_ego[:3, 3].astype(np.float32)]
        
        output_origin_tensor = np.stack(output_origin_list)  # [T, 3]
        return output_origin_tensor
    
    def project_voxels_to_depth(self, voxel_label, cam_to_ego, intrinsics, img_width, img_height):
        """
        Project occupied voxels to image plane to create depth map
        
        Args:
            voxel_label: [X, Y, Z] voxel grid with semantic labels (255 = invalid/empty)
            cam_to_ego: 4x4 camera to ego transformation matrix
            intrinsics: 3x3 camera intrinsics matrix
            img_width: output image width
            img_height: output image height
            
        Returns:
            depth_map: [H, W] depth map with projected voxel depths
        """
        # Get occupied voxels (not empty and not invalid)
        occupied_mask = (voxel_label != 255) & (voxel_label != 17)  # 17 is free/empty class
        occupied_indices = np.argwhere(occupied_mask)  # [N, 3] array of (x, y, z) indices
        
        if len(occupied_indices) == 0:
            return np.zeros((img_height, img_width), dtype=np.float32)
        
        # Convert voxel indices to world coordinates (ego frame)
        # voxel_origin is the minimum corner of the voxel grid
        voxel_size = 0.4  # 0.4m voxel size for nuScenes
        voxel_centers = occupied_indices.astype(np.float32) * voxel_size + self.voxel_origin + voxel_size / 2
        
        # Transform from ego frame to camera frame
        ego_to_cam = np.linalg.inv(cam_to_ego)
        
        # Add homogeneous coordinate
        voxel_centers_homo = np.concatenate([voxel_centers, np.ones((len(voxel_centers), 1))], axis=1)  # [N, 4]
        
        # Transform to camera frame
        points_cam = (ego_to_cam @ voxel_centers_homo.T).T  # [N, 4]
        points_cam = points_cam[:, :3]  # [N, 3]
        
        # Filter points behind camera
        valid_depth = points_cam[:, 2] > 0.1  # At least 0.1m in front
        points_cam = points_cam[valid_depth]
        
        if len(points_cam) == 0:
            return np.zeros((img_height, img_width), dtype=np.float32)
        
        # Project to image plane
        points_2d_homo = (intrinsics @ points_cam.T).T  # [N, 3]
        depths = points_2d_homo[:, 2]
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]  # [N, 2]
        
        # Round to pixel coordinates
        points_2d = np.round(points_2d).astype(np.int32)
        
        # Filter points within image bounds
        valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < img_width)
        valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < img_height)
        valid_points = valid_x & valid_y
        
        points_2d = points_2d[valid_points]
        depths = depths[valid_points]
        
        # Create depth map (keep minimum depth for each pixel)
        depth_map = np.zeros((img_height, img_width), dtype=np.float32)
        for (x, y), depth in zip(points_2d, depths):
            if depth_map[y, x] == 0 or depth < depth_map[y, x]:
                depth_map[y, x] = depth
        
        return depth_map
    
    def load_frame_data(self, scene_name, frame_token, camera_name, lidar_points=None):
        """Load data for a single frame and camera
        
        Args:
            scene_name: Name of the scene
            frame_token: Token of the frame
            camera_name: Name of the camera
            lidar_points: [N, 3] lidar points in lidar frame (optional, for gt_depth generation)
        """
        scene_info = self.annotations['scene_infos'][scene_name]
        frame_info = scene_info[frame_token]
        camera_sensor = frame_info['camera_sensor']
        
        if camera_name not in camera_sensor:
            return None
            
        cam_info = camera_sensor[camera_name]
        
        # Load image
        img_path = os.path.join(self.root,  "imgs", cam_info['img_path'])
        
        img = Image.open(img_path).convert('RGB')
        
        # Load camera mask if available
        mask_path = os.path.join(self.root, "camera_mask", cam_info['img_path'].replace('.jpg', '.png'))
        if os.path.exists(mask_path):
            camera_mask = np.array(Image.open(mask_path))
        else:
            # Create default mask (all valid)
            camera_mask = np.ones((img.height, img.width), dtype=np.uint8)
        

        # Get camera intrinsics
        cam_k = np.array(cam_info['intrinsics'])  # 3x3 matrix
        
        # Get camera pose (world frame)
        cam_pose = self.get_camera_pose(cam_info)
        
        # Get camera to ego transformation
        cam_to_ego = self.get_cam_to_ego(cam_info)
        
        # Resize image, mask, and adjust intrinsics
        place_holder_depth = np.zeros((img.height, img.width), dtype=np.float32)
        downscaled_img, _, intrinsics_resized = crop_resize_if_necessary(
            img, place_holder_depth, cam_k, self.output_resolution
        )
        
        # Resize camera mask to match output resolution
        camera_mask_resized = cv2.resize(
            camera_mask, 
            (self.output_resolution[0], self.output_resolution[1]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        if lidar_points is not None and len(lidar_points) > 0:
            # Project lidar points to create sparse depth map
            gt_depth = self.project_lidar_to_depth(
                lidar_points,
                cam_info, 
                intrinsics_resized, 
                downscaled_img.width, 
                downscaled_img.height,
                frame_token
            )
        else:
            gt_depth = np.zeros((downscaled_img.height, downscaled_img.width), dtype=np.float32)
        
        # Transform images
        if self.base_model == 'da3':
            img_tensor = InputProcessor.NORMALIZE(to_tensor(downscaled_img))
        else:
            img_tensor = ImgNorm(np.array(downscaled_img))
        gdino_img, _ = GroundingDinoImgNorm(downscaled_img, None)
        sam2_img = self.SAM2_transforms(np.array(downscaled_img))
        sam3_img = self.SAM3_transforms(downscaled_img)
        
        # Load bounding boxes if available
        box_dict = {'boxes': np.array([]), 'confidences': np.array([]), 'labels': np.array([])}
        if self.boxes_dir is not None:
            box_path = os.path.join(self.boxes_dir, scene_name, f"{frame_token}_{camera_name}", "boxes.npz")
            assert os.path.exists(box_path), f"Box path {box_path} does not exist"
            box_file = np.load(box_path)
            box_dict = {
                'boxes': box_file['boxes'],
                'confidences': box_file['confidences'],
                'labels': box_file['labels']
            }
            box_file.close()
            
        
        return {
            'img': img_tensor,
            'gdino_img': gdino_img,
            'sam2_img': sam2_img,
            'sam3_img': sam3_img,
            'gt_depth': gt_depth,
            'cam_k_resized': intrinsics_resized,
            'cam_pose': cam_pose,
            'cam_to_ego': cam_to_ego,
            'camera_mask': camera_mask_resized,
            'image_path': img_path,
            'box_dict': box_dict
        }
    
    def __getitem__(self, index):
        video_info = self.videos[index]
        scene_name = video_info['scene_name']
        video_tokens = video_info['video_tokens']
        begin_frame_token = video_info['begin_frame_token']
        
        
        
        # Load voxel labels for the first frame
        scene_info = self.annotations['scene_infos'][scene_name]
        begin_frame_info = scene_info[begin_frame_token]
        gt_path = os.path.join(self.root, begin_frame_info['gt_path'])

        
        voxel_label, voxel_mask_camera, voxel_mask_lidar = self.load_voxel_label(gt_path)
        
        # Apply camera mask to voxel labels if enabled
        if self.apply_camera_mask:
            # Set voxels not visible from camera to 255 (ignore)
            voxel_label[voxel_mask_camera == 0] = 255
        
        # Apply lidar mask to voxel labels if enabled
        if self.apply_lidar_mask:
            # Set voxels not visible from lidar to 255 (ignore)
            voxel_label[voxel_mask_lidar == 0] = 255
        
        if not self.use_surround_label:
            voxel_label[:100, :, :] = 255
            voxel_mask_camera[:100, :, :] = 0
            voxel_mask_lidar[:100, :, :] = 0

        # Load lidar point cloud for the reference frame
        lidar_points = self.load_lidar_pointcloud(scene_name, begin_frame_token)
        
        # Process video frames
        imgs = []
        gdino_imgs = []
        sam2_imgs = []
        sam3_imgs = []
        gt_depths = []
        cam_k_resized = []
        cam_poses = []
        cam_to_egos = []
        camera_masks = []
        image_paths = []
        box_dicts = []
        camera_names = []
        
        # Load data based on temporal and camera configuration
        # Order: temporal frames first, then cameras within each frame
        # This ensures temporal consistency is maintained
        for frame_idx, frame_token in enumerate(video_tokens):
            for camera_name in self.camera_names:
                # Only pass lidar_points for the first frame (begin frame)
                lidar_for_frame = lidar_points if frame_idx == 0 else None
                frame_data = self.load_frame_data(scene_name, frame_token, camera_name, lidar_points=lidar_for_frame)
                imgs.append(frame_data['img'])
                gdino_imgs.append(frame_data['gdino_img'])
                sam2_imgs.append(frame_data['sam2_img'])
                sam3_imgs.append(frame_data['sam3_img'])
                gt_depths.append(frame_data['gt_depth'])
                cam_k_resized.append(frame_data['cam_k_resized'])
                cam_poses.append(frame_data['cam_pose'])
                cam_to_egos.append(frame_data['cam_to_ego'])
                camera_masks.append(frame_data['camera_mask'])
                image_paths.append(frame_data['image_path'])
                box_dicts.append(frame_data['box_dict'])
                camera_names.append(camera_name)
        
        # Stack data
        gt_depths = np.stack(gt_depths)
        cam_k_resized = np.stack(cam_k_resized)
        cam_to_egos = np.stack(cam_to_egos)
        camera_masks = np.stack(camera_masks)
        
        # Compute relative poses (relative to first frame)
        cam_poses = np.stack(cam_poses)
        in_cam0 = np.linalg.inv(cam_poses[0])
        cam_poses_in_cam0 = np.array([in_cam0 @ cam_pose for cam_pose in cam_poses])
        
       
        # For NuScenes, we need cam0_to_ego for the first frame
        # This is used to transform predicted points from cam0 frame to ego frame
        cam0_to_ego = cam_to_egos[0]
        
        # Load novel view RGB images if path is provided
        novel_view_rgbs = None
        if self.novel_view_rgb_path is not None:
            novel_view_rgb_dir = os.path.join(self.novel_view_rgb_path, f"{scene_name}_{begin_frame_token}")
            rgbs_dir = os.path.join(novel_view_rgb_dir, "rgbs")
            if os.path.isdir(rgbs_dir):
                image_files = [f for f in os.listdir(rgbs_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

                def _image_order(filename):
                    stem = os.path.splitext(filename)[0]
                    try:
                        return int(stem.split("_")[-1])
                    except ValueError:
                        return float("inf")

                image_files = sorted(image_files, key=_image_order)

                novel_view_rgbs = []
                for img_name in image_files:
                    img_path = os.path.join(rgbs_dir, img_name)
                    try:
                        img = Image.open(img_path).convert("RGB")
                        
                    except (FileNotFoundError, OSError):
                        continue

                    if self.output_resolution is not None:
                        img = img.resize(self.output_resolution, Image.BILINEAR)

                    np_image = np.asarray(img)
                    ts_image = torch.from_numpy(np_image.copy()).permute(2, 0, 1).to(torch.uint8)
                    novel_view_rgbs.append(ts_image)

                if len(novel_view_rgbs) > 0:
                    novel_view_rgbs = torch.stack(novel_view_rgbs)
                else:
                    novel_view_rgbs = None
        
        # Compute lidar origins for RayIoU evaluation
        lidar_origin = self.compute_lidar_origins(scene_name, video_tokens, begin_frame_token)
        
        data = {
            "camera_names": camera_names,
            "imgs": torch.stack(imgs),
            "gdino_imgs": torch.stack(gdino_imgs),
            "sam2_imgs": torch.stack(sam2_imgs),
            "sam3_imgs": torch.stack(sam3_imgs),
            "begin_frame_token": begin_frame_token,
            "scene_name": scene_name,
            "cam_poses": cam_poses,  # [T*C, 4, 4] global camera poses (camera to world)
            "cam_poses_in_cam0": cam_poses_in_cam0,
            "cam0_to_ego": cam0_to_ego,  # Transformation from first camera to ego frame
            "gt_depths": gt_depths,
            "camera_masks": camera_masks,
            "image_paths": image_paths,
            "cam_k_resized": cam_k_resized,
            "box_dicts": box_dicts,
            "voxel_label": voxel_label,
            "voxel_mask_camera": voxel_mask_camera,
            "voxel_mask_lidar": voxel_mask_lidar,
            "novel_view_rgbs": novel_view_rgbs,
            "lidar_origin": lidar_origin,  # [T, 3] lidar origins for RayIoU
            "lidar_points": lidar_points,  # [N, 3] lidar points in lidar frame
        }
        
        return data
