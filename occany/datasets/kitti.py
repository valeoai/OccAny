import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
import occany.datasets.semantic_kitti_io as SemanticKittiIO
import cv2
from tqdm import tqdm
from occany.datasets.class_mapping import ClassMapping
from occany.utils.helpers import project_lidar_world2camera, crop_resize_if_necessary
from occany.utils.image_util import GroundingDinoImgNorm, ImgNorm, get_SAM2_transforms, get_SAM3_transforms
from depth_anything_3.utils.io.input_processor import InputProcessor
from torchvision.transforms.functional import to_tensor

# Keep a single source of truth for the semantic class order and
# the synonyms/prompts we feed into SAM for each class.
KITTI_CLASS_PROMPTS = [
    ("empty", ["sky"]),  # Used as ignore/empty label
    ("car", ["car"]),
    ("bicycle", ["bicycle", "bike"]),
    ("motorcycle", ["motorcycle", "motorbike", "scooter"]),
    ("truck", ["truck", "lorry"]),
    ("other-vehicle", ["caravan", "trailer", "train", "tram"]),
    ("person", ["person", "pedestrian"]),
    ("bicyclist", ["bicyclist", "cyclist"]),
    ("motorcyclist", ["motorcyclist"]),
    ("road", ["road"]),
    ("parking", ["parking"]),
    ("sidewalk", ["sidewalk"]),
    ("other-ground", ["traffic island", "rail track"]),
    ("building", ["building"]),
    ("fence", ["fence", "barrier"]),
    ("vegetation", ["vegetation", "bush", "shrub", "foliage"]),
    ("trunk", ["trunk", "stem"]),
    ("terrain", ["terrain", "grass", "soil"]),
    ("pole", ["pole", "lamp post"]),
    ("traffic-sign", ["traffic sign"]),
]




def collate_kitti_identity(batch):
    """
    Collate function for KITTI dataset
    
    Args:
        batch: List of dataset items
    """
    data = {
        'imgs': torch.stack([item['imgs'] for item in batch]),
        "gdino_imgs": torch.stack([item['gdino_imgs'] for item in batch]),
        'sam2_imgs': torch.stack([item['sam2_imgs'] for item in batch]),
        'sam3_imgs': torch.stack([item['sam3_imgs'] for item in batch]),
        'begin_frame_id': [item['begin_frame_id'] for item in batch],
        'sequence': [item['sequence'] for item in batch],
        'gt_depths': torch.stack([torch.from_numpy(item['gt_depths']) for item in batch]),
        'cam_k_resized': torch.stack([torch.from_numpy(item['cam_k_resized']) for item in batch]),
        'cam_poses_in_cam0': torch.stack([torch.from_numpy(item['cam_poses_in_cam0']) for item in batch]),
        'T_velo_2_cam': torch.stack([torch.from_numpy(item['T_velo_2_cam']) for item in batch]),
        'voxel_label': torch.stack([torch.from_numpy(item['voxel_label']) for item in batch]),
        'image_paths': [item['image_paths'] for item in batch],
        'box_dicts': [item['box_dicts'] for item in batch],
    }

    if any(item.get('novel_view_rgbs') is not None for item in batch):
        data['novel_view_rgbs'] = torch.stack([item.get('novel_view_rgbs') for item in batch])

    return data


def collate_kitti_metric(batch):
    return {
        'begin_frame_id': [item['begin_frame_id'] for item in batch],
        'sequence': [item['sequence'] for item in batch],
        'voxel_label': torch.stack([torch.from_numpy(item['voxel_label']) for item in batch]),
    }


class KittiDataset(Dataset):
    def __init__(
        self,
        split,
        semkitti_root,
        kittiodo_root,
        remap_lut_path,
        color_jitter=None,
        fliplr=0.0,
        frame_interval=5,
        video_length=10,
        output_resolution = (512, 160),
        video_interval=50, # distance between the init frame in the video
        boxes_dir=None,
        pid=0,
        world=1,
        novel_view_rgb_path=None,
        sam3_resolution=1008,
        base_model="must3r",
    ):
        super().__init__()
        self.base_model = base_model
        self.semkitti_root = semkitti_root
        self.kittiodo_root = kittiodo_root
        self.novel_view_rgb_path = novel_view_rgb_path
        sam2_output_resolution = min(1024, max(output_resolution))
        self.SAM2_transforms = get_SAM2_transforms(resolution=sam2_output_resolution)
        self.sam3_output_resolution = sam3_resolution
        self.SAM3_transforms = get_SAM3_transforms(resolution=self.sam3_output_resolution)
        
        # Derive class names and prompts from the canonical definition above to
        # keep ordering consistent between CLASS_NAMES and PROMPT consumers.
        self.CLASS_NAMES = [name for name, _ in KITTI_CLASS_PROMPTS]

        # # from LOSC
        # self.PROMPT = [
        #     ["sky"], # "empty"
        #     ['car'],
        #     ['bicycle'],
        #     ['motorcycle'],
        #     ['truck'],
        #     ['trailer', 'semi trailer', 'cargo container', 'shipping container', 'freight container',
        #     'caravan', 'bus', 'bulldozer', 'excavator', 'concrete mixer', 'crane', 'dump truck', 'train', 'tram'],
        #     ['person', 'pedestrian'],
        #     ['bicyclist', 'cyclist'],
        #     ['motorcyclist'],
        #     ['road'],
        #     ['parking', 'parking lot'],
        #     ['sidewalk', 'curb', 'bike path', 'walkway', 'pavement','footpath','footway','boardwalk','driveway'],
        #     ['water', 'river', 'lake', 'watercourse', 'waterway', 'canal', 'ditch', 'rail track', 'traffic island',
        #     'traffic median', 'median strip', 'roadway median', 'central reservation'],
        #     ['building', 'house', 'garage', 'wall', 'railing', 'stairs', 'awning', 'roof', 'bridge'],
        #     ['fence', 'barrier', 'barricade'],
        #     ['tree', 'bush', 'shrub', 'plant', 'flower'],
        #     ['tree trunk', 'trunk', 'woods'],
        #     ['terrain', 'grass', 'soil', 'grassland', 'hill', 'sand', 'gravel', 'lawn', 'meadow', 'garden', 'earth', 'peeble', 'rock'],
        #     ['pole'],
        #     ['traffic sign'],
        # ]
        self.PROMPT = [prompt_variants for _, prompt_variants in KITTI_CLASS_PROMPTS]

        # Mapping to superclasses
        self.MAPPING = [
            0, # empty
            3, # car
            3, # bicycle
            3, # motorcycle
            3, # truck
            3, # other-vehicle
            5, # person
            5, # bicyclist
            5, # motorcyclist
            1, # road
            1, # parking
            1, # sidewalk
            1, # other-ground given unclear text description
            2, # building
            6, # fence
            4, # vegetation
            4, # trunk
            4, # terrain
            6, # pole
            6, # traffic-sign
        ]
        
        # self.MAPPING = [
        #     0, # empty
        #     1, # car
        #     2, # bicycle
        #     3, # motorcycle
        #     1, # truck
        #     1, # other-vehicle
        #     4, # person
        #     2, # bicyclist
        #     3, # motorcyclist
        #     6, # road
        #     6, # parking
        #     7, # sidewalk
        #     255, # other-ground given unclear text description
        #     8, # building
        #     8, # fence
        #     9, # vegetation
        #     9, # trunk
        #     5, # terrain
        #     9, # pole
        #     9, # traffic-sign
        # ]
        
        
        self.SUPERCLASS_NAMES = [
            "empty", # 0
            "ground", # 1
            "structure", # 2
            "vehicle", # 3
            "nature", # 4
            "human", # 5
            "object", # 6
            # "sidewalk", # 7
            # "manmade", # 8
            # "vegetation", # 9
            # "parking", # 7
            # "ground", # 11
        ]
        
        self.superclass_empty_class = 0  # "empty" is at index 0 in SUPERCLASS_NAMES
        
        # Default to original class names
        # self.CLASS_NAMES = [
        #     "empty", # 0
        #     "vehicle", # 1
        #     "bicycle", # 2
        #     "motorcycle", # 3
        #     "person", # 4
        #     "terrain", # 5
        #     "drivable surface", # 6
        #     "sidewalk", # 7
        #     "manmade", # 8
        #     "vegetation", # 9
        #     # "parking", # 7
        #     # "ground", # 11
        # ]

        self.COMBINED_TEXT_PROMPT = [
            ["sky"], 
            ["car", "truck", "trailer", "bus", "vans", "tram", "train"], # vehicle
            ["bicycle", "bicyclist"],
            ["motorcycle", "motorcyclist"],
            ["pedestrian", 'adult', 'child'],
            ['grass', 'rolling hill', 'soil', 'sand', 'gravel'],
            ["road", "parking"],
            ["sidewalk"],
            ["building", "fence", "pole", "traffic sign", "wall"],
            ["vegetation", "tree", "bush", "trunk"],
            ['traffic island', 'rail track', 'lake', 'river'],
        ]
        self.COMBINED_COLORS=np.array([
            [0, 0, 0, 255], # "empty"
            [100, 150, 245, 255], # "car"
            [100, 230, 245, 255], # "bicycle"
            [30, 60, 150, 255], # "motorcycle"
            [255, 30, 30, 255], # "person"
            [150, 240, 80, 255], # "terrain"
            [255, 0, 255, 255], # "road"
            # [255, 150, 255, 255], # "parking"
            [75, 0, 75, 255], # "sidewalk"
            [255, 200, 0, 255], # "building"
            [0, 175, 0, 255], # "vegetation"
            # [175, 0, 75, 255], # "other-ground"
            # [80, 30, 180, 255], # "truck"
            # [100, 80, 250, 255], # "other-vehicle"
            # [255, 40, 200, 255], # "rider"
            # [150, 30, 90, 255], # "motorcyclist"
            # [255, 120, 50, 255], # "fence"
            # [135, 60, 0, 255], # "trunk"
            # [255, 240, 150, 255], # "pole"
            # [255, 0, 0, 255], # "traffic-sign"
            [255, 255, 255, 255], # "unknown"
        
        ])

        self.COLORS=np.array([
            [0, 0, 0, 255], # "empty"
            [100, 150, 245, 255], # "car"
            [100, 230, 245, 255], # "bicycle"
            [30, 60, 150, 255], # "motorcycle"
            [80, 30, 180, 255], # "truck"
            [100, 80, 250, 255], # "other-vehicle"
            [255, 30, 30, 255], # "person"
            [255, 40, 200, 255], # "rider"
            [150, 30, 90, 255], # "motorcyclist"
            [255, 0, 255, 255], # "road"
            [255, 150, 255, 255], # "parking"
            [75, 0, 75, 255], # "sidewalk"
            [175, 0, 75, 255], # "other-ground"
            [255, 200, 0, 255], # "building"
            [255, 120, 50, 255], # "fence"
            [0, 175, 0, 255], # "vegetation"
            [135, 60, 0, 255], # "trunk"
            [150, 240, 80, 255], # "terrain"
            [255, 240, 150, 255], # "pole"
            [255, 0, 0, 255], # "traffic-sign"
            [255, 255, 255, 255], # "unknown"
        
        ])

        self.n_classes = len(self.CLASS_NAMES)
        self.empty_class = 0
        self.other_class = self.n_classes
        # self.scene_errors = {
        #     "08": set(["000050"])
        # }
        splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
            "val": ["08"],
            "test": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
            "trainval": ["08", "00", "01", "02", "03", "04", "05", "06", "07", "09", "10"],
        }
        self.split = split
        self.sequences = splits[split]
        self.scene_size = (51.2, 51.2, 6.4)
        self.voxel_origin = np.array([0, -25.6, -2])
        
        self.fliplr = fliplr
        self.boxes_dir = boxes_dir

        self.class_mapping = ClassMapping()

        self.remap_lut = SemanticKittiIO.get_remap_lut(remap_lut_path)
        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1226
        self.img_H = 370
        self.output_resolution = output_resolution
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.videos = {}
        self.frame_names = []
        for sequence in tqdm(self.sequences):
            glob_path = os.path.join(
                self.semkitti_root, "dataset", "sequences", sequence, "voxels", "*.bin"
            )
            voxel_paths = glob.glob(glob_path)
            min_frame_id = 0
            max_frame_id = len(voxel_paths)

            calib = self.read_calib(
                os.path.join(self.semkitti_root, "dataset", "sequences", sequence, "calib.txt")
            )
            P = calib["P2"]
            T_velo_2_cam = calib["Tr"]
            T_cam_2_velo = np.linalg.inv(T_velo_2_cam)
            proj_matrix = P @ T_velo_2_cam

            pose_path = os.path.join(self.semkitti_root, "dataset", "sequences", sequence, "poses.txt")
            global_poses = self.parse_poses(pose_path, calib)
            self.frame2idx = {}

            for begin_frame_id in tqdm(range(min_frame_id, max_frame_id, video_interval), desc=f"Processing frames in {sequence}", leave=False):


                end_frame_id = begin_frame_id + video_length * frame_interval
                if end_frame_id > max_frame_id:
                    break
                label_path = os.path.join(self.semkitti_root, "dataset", "sequences", sequence, "voxels", f"{begin_frame_id:06d}.label")
                assert os.path.exists(label_path), f"Label path {label_path} does not exist"
                voxel_path = label_path.replace(".label", ".bin")
                invalid_path = label_path.replace(".label", ".invalid")
                assert os.path.exists(voxel_path), f"Voxel path {voxel_path} does not exist"

                box_paths = []
                image_paths = []
                semantic_2ds = []
                lidar_paths = []
                lidar_label_paths = []
                poses = []
                cam_poses = []

                for i in range(begin_frame_id, end_frame_id, frame_interval):
                    image_path = os.path.join(self.kittiodo_root, "dataset", "sequences", sequence, "image_2", f"{i:06d}.png")
                    assert os.path.exists(image_path), f"Image path {image_path} does not exist"
                    image_paths.append(image_path)

                    lidar_path = os.path.join(self.semkitti_root, "dataset", "sequences", sequence, "velodyne", f"{i:06d}.bin")
                    assert os.path.exists(lidar_path), f"Lidar path {lidar_path} does not exist"
                    lidar_paths.append(lidar_path)

                    lidar_label_path = os.path.join(self.semkitti_root, "dataset", "sequences", sequence, "labels", f"{i:06d}.label")
                    assert os.path.exists(lidar_label_path), f"Lidar label path {lidar_label_path} does not exist"
                    lidar_label_paths.append(lidar_label_path)

                    if self.boxes_dir is not None:
                        box_path = os.path.join(self.boxes_dir, f"{sequence}_{i:06d}", "boxes.npz")
                        assert os.path.exists(box_path), f"Box path {box_path} does not exist"
                        box_paths.append(box_path)

                    T_velo_2_world = global_poses[i]
                    T_cam_2_world = T_velo_2_world @ T_cam_2_velo
                    cam_poses.append(T_cam_2_world)
                  

                self.frame_names.append(f"{sequence}_{begin_frame_id:06d}")
                self.videos[f"{sequence}_{begin_frame_id:06d}"] = {
                    "begin_frame_id": begin_frame_id,
                    "pose": global_poses[begin_frame_id],
                    "cam_poses": cam_poses,
                    "box_paths": box_paths,
                    "sequence": sequence,
                    "P": P,
                    "T_velo_2_cam": T_velo_2_cam,
                    "proj_matrix": proj_matrix,
                    "voxel_path": voxel_path,
                    "invalid_path": invalid_path,
                    "lidar_paths": lidar_paths,
                    "lidar_label_paths": lidar_label_paths,
                    "label_path": label_path,
                    "image_paths": image_paths,
                    # "semantic_2ds": np.stack(semantic_2ds),

                }
        
        # Filter frame_names based on pid and world for distributed processing
        if world > 1:
            self.frame_names = [frame_name for i, frame_name in enumerate(self.frame_names) if i % world == pid]
            print(f"Process {pid}/{world}: Loading {len(self.frame_names)} samples out of total dataset")
        
        self.frame2idx = {frame_name: i for i, frame_name in enumerate(self.frame_names)}

        self.colors = SemanticKittiIO.get_colors()


    @staticmethod
    def parse_poses(filename, calibration):
        """ read poses file with per-scan poses from given filename
            NOTE: all LiDAR poses in the coordinate system of the left(?) camera.
            Therefore, one needs to pre/post-multiply the calibration matrices to get "real" LiDAR poses
            Returns
            -------
            list
                list of poses as 4x4 numpy arrays.
        """
        file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = np.linalg.inv(Tr)

        for line in file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

        return poses

    def __getitem__(self, index):
        video = self.videos[self.frame_names[index]]
        label_path = video["label_path"]
        sequence = video["sequence"]
        begin_frame_id = video["begin_frame_id"]
        invalid_path = video["invalid_path"]
        voxel_label = self.read_voxel_label(label_path, invalid_path)
        P = video["P"]
        T_velo_2_cam = video["T_velo_2_cam"]
        T_cam_2_velo = np.linalg.inv(T_velo_2_cam)
        proj_matrix = video["proj_matrix"]
        image_paths = video["image_paths"]
        cam_k = P[0:3, 0:3]
        lidar_paths = video["lidar_paths"]
        lidar_label_paths = video["lidar_label_paths"]
        cam_poses = video["cam_poses"]  
        box_paths = video["box_paths"]

        start_img = Image.open(image_paths[0])
        img_w, img_h = start_img.size

        box_dicts = []
        gt_depths = []
        imgs = []
        gdino_imgs = []
        sam2_imgs = []
        sam3_imgs = []
        in_cam0 = np.linalg.inv(cam_poses[0])
        cam_poses_in_cam0 = [in_cam0 @ cam_pose for cam_pose in cam_poses]
        cam_k_resized = []
        for i, (lidar_path, lidar_label_path) in enumerate(zip(lidar_paths, lidar_label_paths)):
            lidar_points, lidar_label = self.read_lidar_label(lidar_path, lidar_label_path)
            semantic_2d_pseudolabel = None
            if i == 0:
                lidar_points_t0 = lidar_points
                lidar_label_t0 = lidar_label.astype(np.uint16)
                lidar_label_t0 = self.class_mapping.map_kitti_id_2_common_id(lidar_label_t0)
           
            img = Image.open(image_paths[i])
    
            output_resolution = self.output_resolution
            place_holder_depth = np.zeros((img.height, img.width), dtype=np.float32)
            downscaled_img, _, intrinsics2 = crop_resize_if_necessary(img, place_holder_depth, cam_k, output_resolution)
            
            gt_depth, _, _, _ = project_lidar_world2camera(
                lidar_points,
                downscaled_img.width,
                downscaled_img.height,
                T_cam_2_velo,
                intrinsics2,
            )
           
            if self.base_model == 'da3':
                imgs.append(InputProcessor.NORMALIZE(to_tensor(downscaled_img)))
            else:
                imgs.append(ImgNorm(np.array(downscaled_img)))
            gdino_img, _ = GroundingDinoImgNorm(downscaled_img, None)
            gdino_imgs.append(gdino_img)
            sam2_imgs.append(self.SAM2_transforms(np.array(downscaled_img)))
            sam3_imgs.append(self.SAM3_transforms(downscaled_img))
            gt_depths.append(gt_depth)
            cam_k_resized.append(intrinsics2)

            if self.boxes_dir is not None:
                box_path = box_paths[i]
                box_file = np.load(box_path)
                box_dict = {
                    'boxes': box_file['boxes'],
                    'confidences': box_file['confidences'],
                    'labels': box_file['labels']
                }
                box_file.close()  # Close the file handle to avoid pickling issues
                box_dicts.append(box_dict)
                
      

        gt_depths = np.stack(gt_depths)


        novel_view_rgbs = None
        if self.novel_view_rgb_path is not None:
            novel_view_rgb_dir = os.path.join(self.novel_view_rgb_path, f"{sequence}_{begin_frame_id:06d}")
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
                    # novel_view_rgbs.append(ImgNorm(np.array(img)))

                if len(novel_view_rgbs) > 0:
                    novel_view_rgbs = torch.stack(novel_view_rgbs)
                else:
                    novel_view_rgbs = None
                
              

        data = {
            "imgs": torch.stack(imgs),
            "gdino_imgs": torch.stack(gdino_imgs),  
            "sam2_imgs": torch.stack(sam2_imgs),
            "sam3_imgs": torch.stack(sam3_imgs),
            "begin_frame_id": begin_frame_id,
            "pose": video["pose"],
            "cam_poses_in_cam0": np.stack(cam_poses_in_cam0),
            "sequence": sequence,
            "P": P,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix": proj_matrix,
            "cam_k": cam_k,
            "gt_depths": gt_depths,
            "image_paths": image_paths,
            "lidar_points": lidar_points_t0,
            "lidar_label": lidar_label_t0.astype(np.uint8),
            "cam_k_resized": np.stack(cam_k_resized),
            "box_dicts": box_dicts,
            "voxel_label": voxel_label,
            "novel_view_rgbs": novel_view_rgbs,
        }
        return data

    def depth_from_lidar(self, lidar_points, proj_matrix, img_w, img_h, semantic_2d_label=None ):



        lidar_hcoords = np.concatenate([lidar_points, np.ones((lidar_points.shape[0], 1), dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ lidar_hcoords.T).T

        depth = img_points[:, 2]
        img_points = img_points[:, :2] / img_points[:, 2:3] # scale to pixel coordinates

        img_points = np.round(img_points).astype(np.int32)
        point_in_frustum_idx  = self.select_points_in_frustum(img_points, 0, 0, img_w, img_h)
        point_in_front_idx = depth > 0 # only keep points in front of the image
        keep_idx = point_in_frustum_idx & point_in_front_idx
        img_points, depth = img_points[keep_idx], depth[keep_idx]

        gt_depth = np.zeros((img_h, img_w)) - 1.0 # -1.0 means invalid depth
        gt_depth[img_points[:, 1], img_points[:, 0]] = depth

        lidar_pseudolabel = None
        if semantic_2d_label is not None:
            lidar_pseudolabel = semantic_2d_label[img_points[:, 1], img_points[:, 0]]


        return gt_depth, lidar_pseudolabel, keep_idx


    def __len__(self):
        return len(self.frame_names)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)
        return keep_ind



    def read_lidar_label(self, lidar_path, lidar_label_path):
        # compute depth from lidar
        lidar_points = np.fromfile(lidar_path, dtype=np.float32)
        lidar_points = lidar_points.reshape((-1, 4))[:, :3]

        label = np.fromfile(lidar_label_path, dtype=np.uint32)
        label = label.reshape((-1))
        label = label & 0xFFFF  # get lower half for semantics
        label = self.remap_lut[label.astype(np.uint16)].astype(np.float32)
        label[label == 0] = 255
        return lidar_points, label

    def read_voxel_label(self, label_path, invalid_path):
        LABEL = SemanticKittiIO._read_label_SemKITTI(label_path)
        INVALID = SemanticKittiIO._read_invalid_SemKITTI(invalid_path)
        LABEL = self.remap_lut[LABEL.astype(np.uint16)].astype(
            np.float32
        )  # Remap 20 classes semanticKITTI SSC
        LABEL[
            np.isclose(INVALID, 1)
        ] = 255  # Setting to unknown all voxels marked on invalid mask...
        LABEL = LABEL.reshape([256, 256, 32])
        return LABEL

    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])
        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out
