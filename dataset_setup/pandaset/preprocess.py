import os
import numpy as np
import os.path as osp
from tqdm import tqdm
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from torch.utils.data import Dataset
from pandaset import DataSet as PandaSet, geometry
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tool import *



class PandasetDataset(Dataset):
    def __init__(self, load_dir, save_dir, target_resolution):
        super().__init__()
        self.save_dir = save_dir
        self.target_resolution = target_resolution
        # PandaSet Provides 6 cameras and 2 lidars
        self.cam_list = [          # {frame_idx}_{cam_id}.jpg
            "front_camera",        # "xxx_0.jpg"
            "front_left_camera",   # "xxx_1.jpg"
            "front_right_camera",  # "xxx_2.jpg"
            "left_camera",         # "xxx_3.jpg"
            "right_camera",        # "xxx_4.jpg"
            "back_camera"          # "xxx_5.jpg"
        ]
        # process_keys=[
        #     "images",
        #     "lidar",
        #     "calib",
        #     "pose",
        # ],
        # 0: mechanical 360° LiDAR, 1: front-facing LiDAR, -1: All LiDARs
        self.lidar_list = [-1]

        self.load_dir = load_dir
        self.save_dir = f"{save_dir}"
        self.pandaset = PandaSet(load_dir)
        self.seq_ids = self.pandaset.sequences()



    def __len__(self):
        return len(self.seq_ids)

    @staticmethod
    def project_lidar_world2camera(pc_world, img_w, img_h, camera_pose, cam_K, filter_outliers=True):
        trans_lidar_to_camera = np.linalg.inv(camera_pose)
        points3d_lidar = pc_world
        points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + \
                            trans_lidar_to_camera[:3, 3].reshape(3, 1)



        inliner_indices_arr = np.arange(points3d_camera.shape[1])
        if filter_outliers:
            condition = points3d_camera[2, :] > 0.0
            points3d_camera = points3d_camera[:, condition]
            inliner_indices_arr = inliner_indices_arr[condition]

        points2d_camera = cam_K @ points3d_camera
        points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T

        if filter_outliers:
            condition = np.logical_and(
                (points2d_camera[:, 1] < img_h) & (points2d_camera[:, 1] > 0),
                (points2d_camera[:, 0] < img_w) & (points2d_camera[:, 0] > 0))
            points2d_camera = points2d_camera[condition]
            points3d_camera = (points3d_camera.T)[condition]
            inliner_indices_arr = inliner_indices_arr[condition]
        return points2d_camera, points3d_camera, inliner_indices_arr


    def __getitem__(self, index):
        scene_idx = self.seq_ids[index]
        scene_data = self.pandaset[scene_idx]
        scene_data.load()
        num_frames = sum(1 for _ in scene_data.timestamps)
        save_scene_dir = osp.join(self.save_dir, scene_idx)
        os.makedirs(save_scene_dir, exist_ok=True)
        for sample_idx_in_scene in tqdm(range(num_frames)):
            pc_world = scene_data.lidar[sample_idx_in_scene].to_numpy()[:, :3]
            # index        x           y         z        i         t       d
            # 0       -75.131138  -79.331690  3.511804   7.0  1.557540e+09  0
            # 1      -112.588306 -118.666002  1.423499  31.0  1.557540e+09  0
            # - `i`: `float`: Reflection intensity in a range `[0,255]`
            # - `t`: `float`: Recorded timestamp for specific point
            # - `d`: `int`: Sensor ID. `0` -> mechnical 360° LiDAR, `1` -> forward-facing LiDAR

            for cam_idx, cam_name in enumerate(self.cam_list):
                frame_id = f"{sample_idx_in_scene:06d}_{cam_idx}"
                camera = scene_data.camera[cam_name]
                pil_image = camera[sample_idx_in_scene]
                image = np.array(pil_image)

                H, W = image.shape[:2]
                poses = camera.poses[sample_idx_in_scene]
                cam2world = geometry._heading_position_to_mat(poses['heading'], poses['position'])

                cam_K = camera.intrinsics
                cam_K = np.array([
                    [cam_K.fx, 0, cam_K.cx],
                    [0, cam_K.fy, cam_K.cy],
                    [0, 0, 1]
                ])



                # downscale image
                output_resolution = (self.target_resolution, 1) if W > H else (1, self.target_resolution)
                image, _, intrinsics2 = rescale_image_depthmap(image, None, cam_K, output_resolution)
                # image.save(osp.join(save_scene_dir, f"{frame_id}.jpg"), quality=80)


                W, H = image.size
                depthmap = np.zeros((H, W), dtype=np.float32)
                points2d_camera, points3d_camera, _ = self.project_lidar_world2camera(pc_world, W, H, cam2world, intrinsics2)
                x, y = points2d_camera.T
                depthmap[y.clip(min=0, max=H - 1).astype(np.int16), x.clip(min=0, max=W - 1).astype(np.int16)] = points3d_camera[:, 2]
                # cv2.imwrite(osp.join(save_scene_dir, f"{frame_id}.exr"), depthmap)



                # # Overlay depth on RGB image
                # max_depth = 50
                # depth_color = cv2.applyColorMap(
                #     (depthmap * 255 / max_depth).astype(np.uint8), cv2.COLORMAP_JET)
                # depth_color[depthmap == 0] = 0

                # image_array = np.array(image)
                # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                # overlay = cv2.addWeighted(image_array, 0.3, depth_color, 0.7, 0)
                # cv2.imwrite(osp.join(save_scene_dir, f"{frame_id}_depthcolor.png"), overlay)

                # np.savez(osp.join(save_scene_dir, f"{frame_id}.npz"),
                #         intrinsics=intrinsics2, cam2world=cam2world)

                np.savez_compressed(os.path.join(save_scene_dir, f"{frame_id}.npz"),
                    image=np.array(image),
                    depthmap=depthmap,
                    intrinsics=intrinsics2, 
                    cam2world=cam2world)

        self.pandaset.unload(scene_idx)

        return None


if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default=None)
    args = parser.parse_args()

    assert 'SCRATCH' in os.environ, "SCRATCH environment variable is not set"
    SCRATCH = os.environ['SCRATCH']
    if args.root is not None:
        pandaset_root = args.root
    else:
        pandaset_root = os.path.join(SCRATCH, 'data', 'PandaSet')
    if args.save_dir is not None:
        preprocessed_root = args.save_dir
    else:
        preprocessed_root = os.path.join(SCRATCH, 'data', "pandaset_processed")
    target_resolution = 1024

    save_dir = os.path.join(preprocessed_root)
    dataset = PandasetDataset(pandaset_root, save_dir, target_resolution=target_resolution)

    # Add stride logic using pid and nproc
    for i in tqdm(range(args.pid, len(dataset), args.nproc)):
        dataset[i]
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
    #                                             collate_fn=lambda x: x,
    #                                             shuffle=False, num_workers=args.n_workers)
    # for data in tqdm(dataloader):
    #     pass

