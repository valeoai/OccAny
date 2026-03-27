import os
import sys
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch
import argparse
import warnings
import json
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
from torch.utils.data import Dataset
# from dust3r.datasets.utils import cropping
# from dust3r.utils.geometry import geotrf, inv
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tool import *
# from occany.utils.helpers import project_lidar_world2camera
from scipy.spatial.transform import Rotation

class OnceDataset(Dataset):
    def __init__(self, dataset_root, save_dir, target_resolution):
        super().__init__()
        self.dataset_root = dataset_root
        self.data_root = osp.join(self.dataset_root, 'data')
        self.seq_ids = os.listdir(self.data_root)
        self.target_resolution = target_resolution
        self.save_dir = save_dir
        self.cameras = [
            "cam01",
            "cam03",
            "cam05",
            "cam06",
            "cam07",
            "cam08",
            "cam09",
        ]
        self.lidar_name = "lidar_roof"


        # self.info_dict = defaultdict(dict)
        self.frame_list = []
        max_new_frame_id = 0
        for seq in tqdm(self.seq_ids, desc="Loading sequences"):
            anno_file_path = osp.join(self.data_root, seq, '{}.json'.format(seq))
            if not osp.isfile(anno_file_path):
                print("no annotation file for sequence {}".format(seq))
                raise FileNotFoundError
            anno_file = json.load(open(anno_file_path, 'r'))
            self.calib = dict()
            for cam_name in self.cameras:
                self.calib[cam_name] = dict()
                self.calib[cam_name]['cam_to_velo'] = np.array(anno_file['calib'][cam_name]['cam_to_velo'])
                self.calib[cam_name]['cam_intrinsic'] = np.array(anno_file['calib'][cam_name]['cam_intrinsic'])
                self.calib[cam_name]['distortion'] = np.array(anno_file['calib'][cam_name]['distortion'])
            frame_annos = anno_file['frames']
            init_timestamp = float(frame_annos[0]['frame_id'])
            for frame_anno in frame_annos:
                timestamp = float(frame_anno['frame_id'])
                frame_id = round((timestamp - init_timestamp)/100) # 100ms
                frame_anno['new_frame_id'] = f"{frame_id:06d}"
                max_new_frame_id = max(max_new_frame_id, frame_id)
                self.frame_list.append(frame_anno)
            #     self.info_dict[seq][frame_anno['frame_id']] = {
            #         'pose': frame_anno['pose'],
            #     }
            #     self.info_dict[seq][frame_anno['frame_id']]['calib'] = dict()
            #     for cam_name in self.cameras:
            #         self.info_dict[seq][frame_anno['frame_id']]['calib'][cam_name] = {
            #             'cam_to_velo': np.array(anno_file['calib'][cam_name]['cam_to_velo']),
            #             'cam_intrinsic': np.array(anno_file['calib'][cam_name]['cam_intrinsic']),
            #             'distortion': np.array(anno_file['calib'][cam_name]['distortion'])
            #         }
            #     if 'annos' in frame_anno.keys():
            #         self.info_dict[seq][frame_anno['frame_id']]['annos'] = frame_anno['annos']
            # self.info_dict[seq]['frame_list'] = sorted(frame_list)
        print(f"max_new_frame_id: {max_new_frame_id}")

    def __len__(self):
        return len(self.frame_list)

    def load_point_cloud(self, seq_id, frame_id):
        bin_path = osp.join(self.data_root, seq_id, 'lidar_roof', '{}.bin'.format(frame_id))
        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        return points

    def load_image(self, seq_id, frame_id, cam_name):
        cam_path = osp.join(self.data_root, seq_id, cam_name, '{}.jpg'.format(frame_id))
        img_buf = cv2.cvtColor(cv2.imread(cam_path), cv2.COLOR_BGR2RGB)
        return img_buf

    def undistort_image(self, image, camera_intrinsic, distortion):

        h, w = image.shape[:2]
        new_cam_intrinsic, _ = cv2.getOptimalNewCameraMatrix(camera_intrinsic,
                                        distortion,
                                        (w, h), alpha=0.0, newImgSize=(w, h))
        undistorted_image = cv2.undistort(image, camera_intrinsic,
                                        distortion,
                                        newCameraMatrix=new_cam_intrinsic)
        return undistorted_image, new_cam_intrinsic

    def pose_matrix(self, pose):
        rotation = Rotation.from_quat(pose[:4]).as_matrix()
        translation = np.array(pose[4:]).reshape(3, 1)
        pose_matrix = np.concatenate([rotation, translation], axis=1)
        pose_matrix = np.concatenate([pose_matrix, np.array([[0, 0, 0, 1]])], axis=0)
        return pose_matrix

    def __getitem__(self, index):
        frame_anno = self.frame_list[index]
        seq = frame_anno['sequence_id']
        frame_id = frame_anno['frame_id']
        new_frame_id = frame_anno['new_frame_id']
        try:
            lidar_pose = self.pose_matrix(frame_anno['pose'])
        except Exception as e:
            warnings.warn(f"Skipping {seq}/{frame_id}: error reading quaternion ({e})")
            return None
        save_scene_dir = osp.join(self.save_dir, seq)
        os.makedirs(save_scene_dir, exist_ok=True)

        pcd = self.load_point_cloud(seq, frame_id)[:, :3]

        for cam_name in self.cameras:
            calib_info = self.calib[cam_name]
            cam_to_velo = calib_info['cam_to_velo']
            cam_intrinsic = calib_info['cam_intrinsic']
            distortion = calib_info['distortion']

            image = self.load_image(seq, frame_id, cam_name) # (1020, 1920, 3)
            # save_path = osp.join("demo_tmp", f"{frame_id}_{cam_name}.jpg")
            # cv2.imwrite(save_path, image)
            image, cam_intrinsic = self.undistort_image(image, cam_intrinsic, distortion)
            # undistorted_save_path = osp.join("demo_tmp", f"{frame_id}_{cam_name}_undistorted.jpg")
            # cv2.imwrite(undistorted_save_path, undistorted_image)

            H, W = image.shape[:2]
            if H > W:
                # Portrait: crop height to match width
                margin = (H - W) // 2
                bbox_square = (0, margin, W, margin + W)
                image, _, cam_intrinsic = crop_image_depthmap(image, None, cam_intrinsic, bbox_square)
                W, H = image.size

            # downscale image
            output_resolution = (self.target_resolution, 1) if W > H else (1, self.target_resolution)
            image, _, cam_intrinsic = rescale_image_depthmap(image, None, cam_intrinsic, output_resolution)
            W, H = image.size
            depthmap, _, _, _ = project_lidar_world2camera(pcd,
                                            img_w=W, img_h=H,
                                            camera_pose=cam_to_velo, cam_K=cam_intrinsic)
            depthmap = depthmap.astype(np.float32)
            # cv2.imwrite(osp.join(save_scene_dir, f"{new_frame_id}_{cam_name}.exr"), depthmap)

            # max_depth = min(np.max(depthmap), 80)
            # depth_color = cv2.applyColorMap(
            #     (depthmap * 255 / max_depth).astype(np.uint8), cv2.COLORMAP_JET)
            # depth_color[depthmap == 0] = 0

            # # Overlay depth on RGB image
            # image_array = np.array(image)
            # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            # overlay = cv2.addWeighted(image_array, 0.3, depth_color, 0.7, 0)
            # cv2.imwrite(osp.join(save_scene_dir, f"{new_frame_id}_{cam_name}_depthcolor.png"), overlay)

            cam2world = lidar_pose @ cam_to_velo
            # np.savez(osp.join(save_scene_dir, f"{new_frame_id}_{cam_name}.npz"),
            #         intrinsics=cam_intrinsic, cam2world=cam2world)
            np.savez_compressed(osp.join(save_scene_dir, f"{new_frame_id}_{cam_name}.npz"),
                    image=np.array(image),
                    depthmap=depthmap,
                    intrinsics=cam_intrinsic, 
                    cam2world=cam2world)


        return None


if __name__ == '__main__':
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--root', type=str, default=None)
    parser.add_argument('--preprocessed_root', type=str, default=None)
    args = parser.parse_args()

    print("pid: {}, nproc: {}, n_workers: {}".format(args.pid, args.nproc, args.n_workers))

    if args.root is not None:
        dataset_root = args.root
    else:
        assert 'SCRATCH' in os.environ, "SCRATCH environment variable is not set"
        SCRATCH = os.environ['SCRATCH']
        DSDIR = os.environ['DSDIR']
        dataset_root = os.path.join(DSDIR, 'ONCE')
    if args.preprocessed_root is not None:  
        preprocessed_root = args.preprocessed_root
    else:
        preprocessed_root = os.path.join(SCRATCH, 'data', "once_processed")
    target_resolution = 1024

    save_dir = os.path.join(preprocessed_root)
    dataset = OnceDataset(dataset_root, save_dir, target_resolution=target_resolution)

    # Use DataLoader with pid/nproc stride for parallelism across workers
    indices = list(range(args.pid, len(dataset), args.nproc))
    subset = torch.utils.data.Subset(dataset, indices)
    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.n_workers,
        collate_fn=lambda x: x,  # avoid collating Nones
        pin_memory=False,
    )
    for _ in tqdm(dataloader, desc="Processing frames"):
        pass

