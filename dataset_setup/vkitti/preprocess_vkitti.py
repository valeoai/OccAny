#!/usr/bin/env python3
# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Preprocessing code for the WayMo Open dataset
# dataset at https://github.com/waymo-research/waymo-open-dataset
# 1) Accept the license
# 2) download all training/*.tfrecord files from Perception Dataset, version 1.4.2
# 3) put all .tfrecord files in '/path/to/waymo_dir'
# 4) install the waymo_open_dataset package with
#    `python3 -m pip install gcsfs waymo-open-dataset-tf-2-12-0==1.6.4`
# 5) execute this script as `python preprocess_waymo.py --waymo_dir /path/to/waymo_dir`
# --------------------------------------------------------
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tool import *

# from dust3r.utils.geometry import geotrf, inv
# from dust3r.utils.image import imread_cv2
# from dust3r.utils.parallel import parallel_processes as parallel_map
# from dust3r.datasets.utils import cropping
# from dust3r.viz import show_raw_pointcloud
from PIL import Image
# from dust3r.utils.image import imread_cv2
from tool import parallel_processes as parallel_map

def get_parser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vkitti_dir', default='data/raw/vkitti/VirtualKitti2')
    parser.add_argument('--output_dir', default='data/vkitti_processed')
    parser.add_argument('--workers', type=int, default=4)
    return parser


def main(vkitti_root, output_dir, workers=1):
    extract_frames(vkitti_root, output_dir, workers=workers)
    # make_crops(output_dir, workers=args.workers)

    # make sure all pairs are there
    # with np.load(pairs_path) as data:
    #     scenes = data['scenes']
    #     frames = data['frames']
    #     pairs = data['pairs']  # (array of (scene_id, img1_id, img2_id)

    # for scene_id, im1_id, im2_id in tqdm(pairs):
    #     for im_id in (im1_id, im2_id):
    #         path = osp.join(output_dir, scenes[scene_id], frames[im_id] + '.jpg')
    #         assert osp.isfile(path), f'Missing a file at {path=}\nDid you download all .tfrecord files?'

    # shutil.rmtree(osp.join(output_dir, 'tmp'))
    print('Done! all data generated at', output_dir)


class VKitti2:
    #            name             id         vk colors        vk-cs colors
    labels = [('Terrain'        ,  0,    [210,   0, 200],   [             ] ),
              ('Sky'            ,  1,    [ 90, 200, 255],   [ 70, 130, 180] ),
              ('Tree'           ,  2,    [  0, 199,   0],   [107, 142,  35] ),
              ('Vegetation'     ,  3,    [ 90, 240,   0],   [107, 142,  35] ),
              ('Building'       ,  4,    [140, 140, 140],   [ 70,  70,  70] ),
              ('Road'           ,  5,    [100,  60, 100],   [128,  64, 128] ),
              ('GuardRail'      ,  6,    [250, 100, 255],   [             ] ),
              ('TrafficSign'    ,  7,    [255, 255,   0],   [220, 220,   0] ),
              ('TrafficLight'   ,  8,    [200, 200,   0],   [250, 170,  30] ),
              ('Pole'           ,  9,    [255, 130,   0],   [153, 153, 153] ),
              ('Misc'           , 10,    [ 80,  80,  80],   [             ] ),
              ('Truck'          , 11,    [160,  60,  60],   [  0,   0, 142] ),
              ('Car'            , 12,    [255, 127,  80],   [  0,   0, 142] ),
              ('Van'            , 13,    [  0, 139, 139],   [  0,   0, 142] )]


def _list_sequences(db_root):
    print('>> Looking for sequences in', db_root)

    sequences = []
    scenes = [d for d in os.listdir(db_root) if os.path.isdir(os.path.join(db_root, d))]

    for scene in scenes:
        sequence_paths = [osp.join(scene, d) for d in os.listdir(osp.join(db_root, scene)) if os.path.isdir(osp.join(db_root, scene, d))]
        sequences.extend(sequence_paths)
    print(f'    found {len(sequences)} sequences')
    return sequences


def extract_frames(vkitti_root, output_dir, workers=8):
    print('>> WARNING: processing only the first sequence for testing')
    sequences = _list_sequences(vkitti_root)
    print('>> outputing result to', output_dir)
    args = [(vkitti_root, output_dir, seq) for seq in sequences]
    parallel_map(process_one_seq, args, star_args=True, workers=workers, front_num=0)


def process_one_seq(db_root, output_dir, seq):
    out_dir = osp.join(output_dir)
    out_dir = osp.join(out_dir, seq.replace("/", "-"))
    extract_frames_one_seq(osp.join(db_root, seq), out_dir)



def load_camera_intrinsics(file_path, camera_id=0):
    intrinsics_dict = {}
    with open(file_path, 'r') as file:
        # Read and ignore the header line.
        header = file.readline()
        # Process each subsequent line.
        for line in file:
            line = line.strip()
            if not line:
                continue  # Skip empty lines.
            parts = line.split()
            # Extract frame and cameraID.
            frame = int(parts[0])
            cameraID = int(parts[1])
            if cameraID != camera_id:
                continue
            # Extract intrinsic parameters.
            K00 = float(parts[2])
            K11 = float(parts[3])
            K02 = float(parts[4])
            K12 = float(parts[5])
            # Build the 3x3 intrinsic matrix.
            K = np.array([[K00, 0,   K02],
                          [0,   K11, K12],
                          [0,   0,   1]])
            # Create key as "frame_cameraID".
            intrinsics_dict[frame] = K
    return intrinsics_dict


def load_extrinsics_matrices(file_path, camera_id=0):
    matrices_dict = {}
    with open(file_path, 'r') as file:
        # Read and ignore the header line.
        header = file.readline()
        # Process each subsequent line.
        for line in file:
            if line.strip() == "":
                continue  # Skip any empty lines.
            parts = line.strip().split()
            # The first two elements are frame and cameraID.
            frame = int(parts[0])
            cameraID = int(parts[1])
            if cameraID != camera_id:
                continue
            # The remaining 16 elements correspond to the 4x4 matrix.
            matrix_values = list(map(float, parts[2:]))
            matrix = np.array(matrix_values).reshape((4, 4))
            matrices_dict[frame] = matrix
    return matrices_dict

def read_depth_map(depth_path):
    image = Image.open(depth_path)  # [H, W, rgb]
    image = np.asarray(image) / 100.0
    return image

def extract_frames_one_seq(seq_path, out_dir, resolution=1024):


    print('>> Opening', seq_path)
    print('>> Outputing to', out_dir)
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    cam_id = 0

    extrinsics = load_extrinsics_matrices(osp.join(seq_path, 'extrinsic.txt'), camera_id=0)
    intrinsics = load_camera_intrinsics(osp.join(seq_path, 'intrinsic.txt'), camera_id=0)
    for frame_id, extrinsic in tqdm(extrinsics.items()):
        cam_k = intrinsics[frame_id]
        rgb_path = osp.join(seq_path, f'frames/rgb/Camera_{cam_id}', f"rgb_{frame_id:05d}.jpg")
        image = imread_cv2(rgb_path)
        H, W = image.shape[:2]

        depth_path = osp.join(seq_path, f'frames/depth/Camera_{cam_id}', f"depth_{frame_id:05d}.png")
        depth_map = read_depth_map(depth_path)

        semantic_path = osp.join(seq_path, f'frames/classSegmentation/Camera_{cam_id}', f"classgt_{frame_id:05d}.png")
        semantic_rgb = Image.open(semantic_path)
        semantic_rgb = np.asarray(semantic_rgb, dtype=np.uint8)

        sky_color = np.array(VKitti2.labels[1][2])
        sky_mask = (semantic_rgb == sky_color).all(axis=2).astype(np.float32)

        output_resolution = (resolution, 1) if W > H else (1, resolution)
        image, depth_map, sky_mask, intrinsics2 = rescale_image_depthmap_semantic(image, depth_map, sky_mask, cam_k, output_resolution)

        mask_output_path = osp.join(out_dir, f"{frame_id:05d}_{cam_id}_sky_mask.png")
        # Image.fromarray((sky_mask * 255).astype(np.uint8)).save(mask_output_path)
        # tqdm.write(f"Saved sky mask to {mask_output_path}")

        image_path = os.path.join(out_dir, f"{frame_id:05d}_{cam_id}.jpg")
        # image.save(image_path)
        # tqdm.write(f"Saved image {image_path}")

        depth_path = os.path.join(out_dir, f"{frame_id:05d}_{cam_id}.exr")
        # cv2.imwrite(depth_path, depth_map.astype(np.float32))
        # tqdm.write(f"Saved depth EXR {depth_path}")

        cam2world = np.linalg.inv(extrinsic)
        cam_params_path = os.path.join(out_dir, f"{frame_id:05d}_{cam_id}.npz")
        # np.savez(cam_params_path, intrinsics=intrinsics2, cam2world=cam2world)
        # tqdm.write(f"Saved cam params {cam_params_path}")

        depth_vis = depth_map.clip(min=0, max=80)
        depth_colored = cv2.applyColorMap(
            (depth_vis * 255 / 80).astype(np.uint8),
            cv2.COLORMAP_JET
        )
        vis_path = osp.join(out_dir, f"{frame_id:05d}_{cam_id}_vis.png")
        # cv2.imwrite(vis_path, depth_colored)
        # tqdm.write(f"Saved depth vis {vis_path}")

        out_path = os.path.join(out_dir, f"{frame_id:05d}_{cam_id}.npz")
        np.savez_compressed(out_path,
                    image=np.array(image),
                    depthmap=depth_map.astype(np.float32),
                    sky_mask=(sky_mask * 255).astype(np.uint8),
                    intrinsics=intrinsics2, 
                    cam2world=cam2world)
        tqdm.write(f"Saved {out_path}")






# def make_crops(output_dir, workers=16, **kw):
#     tmp_dir = osp.join(output_dir, 'tmp')
#     sequences = _list_sequences(tmp_dir)
#     args = [(tmp_dir, output_dir, seq) for seq in sequences]
#     parallel_map(crop_one_seq, args, star_args=True, workers=workers, front_num=0)


# def crop_one_seq(input_dir, output_dir, seq, resolution=512):
# def crop_one_seq(input_dir, output_dir, seq, resolution=1024):
#     seq_dir = osp.join(input_dir, seq)
#     out_dir = osp.join(output_dir, seq)
#     if osp.isfile(osp.join(out_dir, '00100_1.jpg')):
#         return
#     os.makedirs(out_dir, exist_ok=True)

#     # load calibration file
#     try:
#         with open(osp.join(seq_dir, 'calib.json')) as f:
#             calib = json.load(f)
#     except IOError:
#         print(f'/!\\ Error: Missing calib.json in sequence {seq} /!\\', file=sys.stderr)
#         return

#     axes_transformation = np.array([
#         [0, -1, 0, 0],
#         [0, 0, -1, 0],
#         [1, 0, 0, 0],
#         [0, 0, 0, 1]])

#     cam_K = {}
#     cam_distortion = {}
#     cam_res = {}
#     cam_to_car = {}
#     for cam_idx, cam_info in calib:
#         cam_idx = str(cam_idx)
#         cam_res[cam_idx] = (W, H) = (cam_info['width'], cam_info['height'])
#         f1, f2, cx, cy, k1, k2, p1, p2, k3 = cam_info['intrinsics']
#         cam_K[cam_idx] = np.asarray([(f1, 0, cx), (0, f2, cy), (0, 0, 1)])
#         cam_distortion[cam_idx] = np.asarray([k1, k2, p1, p2, k3])
#         cam_to_car[cam_idx] = np.asarray(cam_info['extrinsics']).reshape(4, 4)  # cam-to-vehicle

#     frames = sorted(f[:-3] for f in os.listdir(seq_dir) if f.endswith('.jpg'))

#     # from dust3r.viz import SceneViz
#     # viz = SceneViz()
#     for frame in tqdm(frames, leave=False):
#         cam_idx = frame[-2]  # cam index
#         assert cam_idx in '12345', f'bad {cam_idx=} in {frame=}'
#         data = np.load(osp.join(seq_dir, frame + 'npz'))
#         car_to_world = data['pose']
#         W, H = cam_res[cam_idx]

#         # load depthmap
#         pos2d = data['pixels'].round().astype(np.uint16)
#         x, y = pos2d.T
#         pts3d = data['pts3d']  # already in the car frame
#         pts3d = geotrf(axes_transformation @ inv(cam_to_car[cam_idx]), pts3d)
#         # X=LEFT_RIGHT y=ALTITUDE z=DEPTH

#         # load image
#         image = imread_cv2(osp.join(seq_dir, frame + 'jpg'))



#         # downscale image
#         output_resolution = (resolution, 1) if W > H else (1, resolution)
#         image, _, intrinsics2 = cropping.rescale_image_depthmap(image, None, cam_K[cam_idx], output_resolution)
#         image.save(osp.join(out_dir, frame + 'jpg'), quality=80)

#         # save as an EXR file? yes it's smaller (and easier to load)
#         W, H = image.size
#         depthmap = np.zeros((H, W), dtype=np.float32)
#         pos2d = geotrf(intrinsics2 @ inv(cam_K[cam_idx]), pos2d).round().astype(np.int16)
#         x, y = pos2d.T
#         depthmap[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = pts3d[:, 2]
#         cv2.imwrite(osp.join(out_dir, frame + 'exr'), depthmap)

#         # save camera parametes
#         cam2world = car_to_world @ cam_to_car[cam_idx] @ inv(axes_transformation)
#         np.savez(osp.join(out_dir, frame + 'npz'), intrinsics=intrinsics2,
#                  cam2world=cam2world, distortion=cam_distortion[cam_idx])

        # viz.add_rgbd(np.asarray(image), depthmap, intrinsics2, cam2world)
    # viz.show()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(f"workers: {args.workers}")
    main(args.vkitti_dir, args.output_dir, workers=args.workers)
