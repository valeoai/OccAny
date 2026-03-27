from dgp.datasets import SynchronizedSceneDataset
import os
import cv2
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch
import argparse
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

import PIL.Image
try:
    lanczos = PIL.Image.Resampling.LANCZOS
    bicubic = PIL.Image.Resampling.BICUBIC
except AttributeError:
    lanczos = PIL.Image.LANCZOS
    bicubic = PIL.Image.BICUBIC

class ImageList:
    """ Convenience class to aply the same operation to a whole set of images.
    """

    def __init__(self, images):
        if not isinstance(images, (tuple, list, set)):
            images = [images]
        self.images = []
        for image in images:
            if not isinstance(image, PIL.Image.Image):
                image = PIL.Image.fromarray(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def to_pil(self):
        return tuple(self.images) if len(self.images) > 1 else self.images[0]

    @property
    def size(self):
        sizes = [im.size for im in self.images]
        assert all(sizes[0] == s for s in sizes)
        return sizes[0]

    def resize(self, *args, **kwargs):
        return ImageList(self._dispatch('resize', *args, **kwargs))

    def crop(self, *args, **kwargs):
        return ImageList(self._dispatch('crop', *args, **kwargs))

    def _dispatch(self, func, *args, **kwargs):
        return [getattr(im, func)(*args, **kwargs) for im in self.images]


def opencv_to_colmap_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] += 0.5
    K[1, 2] += 0.5
    return K

def colmap_to_opencv_intrinsics(K):
    """
    Modify camera intrinsics to follow a different convention.
    Coordinates of the center of the top-left pixels are by default:
    - (0.5, 0.5) in Colmap
    - (0,0) in OpenCV
    """
    K = K.copy()
    K[0, 2] -= 0.5
    K[1, 2] -= 0.5
    return K

def camera_matrix_of_crop(input_camera_matrix, input_resolution, output_resolution, scaling=1, offset_factor=0.5, offset=None):
    # Margins to offset the origin
    margins = np.asarray(input_resolution) * scaling - output_resolution
    assert np.all(margins >= 0.0)
    if offset is None:
        offset = offset_factor * margins

    # Generate new camera parameters
    output_camera_matrix_colmap = opencv_to_colmap_intrinsics(input_camera_matrix)
    output_camera_matrix_colmap[:2, :] *= scaling
    output_camera_matrix_colmap[:2, 2] -= offset
    output_camera_matrix = colmap_to_opencv_intrinsics(output_camera_matrix_colmap)

    return output_camera_matrix

def rescale_image_depthmap(image, depthmap, camera_intrinsics, output_resolution, force=True):
    """ Jointly rescale a (image, depthmap)
        so that (out_width, out_height) >= output_res
    """
    image = ImageList(image)
    input_resolution = np.array(image.size)  # (W,H)
    output_resolution = np.array(output_resolution)
    if depthmap is not None:
        # can also use this with masks instead of depthmaps
        assert tuple(depthmap.shape[:2]) == image.size[::-1]

    # define output resolution
    assert output_resolution.shape == (2,)
    scale_final = max(output_resolution / image.size) + 1e-8
    if scale_final >= 1 and not force:  # image is already smaller than what is asked
        return (image.to_pil(), depthmap, camera_intrinsics)
    output_resolution = np.floor(input_resolution * scale_final).astype(int)

    # first rescale the image so that it contains the crop
    image = image.resize(tuple(output_resolution), resample=lanczos if scale_final < 1 else bicubic)
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, output_resolution, fx=scale_final,
                              fy=scale_final, interpolation=cv2.INTER_NEAREST)

    # no offset here; simple rescaling
    camera_intrinsics = camera_matrix_of_crop(
        camera_intrinsics, input_resolution, output_resolution, scaling=scale_final)

    return image.to_pil(), depthmap, camera_intrinsics



def geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (isinstance(Trf, torch.Tensor) and isinstance(pts, torch.Tensor) and
            Trf.ndim == 3 and pts.ndim == 4):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts) + Trf[:, None, None, :d, d]
        else:
            raise ValueError(f'bad shape, not ending with 3 or 4, for {pts.shape=}')
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """ Invert a torch or numpy matrix
    """
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f'bad matrix type = {type(mat)}')


class DDADDataset(SynchronizedSceneDataset):
    def __init__(self, *args, save_dir, split, pid, nproc, target_resolution, camera2idx, **kwargs):
        super().__init__(*args, split=split, **kwargs)
        self.save_dir = save_dir
        self.split = split
        self.target_resolution = target_resolution
        self.dataset_item_index = self.dataset_item_index[pid::nproc]
        self.camera2idx = camera2idx
        print(f"pid::nproc = {pid}::{nproc}")

    def point_cloud_from_depth(self, depth, cam_K):
        """Convert a depth map to a point cloud.

        Args:
            depth: Depth map as a numpy array of shape (H, W)
            cam_K: Camera intrinsics matrix of shape (3, 3)

        Returns:
            points3d: 3D points in camera coordinates, shape (N, 3)
            pixels: Corresponding 2D pixel coordinates, shape (N, 2)
        """
        height, width = depth.shape

        # Create pixel coordinate grid
        v, u = np.indices((height, width))
        pixels = np.stack([u.flatten(), v.flatten()], axis=-1)

        # Get valid depth points
        valid_mask = depth > 0
        valid_pixels = pixels[valid_mask.flatten()]
        valid_depth = depth[valid_mask]

        # Convert to homogeneous coordinates
        homogeneous_pixels = np.concatenate([valid_pixels, np.ones((valid_pixels.shape[0], 1))], axis=1)

        # Apply inverse intrinsics
        rays = np.dot(np.linalg.inv(cam_K), homogeneous_pixels.T).T

        # Scale by depth
        points3d = rays * valid_depth.reshape(-1, 1)

        return points3d, valid_pixels

    def __len__(self):
        return len(self.dataset_item_index)

    def __getitem__(self, index):
        scene_idx, sample_idx_in_scene, datum_names = self.dataset_item_index[index]
        scene = f"{self.split}_{scene_idx}"
        save_scene_dir = osp.join(self.save_dir, scene)
        os.makedirs(save_scene_dir, exist_ok=True)
        for datum_name in [d for d in datum_names if d.startswith('camera')]:
            
            # frame_id = f"{datum_name}_{sample_idx_in_scene:06d}"
            frame_id = f"{sample_idx_in_scene:06d}_{self.camera2idx[datum_name]}"
            datum_data = self.get_datum_data(scene_idx, sample_idx_in_scene, datum_name)

            pil_image = datum_data['rgb']
            image = np.array(pil_image)
            H, W = image.shape[:2]
            depth = datum_data['depth']
            cam_K = datum_data['intrinsics']
            # extrinsics = datum_data['extrinsics'] # they don't use this in the code
            pose = datum_data['pose'].matrix

            points3d, pixels = self.point_cloud_from_depth(depth, cam_K)

            # downscale image
            output_resolution = (self.target_resolution, 1) if W > H else (1, self.target_resolution)
            image, _, intrinsics2 = rescale_image_depthmap(image, None, cam_K, output_resolution)
            # image.save(osp.join(save_scene_dir, f"{frame_id}.jpg"), quality=80)


            W, H = image.size
            downscaled_depth = np.zeros((H, W), dtype=np.float32)
            pos2d = geotrf(intrinsics2 @ inv(cam_K), pixels).round().astype(np.int16)
            x, y = pos2d.T
            downscaled_depth[y.clip(min=0, max=H - 1), x.clip(min=0, max=W - 1)] = points3d[:, 2]
            # cv2.imwrite(osp.join(save_scene_dir, f"{frame_id}.exr"), downscaled_depth)


            # # Overlay depth on RGB image
            # max_depth = 50
            # downscaled_depth_color = cv2.applyColorMap(
            #     (downscaled_depth * 255 / max_depth).astype(np.uint8), cv2.COLORMAP_JET)
            # downscaled_depth_color[downscaled_depth == 0] = 0

            # image_array = np.array(image)
            # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            # overlay = cv2.addWeighted(image_array, 0.3, downscaled_depth_color, 0.7, 0)
            # # cv2.imwrite(osp.join(self.save_dir, f"{frame_id}_overlay.png"), overlay)
            # cv2.imwrite(osp.join(save_scene_dir, f"{frame_id}.png"), overlay)

            cam2world = pose
            # np.savez(osp.join(save_scene_dir, f"{frame_id}.npz"),
            #          intrinsics=intrinsics2, cam2world=cam2world)
            np.savez_compressed(os.path.join(save_scene_dir, f"{frame_id}.npz"),
                    image=np.array(image),
                    depthmap=downscaled_depth,
                    intrinsics=intrinsics2, 
                    cam2world=cam2world)

        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, default=0)
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=2)
    parser.add_argument('--ddad_root', type=str, default=None)
    parser.add_argument('--preprocessed_root', type=str, default=None)
    args = parser.parse_args()

    preprocessed_root = args.preprocessed_root  
    ddad_root = args.ddad_root

    target_resolution = 1024

    splits = ['train', 'val']
    # splits = ['val']
    camera2idx = {
        'camera_01': "0",
        'camera_05': "1",
        'camera_06': "2",
        'camera_07': "3",
        'camera_08': "4",
        'camera_09': "5",
    }
    print(f"n_workers = {args.n_workers}")
    for split in splits:
        save_dir = os.path.join(preprocessed_root)
        dataset = DDADDataset(os.path.join(ddad_root, 'ddad.json'),
                              pid=args.pid,
                              nproc=args.nproc,
                              datum_names=('camera_01', 'camera_05',
                                           'camera_06', 'camera_07',
                                           'camera_08', 'camera_09',
                                           'lidar',
                                           ),
                              camera2idx=camera2idx,
                               target_resolution=target_resolution,
                               save_dir=save_dir,
                               generate_depth_from_datum='LIDAR',
                               split=split)

        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 collate_fn=lambda x: x,
                                                 shuffle=False, num_workers=args.n_workers)
        for data in tqdm(dataloader):
            pass

