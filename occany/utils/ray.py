import numpy as np
import occany.utils.fusion as fusion
import torch
import cv2
from occany.utils.image_util import colorize_depth_maps, chw2hwc
from PIL import Image
import os

def save_rgb_images(rgb_tensors, idx, save_dir: str, stem: str, verbose: bool = False):
    """Save RGB tensor(s) to PNG images.

    The input tensor values are expected to be in the range ``[0, 1]``.

    Supported tensor shapes::
        (B, N, H, W, 3)
        (B, 3, H, W)
        (H, W, 3)

    For each frame an image ``{stem}_XXXX.png`` is written inside
    ``save_dir``.

    Args:
        rgb_tensors: A ``torch.Tensor`` or a list/tuple of tensors containing RGB
            data in the range ``[0, 1]``.
        save_dir: Target directory where PNG files will be stored.
        stem:     Prefix for output filenames.
        verbose:  If ``True`` prints the path of every saved image.
    """
    # Accept both single tensor and iterable of tensors
    if isinstance(rgb_tensors, (list, tuple)):
        tensors = rgb_tensors
    else:
        tensors = [rgb_tensors]

    os.makedirs(save_dir, exist_ok=True)

    for t_id, tensor in enumerate(tensors):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("rgb_tensors must be torch.Tensor or list/tuple of Tensors")

        t = tensor.detach().cpu().clamp(0.0, 1.0) # # (B, N, H, W, 3)

        # Bring to (M, H, W, 3)
        # if t.dim() == 5:  # (B, N, H, W, 3)
        #     t = t.reshape(-1, *t.shape[-3:])
        # elif t.dim() == 4:  # (B, 3, H, W)
        #     t = t.permute(0, 2, 3, 1)
        # elif t.dim() == 3:  # (H, W, 3)
        #     t = t.unsqueeze(0)
        # else:
        #     raise ValueError(f"Unsupported tensor shape: {t.shape}")
  
        # for f_id, img in enumerate(t):
        for f_id in idx:
            img = t[0, f_id]
            img_uint8 = (img * 255.0).round().to(torch.uint8).numpy()  # RGB
            img_bgr = img_uint8[..., ::-1]  # Convert to BGR for OpenCV
            filename = os.path.join(
                save_dir,
                f"{stem}_{t_id:02d}_{f_id:03d}.png" if len(tensors) > 1 else f"{stem}_{f_id:03d}.png",
            )
            cv2.imwrite(filename, img_bgr)
            if verbose:
                print(f"Saved {filename}")




def get_ray_map(c2w, intrinsics, h, w):
    # c2w = np.linalg.inv(c2w1) @ c2w2
    i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
    grid = np.stack([i, j, np.ones_like(i)], axis=-1)
    ro = c2w[:3, 3]
    rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
    rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
    rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
    ro = np.broadcast_to(ro, (h, w, 3))
    breakpoint()
    ray_map = np.concatenate([ro, rd], axis=-1)
    ray_map = np.transpose(ray_map, (2, 0, 1))
    ray_map = np.float32(ray_map)
    return ray_map


def visualize_depth(depth, save_path, colormap="Spectral",
                    min_val=0, max_val=80, mask=None):
    depth_colored = colorize_depth_maps(
        depth, min_val, max_val, cmap=colormap
    ).squeeze()  # [3, H, W], value in (0, 1)
    if mask is not None:
        depth_colored = depth_colored * mask[np.newaxis, :, :]
    depth_colored = (depth_colored * 255).astype(np.uint8)
    depth_colored_hwc = chw2hwc(depth_colored)
    depth_colored_img = Image.fromarray(depth_colored_hwc)
    # if mask is not None:
    #     for h, w in np.column_stack(np.where(mask)):
    #         cv2.circle(depth_colored_img, (w, h), radius=5, color=(0, 0, 255))
    depth_colored_img.save(save_path)



def project_lidar_world2camera(pc_world, img_w, img_h, camera_pose, cam_K, filter_outliers=True):
    """
    Projects a LiDAR point cloud from the world coordinate frame onto the camera image plane.

    This function transforms the input LiDAR points from world coordinates into the camera coordinate system by
    applying the inverse of the provided camera pose. It then projects the 3D points onto the 2D image plane using
    the camera intrinsic matrix. Optionally, it filters out points that are behind the camera or that project outside
    the image boundaries.

    Args:
        pc_world (np.ndarray): A 2D array of shape (N, 3) representing the 3D LiDAR points in world coordinates.
        img_w (int): The width of the target image in pixels.
        img_h (int): The height of the target image in pixels.
        camera_pose (np.ndarray): A 4x4 transformation matrix representing the camera pose (from camera to world coordinates).
        cam_K (np.ndarray): A 3x3 intrinsic matrix of the camera.
        filter_outliers (bool, optional): If True, filters out points with non-positive depth values or those projecting
                                          outside the image boundaries. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - depthmap (np.ndarray): A 2D array of shape (img_h, img_w) containing the depth values of the projected points.
            - points2d_camera (np.ndarray): A 2D array of shape (M, 2) with the 2D pixel coordinates of the projected points.
            - points3d_camera (np.ndarray): A 2D array of shape (M, 3) with the 3D coordinates of the points in the camera frame.
            - inliner_indices_arr (np.ndarray): A 1D array of indices corresponding to the valid inlier points in the original point cloud.
    """
    trans_lidar_to_camera = np.linalg.inv(camera_pose)
    points3d_lidar = pc_world
    points3d_camera = trans_lidar_to_camera[:3, :3] @ (points3d_lidar.T) + trans_lidar_to_camera[:3, 3].reshape(3, 1)



    inliner_indices_arr = np.arange(points3d_camera.shape[1])
    if filter_outliers:
        condition = points3d_camera[2, :] > 0.0
        points3d_camera = points3d_camera[:, condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    points2d_camera = cam_K @ points3d_camera
    points2d_camera = (points2d_camera[:2, :] / points2d_camera[2, :]).T
    points3d_camera = points3d_camera.T
    
    if filter_outliers:
        condition = np.logical_and(
            (points2d_camera[:, 1] < img_h) & (points2d_camera[:, 1] > 0),
            (points2d_camera[:, 0] < img_w) & (points2d_camera[:, 0] > 0))
        points2d_camera = points2d_camera[condition]
        points3d_camera = points3d_camera[condition]
        inliner_indices_arr = inliner_indices_arr[condition]

    # depthmap = np.zeros((img_h, img_w))
    xs = np.round(points2d_camera[:, 0]).clip(0, img_w - 1).astype(np.int32)
    ys = np.round(points2d_camera[:, 1]).clip(0, img_h - 1).astype(np.int32)
    depthmap = np.zeros((img_h, img_w))
    depthmap[ys, xs] = points3d_camera[:, 2]    
    # Before the final assignment

    # depthmap = np.full((img_h, img_w), np.inf)  # Initialize with infinity
    # for i in range(len(points2d_camera)):
    #     x, y = xs[i], ys[i]
    #     depthmap[y, x] = min(depthmap[y, x], points3d_camera[i, 2])
    # depthmap[depthmap == np.inf] = 0  # Replace unprojected pixels with 0
    
    return depthmap, points2d_camera, points3d_camera, inliner_indices_arr


def save_depth_as_colored_png(gt_depth, invalid_value, save_png_path, 
    colormap=cv2.COLORMAP_JET, min_depth=None, max_depth=None):
    """
    Save depth map as colored PNG image with invalid values set to black.

    Args:
        gt_depth (np.ndarray): Depth map array
        invalid_value (float): Value representing invalid depth
        save_png_path (str): Path to save the output PNG
        colormap (int): OpenCV colormap to use (default: JET)
        min_depth (float, optional): Minimum depth value for normalization
        max_depth (float, optional): Maximum depth value for normalization
    """
    # Normalize valid depth values to [0, 1]
    valid_mask = gt_depth > 0
    if valid_mask.any():
        min_val = min_depth if min_depth is not None else gt_depth[valid_mask].min()
        max_val = max_depth if max_depth is not None else gt_depth[valid_mask].max()
        depth_normalized = np.clip((gt_depth - min_val) / (max_val - min_val), 0, 1)
    else:
        depth_normalized = np.zeros_like(gt_depth)

    # Apply colormap and convert to 8-bit
    depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), colormap)

    # Set invalid values to black
    depth_colored[gt_depth == invalid_value] = 0

    # Save as PNG
    cv2.imwrite(save_png_path, depth_colored)



def voxel_to_pointcloud(voxel_label, voxel_size, voxel_origin, colors=None):
    occupied_mask = (voxel_label != 255) & (voxel_label != 0)
    coords = np.argwhere(occupied_mask)
    points = coords * voxel_size + voxel_origin

    point_labels = voxel_label[coords[:, 0], coords[:, 1], coords[:, 2]]
    point_colors = colors[point_labels.astype(np.uint16)]

    return points, point_colors


def save_pcd_as_txt(filename, points, features=None):
    with open(filename, 'w') as f:
        # Get number of columns from first point
        n_cols = points.shape[1]

        # Write each point's coordinates
        for i in range(len(points)):
            # Write all columns space-separated
            line = " ".join([f"{points[i,j]}" for j in range(n_cols)])

            # Add colors if provided
            if features is not None:
                line += " " + " ".join([f"{features[i,j]}" for j in range(features.shape[1])])

            f.write(line + "\n")

def voxel_to_pointcloud(voxel_label, voxel_size, voxel_origin, colors=None):
    occupied_mask = (voxel_label != 255) & (voxel_label != 0)

    coords = np.argwhere(occupied_mask)
    points = coords * voxel_size + voxel_origin

    if colors is not None:
        point_labels = voxel_label[coords[:, 0], coords[:, 1], coords[:, 2]]
        point_colors = colors[point_labels.astype(np.uint16)]
    else:
        point_colors = None

    return points, point_colors

def voxelize_points(points, voxel_origin, voxel_size, voxel_dim):
    voxelized_points = (points - voxel_origin) / voxel_size
    voxelized_points = np.floor(voxelized_points).astype(np.int32)
    keep_ind = (voxelized_points[:, 0] >= 0) & (voxelized_points[:, 0] < voxel_dim[0]) & \
               (voxelized_points[:, 1] >= 0) & (voxelized_points[:, 1] < voxel_dim[1]) & \
               (voxelized_points[:, 2] >= 0) & (voxelized_points[:, 2] < voxel_dim[2])
    voxelized_points = voxelized_points[keep_ind]
    return voxelized_points, keep_ind

def transform_points(T, points):
    h_points = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    t_points = np.matmul(T, h_points.T).T
    points = t_points[:, :3]
    return points

def transform_points_torch(T, points):
    h_points = torch.cat([points, torch.ones_like(points[:, :1], device=points.device)], dim=-1)
    t_points = torch.matmul(T, h_points.T).T
    points = t_points[:, :3]
    return points


def vox2pix(cam_E, cam_k,
            vox_origin, voxel_size,
            img_W, img_H,
            scene_size):
    """
    compute the 2D projection of voxels centroids

    Parameters:
    ----------
    cam_E: 4x4
       =camera pose in case of NYUv2 dataset
       =Transformation from camera to lidar coordinate in case of SemKITTI
    cam_k: 3x3
        camera intrinsics
    vox_origin: (3,)
        world(NYU)/lidar(SemKITTI) cooridnates of the voxel at index (0, 0, 0)
    img_W: int
        image width
    img_H: int
        image height
    scene_size: (3,)
        scene size in meter: (51.2, 51.2, 6.4) for SemKITTI and (4.8, 4.8, 2.88) for NYUv2

    Returns
    -------
    projected_pix: (N, 2)
        Projected 2D positions of voxels
    fov_mask: (N,)
        Voxels mask indice voxels inside image's FOV
    pix_z: (N,)
        Voxels'distance to the sensor in meter
    """
    # Compute the x, y, z bounding of the scene in meter
    vol_bnds = np.zeros((3,2))
    vol_bnds[:,0] = vox_origin
    vol_bnds[:,1] = vox_origin + np.array(scene_size)

    # Compute the voxels centroids in lidar cooridnates
    vol_dim = np.ceil((vol_bnds[:,1]- vol_bnds[:,0])/ voxel_size).copy(order='C').astype(int)
    xv, yv, zv = np.meshgrid(
            range(vol_dim[0]),
            range(vol_dim[1]),
            range(vol_dim[2]),
            indexing='ij'
          )
    vox_coords = np.concatenate([
            xv.reshape(1,-1),
            yv.reshape(1,-1),
            zv.reshape(1,-1)
          ], axis=0).astype(int).T

    # Project voxels'centroid from lidar coordinates to camera coordinates
    cam_pts = fusion.TSDFVolume.vox2world(vox_origin, vox_coords, voxel_size)
    cam_pts = fusion.rigid_transform(cam_pts, cam_E)

    # Project camera coordinates to pixel positions
    projected_pix = fusion.TSDFVolume.cam2pix(cam_pts, cam_k)
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]

    # Eliminate pixels outside view frustum
    pix_z = cam_pts[:, 2]
    fov_mask = np.logical_and(pix_x >= 0,
                np.logical_and(pix_x < img_W,
                np.logical_and(pix_y >= 0,
                np.logical_and(pix_y < img_H,
                pix_z > 0))))


    return projected_pix, fov_mask, pix_z


def compute_local_frustum(pix_x, pix_y, min_x, max_x, min_y, max_y, pix_z):
    valid_pix = np.logical_and(pix_x >= min_x,
                np.logical_and(pix_x < max_x,
                np.logical_and(pix_y >= min_y,
                np.logical_and(pix_y < max_y,
                pix_z > 0))))
    return valid_pix

def compute_local_frustums(projected_pix, pix_z, target, img_W, img_H, dataset, n_classes, size=4):
    """
    Compute the local frustums mask and their class frequencies

    Parameters:
    ----------
    projected_pix: (N, 2)
        2D projected pix of all voxels
    pix_z: (N,)
        Distance of the camera sensor to voxels
    target: (H, W, D)
        Voxelized sematic labels
    img_W: int
        Image width
    img_H: int
        Image height
    dataset: str
        ="NYU" or "kitti" (for both SemKITTI and KITTI-360)
    n_classes: int
        Number of classes (12 for NYU and 20 for SemKITTI)
    size: int
        determine the number of local frustums i.e. size * size

    Returns
    -------
    frustums_masks: (n_frustums, N)
        List of frustums_masks, each indicates the belonging voxels
    frustums_class_dists: (n_frustums, n_classes)
        Contains the class frequencies in each frustum
    """
    H, W, D = target.shape
    ranges = [(i * 1.0/size, (i * 1.0 + 1)/size) for i in range(size)]
    local_frustum_masks = []
    local_frustum_class_dists = []
    pix_x, pix_y = projected_pix[:, 0], projected_pix[:, 1]
    for y in ranges:
        for x in ranges:
            start_x = x[0] * img_W
            end_x = x[1] * img_W
            start_y = y[0] * img_H
            end_y = y[1] * img_H
            local_frustum = compute_local_frustum(pix_x, pix_y, start_x, end_x, start_y, end_y, pix_z)
            if dataset == "NYU":
                mask = (target != 255) & np.moveaxis(local_frustum.reshape(60, 60, 36), [0, 1, 2], [0, 2, 1])
            elif dataset == "kitti":
                mask = (target != 255) & local_frustum.reshape(H, W, D)

            local_frustum_masks.append(mask)
            classes, cnts = np.unique(target[mask], return_counts=True)
            class_counts = np.zeros(n_classes)
            class_counts[classes.astype(int)] = cnts
            local_frustum_class_dists.append(class_counts)
    frustums_masks, frustums_class_dists = np.array(local_frustum_masks), np.array(local_frustum_class_dists)
    return frustums_masks, frustums_class_dists