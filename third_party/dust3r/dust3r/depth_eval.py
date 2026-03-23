import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.utils.image import load_images
from copy import deepcopy
from scipy.optimize import minimize
import os
from collections import defaultdict
from dust3r.eval_metadata import dataset_metadata
from occany.utils.helpers import transform_points_torch
# import open3d as o3d




def eval_mono_depth_estimation(args, model, device):
    metadata = dataset_metadata.get(args.eval_dataset)
    if metadata is None:
        raise ValueError(f"Unknown dataset: {args.eval_dataset}")
    
    img_path = metadata.get('img_path')
    if 'img_path_func' in metadata:
        img_path = metadata['img_path_func'](args)
    
    process_func = metadata.get('process_func')
    if process_func is None:
        raise ValueError(f"No processing function defined for dataset: {args.eval_dataset}")
    
    for filelist, save_dir in process_func(args, img_path):
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        eval_mono_depth(args, model, device, filelist, save_dir=save_dir)


def eval_mono_depth(args, model, device, filelist, save_dir=None):
    model.eval()
    load_img_size = 512
    for file in tqdm(filelist):
        # construct the "image pair" for the single image
        file = [file]
        imgs = load_images(file, size=load_img_size, verbose=False, crop= not args.no_crop)
        imgs = [imgs[0], deepcopy(imgs[0])]
        imgs[1]['idx'] = 1

        pairs = make_pairs(imgs, symmetrize=True, prefilter=None)
        output = inference(pairs, model, device, batch_size=1, verbose=False)
        depth_map = output['pred1']['pts3d'][...,-1].mean(dim=0)

        if save_dir is not None:
            #save the depth map to the save_dir as npy
            np.save(f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.npy')}", depth_map.cpu().numpy())
            # also save the png
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            depth_map = (depth_map * 255).cpu().numpy().astype(np.uint8)
            cv2.imwrite(f"{save_dir}/{file[0].split('/')[-1].replace('.png','depth.png')}", depth_map)



## used for calculating the depth evaluation metrics


def group_by_directory(pathes, idx=-1):
    """
    Groups the file paths based on the second-to-last directory in their paths.

    Parameters:
    - pathes (list): List of file paths.

    Returns:
    - dict: A dictionary where keys are the second-to-last directory names and values are lists of file paths.
    """
    grouped_pathes = defaultdict(list)

    for path in pathes:
        # Extract the second-to-last directory
        dir_name = os.path.dirname(path).split('/')[idx]
        grouped_pathes[dir_name].append(path)
    
    return grouped_pathes


def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    
    initial_params = [s, t]  # s = 1, t = 0
    
    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))
    
    s, t = result.x  
    return s, t

def absolute_value_scaling2(predicted_depth, ground_truth_depth, 
                            conf=None,
                            optimize_shift=True,
                            s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)

    params = [s]
    if optimize_shift:
        t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
        params.append(t)
    else:
        t = torch.tensor([t_init], device=predicted_depth.device, dtype=predicted_depth.dtype)

    optimizer = torch.optim.Adam(params, lr=lr)
    
    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)
        if conf is not None:
            abs_error = abs_error * conf
        # Compute loss
        loss = torch.sum(abs_error)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()

def depth_evaluation(predicted_depth_original, ground_truth_depth_original, max_depth=80, custom_mask=None, post_clip_min=None, post_clip_max=None, pre_clip_min=None, pre_clip_max=None,
                     align_with_lstsq=False, align_with_lad=False, align_with_lad2=False, lr=1e-4, max_iters=1000, use_gpu=False, align_with_scale=False,
                     disp_input=False):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.
    
    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.
    
    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
    
    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = (ground_truth_depth_original > 0)
    
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input: # align the pred to gt in the disparity space
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if align_with_lstsq:
        # Convert to numpy for lstsq
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)
        
        # Add a column of ones for the shift term
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        
        # Solve for scale (s) and shift (t) using least squares
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        # convert to torch tensor
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)
        
        # Apply scale and shift
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(predicted_depth, ground_truth_depth, s=torch.median(ground_truth_depth) / torch.median(predicted_depth))
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=s_init, lr=lr, max_iters=max_iters)
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-8  # Add small constant to avoid division by zero
            
            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals
            
            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth ** 2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)
        
        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()
        
        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth

    else:
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    
    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()
    
    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(predicted_depth) - torch.log(ground_truth_depth)) ** 2)).item()
    
    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()
    threshold_2 = torch.mean((max_ratio < 1.25 ** 2).float()).item()
    threshold_3 = torch.mean((max_ratio < 1.25 ** 3).float()).item()

    # Compute the depth error parity map
    if align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    
    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)

    predict_depth_map_full = predicted_depth_original

    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)

    num_valid_pixels = torch.sum(mask).item() if custom_mask is None else torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        abs_rel, sq_rel, rmse, log_rmse, threshold_1, threshold_2, threshold_3 = 0, 0, 0, 0, 0, 0, 0

    results = {
        'Abs Rel': abs_rel,
        'Sq Rel': sq_rel,
        'RMSE': rmse,
        'Log RMSE': log_rmse,
        'δ < 1.25': threshold_1,
        'δ < 1.25^2': threshold_2,
        'δ < 1.25^3': threshold_3,
        'valid_pixels': num_valid_pixels,
        'scale': s,
        'shift': t
    }

    return results, depth_error_parity_map_full, predict_depth_map_full, gt_depth_map_full



def iou(pred, gt, th=.5):
    pred = pred > th
    gt = gt > th
    intersect = torch.sum(pred & gt).float()
    union = torch.sum(pred | gt).float()
    iou_score = intersect / (union + 1e-8)
    return iou_score


def occ_scaling(points, T_cam_2_velo, voxel_origin, voxel_size, voxel_label,
                init_scale=1.0, init_shift=0.0, lr=0.1, max_iters=300, tol=1e-6):
    # Initialize s and t as torch tensors with requires_grad=True
    device = points.device
    T_cam_2_velo = T_cam_2_velo.to(device)
    voxel_label = voxel_label.to(device)
    voxel_origin = voxel_origin.to(device)
    s = torch.tensor([init_scale], requires_grad=True, device=device, dtype=points.dtype)
    # t = torch.tensor([init_shift] * 3, requires_grad=True, device=device, dtype=points.dtype)
    # t = torch.tensor([init_shift], requires_grad=True, device=device, dtype=points.dtype)
    t = torch.tensor([0, 0, init_shift], requires_grad=True, device=device, dtype=points.dtype)
    
    voxel_dim = voxel_label.shape
    occ_label = torch.zeros_like(voxel_label)
    occ_label[(voxel_label > 0) & (voxel_label != 255)]  = 1.0
    valid_mask = (voxel_label != 255)
    voxel_origin = voxel_origin.reshape(1, 3)
    
    # label_points, label_colors = voxel_to_pointcloud(voxel_label.cpu().numpy(), voxel_size, voxel_origin.cpu().numpy())
    # save_pcd_as_txt("demo_tmp/label.txt", label_points, label_colors)
    
    # Store original points
    original_points = points.clone()
    original_points.requires_grad = False
    
    optimizer = torch.optim.Adam([s, t], lr=lr)
    prev_loss = None

    pbar = tqdm(range(max_iters), desc="Iterations")
    for i in pbar:
        optimizer.zero_grad()

        # Create transformation matrix for this iteration
        transformation_matrix = torch.eye(4, dtype=torch.float32, device=device)
        transformation_matrix[:3, :3] = torch.eye(3, device=device) * s
        transformation_matrix[:3, 3] = t
        # transformation_matrix[2, 3] = t
        

        # Transform points
        scaled_points = (transformation_matrix[:3, :3] @ original_points.T).T + transformation_matrix[:3, 3].unsqueeze(0)
        points_in_velo = transform_points_torch(T_cam_2_velo, scaled_points)
        
        point_features = torch.ones(points_in_velo.shape[0], 19, device=device)
        voxel, sem = pointcloud2voxel(points_in_velo, point_features, voxel_dim, voxel_origin, voxel_size)
        occ_label_valid = occ_label[valid_mask]
        voxel_valid = voxel[valid_mask]
        
        intersection = (voxel_valid * occ_label_valid).sum()
        union = (voxel_valid + occ_label_valid).sum() - intersection
        loss = 1.0 - intersection / (union + 1e-8)
        
        # Calculate IOU score (detached to avoid graph retention)
        iou_score = iou(voxel_valid, occ_label_valid, 0.0)

        # Display current iou, iter, and loss
        pbar.set_description(f'Iter {i}: iou (th=0.0): {iou_score.item():.4f}, loss: {loss.item():.4f}, scale: {s.item():.4f}, shift: {", ".join(f"{num:.4f}" for num in t.tolist())}')
        
        # Backward pass with retain_graph=False (default)
        loss.backward()
        optimizer.step()

        # Check convergence using detached loss
        if prev_loss is not None and abs(prev_loss - loss.item()) < tol:
            break

        prev_loss = loss.item()
    # Save final point cloud
    # with torch.no_grad():
    #     final_points_in_velo = transform_points_torch(T_cam_2_velo, scaled_points)
    #     save_pcd_as_txt("demo_tmp/after.txt", final_points_in_velo.cpu().numpy(), 
    #                    features=1 + np.ones_like(final_points_in_velo.cpu().numpy()))
    
    # return voxel, transformation_matrix
    return transformation_matrix.detach().cpu()


def occ_evaluation(predicted_depth_original, ground_truth_depth_original, 
                   points, T_cam_2_velo, voxel_origin, voxel_size, voxel_label,
                   optimize_iou=False, gt_scale=True, optimize_shift=True,
                   max_depth=50, use_gpu=True, max_iters=300, tol=1e-6, lr=0.1):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.
    
    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.
    
    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if isinstance(points, np.ndarray):
        points = torch.from_numpy(points)
    if isinstance(T_cam_2_velo, np.ndarray):
        T_cam_2_velo = torch.from_numpy(T_cam_2_velo).float()
    if isinstance(voxel_origin, np.ndarray):
        voxel_origin = torch.from_numpy(voxel_origin).float()
    if isinstance(voxel_size, np.ndarray):
        voxel_size = torch.from_numpy(voxel_size).float()
    if isinstance(voxel_label, np.ndarray):
        voxel_label = torch.from_numpy(voxel_label).float()
    
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
    
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = (ground_truth_depth_original > 0)
    
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    if gt_scale:
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        scale, shift = absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=s_init, optimize_shift=optimize_shift)
    else:
        scale = 1.0
        shift = 0.0
    if optimize_iou:
        points = points.cuda()
        T = occ_scaling(points, T_cam_2_velo, voxel_origin, voxel_size, voxel_label,
                       max_iters=max_iters, tol=tol, lr=lr,
                       init_scale=scale, init_shift=shift)
    else:
        T = torch.eye(4, dtype=torch.float32, device=points.device)
        T[:3, :3] = torch.eye(3, device=points.device) * scale
        T[:3, 3] = torch.tensor([0, 0, shift], device=points.device)
    return T
    
    
    
def compute_gt_depth_scale(
    predicted_depth, ground_truth_depth, 
    conf=None,
    optimize_shift=True,
    max_depth=50, use_gpu=True, max_iters=300, tol=1e-6, lr=0.1):
    """
    Compute the scale factor between predicted depth and ground truth depth.
    
    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        optimize_shift (bool): If True, optimize for both scale and shift. Default is True.
        max_depth (float): The maximum depth value to consider. Default is 50 meters.
        use_gpu (bool): If True, use GPU for computation. Default is True.
        max_iters (int): Maximum number of iterations for optimization. Default is 300.
        tol (float): Tolerance for optimization convergence. Default is 1e-6.
        lr (float): Learning rate for optimization. Default is 0.1.
    
    Returns:
        float: The computed scale factor between predicted and ground truth depth.
    """
    # Convert numpy arrays to torch tensors if needed
    if isinstance(predicted_depth, np.ndarray):
        predicted_depth = torch.from_numpy(predicted_depth)
    if isinstance(ground_truth_depth, np.ndarray):
        ground_truth_depth = torch.from_numpy(ground_truth_depth)
    
    # Move to GPU if requested
    if use_gpu and torch.cuda.is_available():
        predicted_depth = predicted_depth.cuda()
        ground_truth_depth = ground_truth_depth.cuda()
    
    # Apply depth mask
    if max_depth is not None:
        mask = (ground_truth_depth > 0) & (ground_truth_depth < max_depth)
    else:
        mask = (ground_truth_depth > 0)
        
    # if conf is not None:
    #     # Apply confidence threshold - keep only top 10% most confident pixels
    #     conf_masked = conf[mask]
    #     top_k = max(1, int(0.1 * conf_masked.size(0)))  # At least 1 pixel
    #     threshold = torch.topk(conf_masked, top_k).values[-1]
    #     mask = mask & (conf >= threshold)
    
    # Apply mask to both depth maps
    predicted_depth_masked = predicted_depth[mask]
    ground_truth_depth_masked = ground_truth_depth[mask]
    conf_masked = conf[mask]

    # print((predicted_depth_masked - ground_truth_depth_masked).abs().mean().item())
    
    # Compute initial scale estimate using median ratio
    s_init = (torch.median(ground_truth_depth_masked) / torch.median(predicted_depth_masked)).item()
    
    # Optimize scale (and shift if requested) using absolute_value_scaling2
    scale, _ = absolute_value_scaling2(predicted_depth_masked, ground_truth_depth_masked, 
                                       conf=conf_masked,
                                    #  conf=None,
                                     s_init=s_init, 
                                     optimize_shift=optimize_shift,
                                     max_iters=max_iters, 
                                     tol=tol,  
                                     lr=lr)
    mask = mask.cpu().numpy()
    error_before = torch.abs(predicted_depth_masked - ground_truth_depth_masked).mean().item()
    error_after = torch.abs((predicted_depth_masked * scale) - ground_truth_depth_masked).mean().item()
    print("Error before scaling:", error_before)
    print("Error after scaling:", error_after)
    # if error_before > 5:
    #     breakpoint()
    #     save_depth_as_colored_png(predicted_depth.cpu().numpy(), 0, "demo_data/predicted_depth.png", min_depth=0, max_depth=50)
    #     save_depth_as_colored_png(ground_truth_depth.cpu().numpy(), 0, "demo_data/ground_truth_depth.png", min_depth=0, max_depth=50)
    #     save_depth_as_colored_png(conf.cpu().numpy(), 0, "demo_data/conf.png", min_depth=0, max_depth=conf.max().item())
    return scale, mask