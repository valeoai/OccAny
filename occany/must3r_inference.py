# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
from contextlib import nullcontext
import numpy as np
import itertools
import roma
from occany.model.must3r_blocks.head import ActivationType, apply_activation
from occany.model.sam3_model import Sam3ProcessorWrapper

from dust3r.post_process import estimate_focal_knowing_depth
from depth_anything_3.utils.geometry import affine_inverse
from occany.utils.image_util import quaternion_to_matrix, camera_to_pose_encoding
import torch.nn.functional as F
from occany.utils.helpers import get_ray_map_lsvm
from dust3r.utils.geometry import geotrf
from occany.utils.helpers import depth2rgb, generate_intermediate_poses, generate_novel_straight_rotated_poses

from torch_scatter import scatter_min
import copy






@torch.autocast("cuda", dtype=torch.float32)
def postprocess(pointmaps, pose_out=None, pointmaps_activation=ActivationType.NORM_EXP, 
                compute_cam=False, compute_raymap=False, pose_type="lvsm"):
    out = {}
    channels = pointmaps.shape[-1]
    out['pts3d'] = pointmaps[..., :3]
    out['pts3d'] = apply_activation(out['pts3d'], activation=pointmaps_activation)
    if channels >= 6:
        out['pts3d_local'] = pointmaps[..., 3:6]
        out['pts3d_local'] = apply_activation(out['pts3d_local'], activation=pointmaps_activation)
    if channels == 4 or channels >= 7:
        out['conf'] = 1.0 + pointmaps[..., 6].exp()
    if channels == 10:
        eps = 1e-6
        out['rgb'] = pointmaps[..., 7:].sigmoid() * (1 - 2 * eps) + eps
        out['rgb'] = (out['rgb'] - 0.5) * 2
      
    if compute_cam:
        H, W = out['conf'].shape[-2:]
        pp = torch.tensor((W / 2, H / 2), device=out['pts3d'].device)
        focal = estimate_focal_knowing_depth(out['pts3d_local'][:, 0], pp, focal_mode='weiszfeld')        
        out['focal'] = focal[:, None].expand(-1, out['pts3d_local'].shape[1])

        batch_dims = out['pts3d'].shape[:-3]
        num_batch_dims = len(batch_dims)
        R, T = roma.rigid_points_registration(
            out['pts3d_local'].reshape(*batch_dims, -1, 3),
            out['pts3d'].reshape(*batch_dims, -1, 3),
            weights=out['conf'].reshape(*batch_dims, -1) - 1.0, compute_scaling=False)

        c2w = torch.eye(4, device=out['pts3d'].device)
        c2w = c2w.view(*([1] * num_batch_dims), 4, 4).repeat(*batch_dims, 1, 1)
        c2w[..., :3, :3] = R
        c2w[..., :3, 3] = T.view(*batch_dims, 3)
        out['c2w'] = c2w

        out['pose_trans_registered'] = c2w[..., :3, 3]
        out['pose_rotmat_registered'] = c2w[..., :3, :3]
        out['pts3d_from_local_and_pose_registered'] = torch.einsum("bnij, bnhwj -> bnhwi", out['pose_rotmat_registered'], out['pts3d_local']) + out['pose_trans_registered'][:, :, None, None, :]


    if pose_out is not None:
        if pose_out.dim() == 3:
            B, N, _ = pose_out.shape
            out['pose_trans'] = pose_out[..., :3] # bs, n_imgs, 3
            out['pose_rotmat'] = quaternion_to_matrix(pose_out[..., 3:]) # bs, n_imgs, 3, 3
            c2w_pose = torch.eye(4, device=pose_out.device).expand(B, N, 4, 4).clone()
            c2w_pose[..., :3, :3] = out['pose_rotmat']
            c2w_pose[..., :3, 3] = out['pose_trans']
            out['pose_absT_quaR'] = pose_out
        else:
            B, N, _, _ = pose_out.shape
            c2w_pose = pose_out
            out['pose_rotmat'] = c2w_pose[..., :3, :3]
            out['pose_trans'] = c2w_pose[..., :3, 3]
            out['pose_absT_quaR'] = camera_to_pose_encoding(pose_out)
        
        out['c2w_pose'] = c2w_pose
       
        out['pts3d_from_local_and_pose'] = torch.einsum("bnij, bnhwj -> bnhwi", out['pose_rotmat'], out['pts3d_local']) + out['pose_trans'][:, :, None, None, :]
    
        
    # Optionally compute ray map using c2w and focal (no intrinsics argument)
    if compute_raymap:
        assert 'c2w' in out and 'focal' in out, "compute_raymap=True requires compute_cam=True to estimate focal length."
        # Choose camera-to-world matrices depending on available pose
        c2w_mats = c2w_pose if pose_out is not None else c2w  # (B, N, 4, 4)

        # Shapes and device
        B, N, H, W = out['pts3d'].shape[:4]
        device = out['pts3d'].device

        # Normalize focal tensor to shape (B, N)
        f = out['focal']  # scalar, (B,), or (B, N)
        if f.ndim == 0:
            f = f.view(1, 1).expand(B, N)
        elif f.ndim == 1 and f.shape[0] == B:
            f = f.view(B, 1).expand(B, N)
        # else: already (B, N)

        # Precompute constants
        cx, cy = W / 2.0, H / 2.0

        fxfycxcy = torch.zeros((B, N, 4), device=device, dtype=torch.float32)
        fxfycxcy[:, :, 0] = f
        fxfycxcy[:, :, 1] = f
        fxfycxcy[:, :, 2] = cx
        fxfycxcy[:, :, 3] = cy
       
        ray_map_out = get_ray_map_lsvm(c2w_mats, fxfycxcy, H, W, device=device)
       
        out['ray_map'] = ray_map_out
    

    return out


def split_list(lst, split_size):
    return [lst[i:i + split_size] for i in range(0, len(lst), split_size)]


def split_list_of_tensors(tensor, max_bs):
    tensor_splits = []
    for s in tensor:
        if isinstance(s, list):
            tensor_splits.extend(split_list(s, max_bs))
        else:
            tensor_splits.extend(torch.split(s, max_bs))
    return tensor_splits


def stack_views(true_shape, values, max_bs=None):
    # first figure out what the unique aspect ratios are
    unique_true_shape, inverse_indices = torch.unique(true_shape, dim=0, return_inverse=True)

    # we group the values that share the same AR
    true_shape_stacks = [[] for _ in range(unique_true_shape.shape[0])]
    index_stacks = [[] for _ in range(unique_true_shape.shape[0])]
    value_stacks = [
        [[] for _ in range(unique_true_shape.shape[0])]
        for _ in range(len(values))
    ]

    for i in range(true_shape.shape[0]):
        true_shape_stacks[inverse_indices[i]].append(true_shape[i])
        index_stacks[inverse_indices[i]].append(i)

        for j in range(len(values)):
            value_stacks[j][inverse_indices[i]].append(values[j][i])

    # regroup all None values together (these typically are missing encoder features that'll be recomputed later)
    for i in range(len(true_shape_stacks)):
        # get a mask for each type of value
        none_mask = [[vl == None for vl in v[i]]
                     for v in value_stacks
                     ]
        # apply "or" on all the different types of values
        none_mask = [any([v[j] for v in none_mask]) for j in range(len(true_shape_stacks[i]))]
        if not any(none_mask) or all(none_mask):
            # there was no None or all were None skip
            continue
        not_none_mask = [not x for x in none_mask]

        def get_filtered_list(l, local_mask):
            return [v for v, m in zip(l, local_mask) if m]
        true_shape_stacks.append(get_filtered_list(true_shape_stacks[i], none_mask))
        true_shape_stacks[i] = get_filtered_list(true_shape_stacks[i], not_none_mask)

        index_stacks.append(get_filtered_list(index_stacks[i], none_mask))
        index_stacks[i] = get_filtered_list(index_stacks[i], not_none_mask)

        for j in range(len(value_stacks)):
            value_stacks[j].append(get_filtered_list(value_stacks[j][i], none_mask))
            value_stacks[j][i] = get_filtered_list(value_stacks[j][i], not_none_mask)

    # stack tensors
    true_shape_stacks = [torch.stack(true_shape_stack, dim=0) for true_shape_stack in true_shape_stacks]
    value_stacks = [
        [torch.stack(v, dim=0) if None not in v else v for v in value_stack]
        for value_stack in value_stacks
    ]

    # split all sub-tensors in blocks of max_size = max_bs
    if max_bs is not None:
        true_shape_stacks = split_list_of_tensors(true_shape_stacks, max_bs)

        index_stacks = [torch.tensor(s) for s in index_stacks]
        index_stacks = split_list_of_tensors(index_stacks, max_bs)
        index_stacks = [s.tolist() for s in index_stacks]

        value_stacks = [
            split_list_of_tensors(value_stack, max_bs)
            for value_stack in value_stacks
        ]

    # some cleaning, replace list of None by a single None
    for value_stack in value_stacks:
        for j in range(len(value_stack)):
            if isinstance(value_stack[j], list):
                if None in value_stack[j]:
                    value_stack[j] = None

    return true_shape_stacks, index_stacks, *value_stacks


def _remove_from_mem(mem_values, mem_labels, idx):
    to_keep_mask = mem_labels != idx
    B, _, D = mem_values[0].shape
    mem_values = [
        mem_value[to_keep_mask].view(B, -1, D)
        for mem_value in mem_values
    ]
    mem_labels = mem_labels[to_keep_mask].view(B, -1)
    return mem_values, mem_labels


def _restore_label_in_mem(mem_labels, old_idx_to_restore, new_idx_to_remove):
    mask = mem_labels == new_idx_to_remove
    mem_labels[mask] = old_idx_to_restore
    return mem_labels


def _update_in_mem(old_values, new_values, old_labels, new_labels, old_idx, new_idx):
    old_mask = old_labels == old_idx
    new_mask = new_labels == new_idx

    for k in range(len(old_values)):  # iterate over mem_vals
        old_values[k][old_mask] = new_values[k][new_mask]
    return old_values



def get_Nmem(mem):
    if mem is None:
        return 0
    mem_labels = mem[1]
    _, Nmem = mem_labels.shape
    return Nmem


def unstack_pointmaps(index_stacks_i, pointmaps_0_i):
    num_elements = max([max(index_stack_i) for index_stack_i in index_stacks_i]) + 1
    pointmaps_0 = [None for _ in range(num_elements)]
    for pointmaps_0_i_stack, index_stack_i in zip(pointmaps_0_i, index_stacks_i):
        out_pointmaps_0_i = {}
        for k, v in pointmaps_0_i_stack.items():
            for j in range(v.shape[0]):
                if j not in out_pointmaps_0_i:
                    out_pointmaps_0_i[j] = {}
                out_pointmaps_0_i[j][k] = v[j]

        for j in out_pointmaps_0_i.keys():
            pointmaps_0[index_stack_i[j]] = out_pointmaps_0_i[j]
    return pointmaps_0


def groupby_consecutive(data):
    """
    identify groups of consecutive numbers
    """
    if not data:
        return []
    # Sort the data to ensure consecutive numbers are adjacent
    data = sorted(data)
    result = []
    # consecutive numbers have the same (value - index)
    for k, g in itertools.groupby(enumerate(data), lambda x: x[1] - x[0]):
        group = list(map(lambda x: x[1], g))
        result.append((group[0], group[-1]))
    return result


def inference_encoder(encoder, imgs, true_shape_view,
                     max_bs=None,
                     requires_grad=False,
                     mem_raymap=None,
                     mem_pos=None,
                     mem=None,
                     mem_timesteps=None,
                     timesteps=None):
    
    def encoder_get_context():
        # inference_mode is faster and more memory efficient when gradients are disabled
        return torch.no_grad() if not requires_grad else nullcontext()

    with encoder_get_context():
        # Flatten batch for efficient encoding
        B, nimgs = imgs.shape[:2]
        
        imgs_view = imgs.reshape(B * nimgs, *imgs.shape[2:])
        tshape_view = true_shape_view.reshape(B * nimgs, *true_shape_view.shape[1:])

        # Prepare memory per image efficiently if provided
        # mem_view = None
        # mem_raymap_view = None
        # if mem is not None:
        #     mem_D = mem.shape[-1]
        #     # mem: (B, Nm, D) -> repeat for each image
        #     mem_view = mem.repeat_interleave(nimgs, dim=0)
        #     breakpoint()
        #     mem_raymap_view = mem_raymap.repeat_interleave(nimgs, dim=1)
            

        if max_bs is None:
            # Encode all at once
            if mem is not None:
                x, pos = encoder(imgs_view, tshape_view, mem=mem, 
                    mem_raymap=mem_raymap, mem_pos=mem_pos, mem_timesteps=mem_timesteps,
                    timesteps=timesteps)
            else:
                x, pos = encoder(imgs_view, tshape_view)
        else:
            raise NotImplementedError("not implement for mem_raymap yet")
            # Slice into chunks to fit memory
            x_chunks, pos_chunks = [], []
            imgs_splits = torch.split(imgs_view, max_bs)
            tshape_splits = torch.split(tshape_view, max_bs)
            if mem_view is not None:
                if isinstance(mem_view, (list, tuple)):
                    # Split each memory tensor per chunk and align by index
                    mem_splits_per_layer = [torch.split(mv, max_bs) if mv is not None else [None] * len(imgs_splits)
                                            for mv in mem_view]
                    for chunk_idx, (imgs_slice, tshape_slice) in enumerate(zip(imgs_splits, tshape_splits)):
                        mem_slice = [ms[chunk_idx] if ms is not None else None for ms in mem_splits_per_layer]
                        xi, posi = encoder(imgs_slice, tshape_slice, mem=mem_slice, mem_raymap=mem_raymap)
                        x_chunks.append(xi)
                        pos_chunks.append(posi)
                else:
                    mem_splits = torch.split(mem_view, max_bs)
                    iter_args = zip(imgs_splits, tshape_splits, mem_splits)
                    for imgs_slice, tshape_slice, mem_slice in iter_args:
                        xi, posi = encoder(imgs_slice, tshape_slice, mem=mem_slice)
                        x_chunks.append(xi)
                        pos_chunks.append(posi)
            else:
                iter_args = zip(imgs_splits, tshape_splits)
                for imgs_slice, tshape_slice in iter_args:
                    xi, posi = encoder(imgs_slice, tshape_slice)
                    x_chunks.append(xi)
                    pos_chunks.append(posi)
            x = torch.cat(x_chunks, dim=0)
            pos = torch.cat(pos_chunks, dim=0)

        return x.view(B, nimgs, *x.shape[1:]), pos.view(B, nimgs, *pos.shape[1:])



def inference_encoder_raymap(encoder, raymaps, true_shape_view,
                     max_bs=None,
                     requires_grad=False,
                     mem_raymap=None,
                     mem_pos=None,
                     mem=None,
                     mem_timesteps=None,
                     timesteps=None):

    B, nimgs = raymaps.shape[:2]
    raymaps_view = raymaps.reshape(B * nimgs, *raymaps.shape[2:])
    tshape_view = true_shape_view.reshape(B * nimgs, *true_shape_view.shape[1:])



    x, pos = encoder(raymaps_view, tshape_view, mem=mem, 
            mem_raymap=mem_raymap, mem_pos=mem_pos, mem_timesteps=mem_timesteps,
            timesteps=timesteps)

    return x, pos

def inference_img(decoder, x, pos, true_shape, mem_batches,
                  verbose=False,
                  train_decoder_skip=0,
                  timesteps=None):
    B, nimgs = x.shape[:2]
    _, _, N, D = x.shape

    # use the decoder to update the memory
    # we'll also get first pass pointmaps in pointmaps_0
    # not all images have to update the memory
    mem = None
    mem_batches = [0] + np.cumsum(mem_batches).tolist()
    
    # x_view_0 = x[:, 0:1].contiguous()
    # img_view_0 = imgs[:, 0:1].contiguous()
  

    pointmaps_0 = []
    pose_out_0 = []
    for i in range(train_decoder_skip, len(mem_batches) - 1):
        xi = x[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        posi = pos[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        true_shapei = true_shape[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        # imgsi = imgs[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        # timestepsi = timesteps[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        
    
        dec_out = decoder(xi, posi, true_shapei, mem)
        if len(dec_out) == 3:
            mem, pointmaps_0i, pose_out_0i = dec_out
        else:
            mem, pointmaps_0i = dec_out
            pose_out_0i = None
      
      
      
        pointmaps_0.append(pointmaps_0i)
        pose_out_0.append(pose_out_0i)

    # concatenate the first pass pointmaps together
    # if len(pointmaps_0) > 0:
    #     # B, mem_batches[-1] - mem_batches[train_decoder_skip], N, D
    pointmaps_0 = torch.concatenate(pointmaps_0, dim=1)
    if pose_out_0[0] is not None:
        pose_out_0 = torch.concatenate(pose_out_0, dim=1)
    else:
        pose_out_0 = None
    # else:
    #     pointmaps_0 = torch.empty((B, 0, *outshape[2:]), dtype=x.dtype, device=x.device)
    #     pose_out_0 = torch.empty((B, 0, 7), dtype=x.dtype, device=x.device)

    # render pointmaps using the accumulated memory
    assert mem is not None
    mem_vals, mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = mem
    try:
        _, Nmem, Dmem = mem_vals[-1].shape
    except Exception:
        _, Nmem, Dmem = mem_vals[0][-1].shape
    if verbose:
        print(f"Nmem={Nmem}")
   
 
    # print(mem[0][0].shape) 
    # print("mem_labels", mem[1])
    # print("mem_nimgs", mem[2])
    # print("mem_protected_imgs", mem[3])
    # print("mem_protected_tokens", mem[4])
    # breakpoint()
    # render all images (concat them in the batch dimension for efficiency)
    if pose_out_0 is not None:
        _, pointmaps, pose_out = decoder(x, pos, true_shape, mem, render=True,
                                    timesteps=timesteps)
    else:
        _, pointmaps = decoder(x, pos, true_shape, mem, render=True)
        pose_out = None
    # ray_map=ray_map, ray_map_mask=ray_map_mask,
                                    # x_view_0=x_view_0,
                                    # img_view_0=img_view_0,
  

    return pointmaps_0, pointmaps, pose_out_0, pose_out, mem, x



def inference_img_online(decoder, x, pos, true_shape, mem_batches,
                  verbose=False,
                  train_decoder_skip=0,
                  timesteps=None):
    B, nimgs = x.shape[:2]
    _, _, N, D = x.shape
    
    # use the decoder to update the memory
    # we'll also get first pass pointmaps in pointmaps_0
    # not all images have to update the memory
    mem = None
    mem_batches = [0] + np.cumsum(mem_batches).tolist()

    pointmaps_0 = []
    pose_out_0 = []
    sam_feats_0 = []
    for i in range(train_decoder_skip, len(mem_batches) - 1):
        xi = x[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        posi = pos[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        true_shapei = true_shape[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        # imgsi = imgs[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        # timestepsi = timesteps[:, mem_batches[i]:mem_batches[i + 1]].contiguous()
        
        dec_out = decoder(xi, posi, true_shapei, mem)
        if len(dec_out) == 4:
            mem, pointmaps_0i, pose_out_0i, sam_feats_i = dec_out
        elif len(dec_out) == 3:
            mem, pointmaps_0i, pose_out_0i = dec_out
            sam_feats_i = None
        else:
            mem, pointmaps_0i = dec_out
            pose_out_0i = None
            sam_feats_i = None
      
      
      
        pointmaps_0.append(pointmaps_0i)
        pose_out_0.append(pose_out_0i)
        if sam_feats_i is not None:
            sam_feats_0.append(sam_feats_i)
    
    if len(sam_feats_0) > 0:
        # Concatenate all feature maps (3 for SAM2, 4 for SAM3 with pre_neck_feat)
        num_feats = len(sam_feats_0[0])
        sam_feats_0 = tuple(
            torch.concatenate([t[i] for t in sam_feats_0], dim=1) 
            for i in range(num_feats)
        )
   
    # concatenate the first pass pointmaps together
    # if len(pointmaps_0) > 0:
    #     # B, mem_batches[-1] - mem_batches[train_decoder_skip], N, D
    pointmaps_0 = torch.concatenate(pointmaps_0, dim=1)
    if pose_out_0[0] is not None:
        pose_out_0 = torch.concatenate(pose_out_0, dim=1)
    else:
        pose_out_0 = None
   
    return pointmaps_0, pose_out_0, sam_feats_0, mem



def inference_render(decoder,
                     x, pos, true_shape, mem,
                     freeze_decoder=False,
                     verbose=False,
                     timesteps=None):
    if freeze_decoder:
        flags = [p.requires_grad for p in decoder.parameters()]
        for p in decoder.parameters(): p.requires_grad_(False)
    # x, pos are precomputed encoder outputs of shape [B, nimgs, ...]
    B, nimgs = x.shape[:2]
    _, _, N, D = x.shape
    # render pointmaps using the accumulated memory
    assert mem is not None
    mem_vals, mem_labels, mem_nimgs, mem_protected_imgs, mem_protected_tokens = mem
    try:
        _, Nmem, Dmem = mem_vals[-1].shape
    except Exception:
        _, Nmem, Dmem = mem_vals[0][-1].shape
    if verbose:
        print(f"Nmem={Nmem}")
   
 
    # print(mem[0][0].shape) 
    # print("mem_labels", mem[1])
    # print("mem_nimgs", mem[2])
    # print("mem_protected_imgs", mem[3])
    # print("mem_protected_tokens", mem[4])
    # breakpoint()
    # render all images (concat them in the batch dimension for efficiency)
    dec_out = decoder(x, pos, true_shape, mem, render=True,
                                    timesteps=timesteps)
                                    # ray_map=ray_map, ray_map_mask=ray_map_mask,
                                    # x_view_0=x_view_0,
                                    # img_view_0=img_view_0,
    if len(dec_out) == 4:
        _, pointmaps, pose_out, sam_feats = dec_out
    elif len(dec_out) == 3:
        _, pointmaps, pose_out = dec_out
        sam_feats = None
    else:
        _, pointmaps = dec_out
        pose_out = None
        sam_feats = None
    if freeze_decoder:
        for p, f in zip(decoder.parameters(), flags): p.requires_grad_(f)
    return pointmaps, pose_out, sam_feats


def prepare_imgs_or_raymaps_and_true_shape_mem_batches(views, device, is_raymap=False):

    
    if is_raymap:
        imgs_or_raymaps = [b['ray_map'] for b in views]
        imgs_or_raymaps = torch.stack(imgs_or_raymaps, dim=1).to(device)
        # ray_map_mask = [b['ray_map_mask'] for b in filtered_views]
        # ray_map_mask = torch.stack(ray_map_mask, dim=1).to(device)
    else:
        imgs_or_raymaps = [b['img'] for b in views]
        imgs_or_raymaps = torch.stack(imgs_or_raymaps, dim=1).to(device)
    B, nimgs, C, H, W, = imgs_or_raymaps.shape
    true_shape = [torch.tensor(b['true_shape']) for b in views]
    true_shape = torch.stack(true_shape, dim=1).to(device)
    mem_batches = [2]
    while sum(mem_batches) < nimgs:
        mem_batches.append(1)

    # memory_batch_views = 2
    # mem_batches = []
    # while (sum_b := sum(mem_batches)) != nimgs:
    #     size_b = min(memory_batch_views, nimgs - sum_b)
    #     mem_batches.append(size_b)

    timesteps = [b['timestep'] for b in views]
    timesteps = torch.stack(timesteps, dim=1).to(device).type_as(imgs_or_raymaps)

    # if is_distill:
    #     distill_imgs = [b['distill_img'] for b in views]
    #     distill_imgs = torch.stack(distill_imgs, dim=1).to(device)
    # else:
    #     distill_imgs = None
    
    
    return imgs_or_raymaps, true_shape, mem_batches, timesteps #, distill_imgs



def inference_occany_gen(img_views, gen_views,
                     raymap_encoder, img_encoder, 
                     decoder, decoder_gen,
                     pointmaps_activation,
                     device,
                     pred_raymap=False,
                     gen_novel_poses=False,
                     views_per_interval=2,
                     gen_forward_novel_poses_dist=1.0,
                     gen_rotate_novel_poses_angle=0,
                     num_seed_rotations=0,
                     seed_rotation_angle=None,
                     seed_translation_distance=None,
                     use_local_points_with_pose_as_pts3d=False,
                     use_raymap_only_conditioning=False,
                     raymap_batch_size=12,
                     key_to_get_pts3d='pts3d',
                     dtype=torch.float32,
                     sam_model="SAM2"):
    # Reconstruction path (frozen decoder - already in eval mode with requires_grad=False)
    with torch.autocast("cuda", dtype=dtype):
        imgs, true_shape_img, mem_batches, img_timesteps = prepare_imgs_or_raymaps_and_true_shape_mem_batches(img_views, device, is_raymap=False)

        B, nimgs, C, H, W = imgs.shape
        
        # Encoder forward - no gradients needed for frozen reconstruction
        with torch.no_grad():
            x_img, pos_img = inference_encoder(
                encoder=img_encoder,
                imgs=imgs,
                true_shape_view=true_shape_img.view(B * nimgs, 2),
                max_bs=None,
                requires_grad=False,
            )
            
            img_out_0, pose_img_out_0, sam_feats_0, mem = inference_img_online(
                decoder=decoder,
                x=x_img,
                pos=pos_img,
                true_shape=true_shape_img,
                mem_batches=mem_batches,
                verbose=False,
            )
            
            img_out, pose_img_out, sam_feats = inference_render(
                decoder=decoder,
                x=x_img,
                pos=pos_img,
                true_shape=true_shape_img,
                mem=mem,
                freeze_decoder=False,
                verbose=False,
            )
        
        # IMPORTANT: Even though created in no_grad(), these are "inference tensors"
        # that cannot be used in gradient-enabled contexts. .detach() creates new
        # tensor views that CAN participate in autograd (as leaves with no history)
        x_img = x_img.detach()
        pos_img = pos_img.detach()
        mem = [t.detach() if torch.is_tensor(t) else t for t in mem]

    # Postprocess reconstruction outputs
    with torch.autocast("cuda", dtype=torch.float32):
        # img_out_0 = postprocess(img_out_0, pose_img_out_0, 
        #                         pointmaps_activation=pointmaps_activation,
        #                         compute_cam=True)
        img_out = postprocess(img_out, pose_img_out, 
                            pointmaps_activation=pointmaps_activation,
                            compute_cam=True)

    # Extract features for generation conditioning
    pts3d = img_out['pts3d']  # B, nimgs, H, W, 3
    # pts3d_0 = img_out_0['pts3d']  # B, nimgs, H, W, 3
    conf = img_out['conf']  # B, nimgs, H, W
    focal = img_out['focal'].mean(dim=1)
    # focal_0 = img_out_0['focal'].mean(dim=1)

    rgb = torch.stack([v['img'] for v in img_views], dim=1).to(device)
    rgb = rgb.permute(0, 1, 3, 4, 2)  # B, nimgs, H, W, 3

    # Determine which projection features are enabled for the raymap encoder
    base_raymap_encoder = getattr(raymap_encoder, 'module', raymap_encoder)
    if hasattr(base_raymap_encoder, 'projection_features'):
        projection_features = base_raymap_encoder.projection_features
    else:
        # Backward-compatible default ordering
        projection_features = ['pts3d_local', 'pts3d', 'rgb', 'conf', 'sam']

    if sam_feats is not None:
        sam_feats = sam_feats[:3]
        sam_feats_resized = []
        for i in range(len(sam_feats)):
            sam_feat_var = F.interpolate(
                sam_feats[i].reshape(B * nimgs, -1, sam_feats[i].shape[3], sam_feats[i].shape[4]),
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            sam_feat_var = sam_feat_var.reshape(B, nimgs, -1, H, W).permute(0, 1, 3, 4, 2)
            sam_feats_resized.append(sam_feat_var)
    else:
        # Create zero tensors with same shape as SAM features would have
        if sam_model == "SAM2":
            # SAM2 features have dimensions [256, 64, 32]
            sam_feats_resized = [
                torch.zeros(B, nimgs, H, W, 256, device=device, dtype=pts3d.dtype),
                torch.zeros(B, nimgs, H, W, 64, device=device, dtype=pts3d.dtype),
                torch.zeros(B, nimgs, H, W, 32, device=device, dtype=pts3d.dtype),
            ]
        elif sam_model == "SAM3":
            # SAM3 features have dimensions [256, 256, 256]
            sam_feats_resized = [
                torch.zeros(B, nimgs, H, W, 256, device=device, dtype=pts3d.dtype),
                torch.zeros(B, nimgs, H, W, 256, device=device, dtype=pts3d.dtype),
                torch.zeros(B, nimgs, H, W, 256, device=device, dtype=pts3d.dtype),
            ]

    # Build pts_features in the same semantic order used by RaymapEncoderDiT
    pts_features_list = []
    if 'pts3d' in projection_features:
        pts_features_list.append(pts3d)
    if 'rgb' in projection_features:
        pts_features_list.append(rgb)
    if 'conf' in projection_features:
        pts_features_list.append(conf.unsqueeze(-1) - 1.0)
    if 'sam' in projection_features or 'sam3' in projection_features:
        pts_features_list.extend(sam_feats_resized)

    if len(pts_features_list) > 0:
        pts_features = torch.cat(pts_features_list, dim=-1)
    else:
        # No additional per-point features (e.g., pts3d_local-only conditioning)
        pts_features = torch.zeros(B, nimgs, H, W, 0, device=device, dtype=pts3d.dtype)
   
    raymap_out = None
    x_ray = None
    sam_feats_raymap = None
    recon_2_gen_mapping = None

    if gen_novel_poses:
        gen_views = []
        n_intervals = max(nimgs - 1, 0)
        recon_poses = img_out['c2w']

        # Allow novel pose generation even with single frame (nimgs=1)
        if nimgs > 0 and views_per_interval > 0:

            gen_poses, recon_2_gen_mapping = generate_intermediate_poses(
                recon_poses,
                views_per_interval,
                device,
                rotate_angle=gen_rotate_novel_poses_angle,
                forward=gen_forward_novel_poses_dist,
                num_seed_rotations=num_seed_rotations,
                seed_rotation_angle=seed_rotation_angle,
                seed_translation_distance=seed_translation_distance,
            )
            n_gen_views = gen_poses.shape[1]

            for v in range(n_gen_views):
                view = copy.deepcopy(img_views[0])  # use the first view as template
                view['camera_pose'] = gen_poses[:, v]
                view['ray_map_mask'] = np.array([1.0], dtype=np.float32)
                view['is_raymap'] = True
                gen_views.append(view)
    
    if gen_views is not None and len(gen_views) > 0:        
        raymap_c2w = torch.stack([v['camera_pose'] for v in gen_views], dim=1).to(device)
        # Batch the conditioning creation to reduce memory usage
        # raymap_c2w shape: [B, nraymaps, 4, 4], we batch over nraymaps dimension
        B, nraymaps = raymap_c2w.shape[:2]
        if raymap_batch_size is None or raymap_batch_size <= 0:
            raymap_batch_size_eff = nraymaps
        else:
            raymap_batch_size_eff = min(raymap_batch_size, nraymaps)
        for raymap_start in range(0, nraymaps, raymap_batch_size_eff):
            raymap_end = min(raymap_start + raymap_batch_size_eff, nraymaps)
            gen_views_batch = gen_views[raymap_start:raymap_end]
            # Get raymap slices (batch over dimension 1)
            raymap_c2w_batch = raymap_c2w[:, raymap_start:raymap_end]
            
            # Create conditioning for this batch of raymaps
            cond_features_batch = create_gen_conditioning(pts3d, pts_features, focal, raymap_c2w_batch, 
                raymap_views=gen_views_batch,
                use_raymap_only_conditioning=use_raymap_only_conditioning,
                projection_features=projection_features)
            
            _, true_shape_raymap, _, raymap_timesteps = prepare_imgs_or_raymaps_and_true_shape_mem_batches(
                gen_views_batch, device, is_raymap=False)
        
            raymaps_batch = cond_features_batch.permute(0, 1, 4, 2, 3).detach()
            B, nraymaps_batch = raymaps_batch.shape[:2]

            # Generation path with gradients (trainable decoder_gen)
            with torch.autocast("cuda", dtype=dtype):
                # Use detached tensors from frozen reconstruction
                x_ray, pos_ray = inference_encoder_raymap(
                    encoder=raymap_encoder,
                    raymaps=raymaps_batch,
                    true_shape_view=true_shape_raymap.view(B * nraymaps_batch, 2),
                    max_bs=None,
                    requires_grad=True,
                    mem=x_img,  # Detached above - can now be used in autograd
                    mem_pos=pos_img,  # Detached above - can now be used in autograd
                    mem_timesteps=img_timesteps,
                    timesteps=raymap_timesteps,
                )

                raymap_out_batch, pose_raymap_out, sam_feats_raymap_batch = inference_render(
                    decoder=decoder_gen,
                    x=x_ray,
                    pos=pos_ray,
                    true_shape=true_shape_raymap,
                    mem=mem,  # Detached above - can now be used in autograd
                    freeze_decoder=False,
                    verbose=False,
                )

            with torch.autocast("cuda", dtype=torch.float32):
                # raymap_out = postprocess(raymap_out, pose_raymap_out, 
                #                 pointmaps_activation=pointmaps_activation,
                #                 compute_cam=True)
                raymap_out_batch = postprocess(raymap_out_batch, 
                                pose_out=raymap_c2w_batch, 
                                pointmaps_activation=pointmaps_activation,
                                compute_cam=True)
                required_raymap_keys = {'pts3d', 'pts3d_local', 'conf', 'focal', 'c2w'}
                if key_to_get_pts3d != 'pts3d':
                    required_raymap_keys.add(key_to_get_pts3d)
                raymap_out_batch = {
                    k: v
                    for k, v in raymap_out_batch.items()
                    if k in required_raymap_keys
                }
                # if use_local_points_with_pose_as_pts3d:
                #     raymap_out['pts3d'] = raymap_out['pts3d_from_local_and_pose']
                raymap_out_batch["c2w_input"] = raymap_c2w_batch

            if raymap_out is None:
                raymap_out = {}
                for k, v in raymap_out_batch.items():
                    if torch.is_tensor(v):
                        raymap_out[k] = torch.empty(
                            (B, nraymaps, *v.shape[2:]),
                            device=v.device,
                            dtype=v.dtype,
                        )
            for k, v in raymap_out_batch.items():
                if torch.is_tensor(v):
                    raymap_out[k][:, raymap_start:raymap_end] = v

            if sam_feats_raymap_batch is not None:
                if sam_feats_raymap is None:
                    sam_feats_raymap = [
                        torch.empty(
                            (B, nraymaps, *feat.shape[2:]),
                            device=feat.device,
                            dtype=feat.dtype,
                        )
                        for feat in sam_feats_raymap_batch
                    ]
                for j, feat in enumerate(sam_feats_raymap_batch):
                    sam_feats_raymap[j][:, raymap_start:raymap_end] = feat

            del cond_features_batch, raymaps_batch, raymap_out_batch
            
    
    return img_out, raymap_out, x_ray, sam_feats, sam_feats_raymap, recon_2_gen_mapping


    

def inference_occany(img_views, gen_views,
                     raymap_encoder, img_encoder, decoder,
                     pointmaps_activation,
                     device,
                     pred_raymap=False,
                     gen_novel_poses=False,
                     views_per_interval=2,
                     gen_rotate_novel_poses_angle=0,
                     gen_forward_novel_poses_dist=1.0,
                     num_seed_rotations=0,
                     seed_rotation_angle=None,
                     seed_translation_distance=None,
                     dtype=torch.float32,
                     sam_model="SAM2",
                     encoder_requires_grad=False):
    with torch.autocast("cuda", dtype=dtype):
       
        imgs, true_shape_img, mem_batches, img_timesteps = prepare_imgs_or_raymaps_and_true_shape_mem_batches(img_views, device, is_raymap=False)
    

        B, nimgs, C, H, W = imgs.shape
        x_img, pos_img = inference_encoder(
            encoder=img_encoder,
            imgs=imgs,
            true_shape_view=true_shape_img.view(B * nimgs, 2),
            max_bs=None,
            requires_grad=encoder_requires_grad,
        )
        
        img_out_0, pose_img_out_0, sam_feats_0, mem = inference_img_online(
            decoder=decoder,
            x=x_img,
            pos=pos_img,
            true_shape=true_shape_img,
            mem_batches=mem_batches,
            verbose=False,
        )

        # 1. get pts3d from image out
        with torch.autocast("cuda", dtype=torch.float32):
            img_out_0 = postprocess(img_out_0, pose_img_out_0, 
                                    pointmaps_activation=pointmaps_activation,
                                    compute_cam=True)

        if pred_raymap:
            pts3d_0 = img_out_0['pts3d'] # B, nimgs, H, W, 3
            conf_0 = img_out_0['conf'] # B, nimgs, H, W
            focal_0 = img_out_0['focal'].mean(dim=1)
            

            rgb_0 = torch.stack([v['img'] for v in img_views], dim=1).to(device)
            rgb_0 = rgb_0.permute(0, 1, 3, 4, 2) # B, nimgs, H, W, 3
            
            sam_feats_0_resized = []
            for i in range(len(sam_feats_0)):
                sam_feat_var = F.interpolate(
                    sam_feats_0[i].reshape(B * nimgs, -1, sam_feats_0[i].shape[3], sam_feats_0[i].shape[4]),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )
                sam_feat_var = sam_feat_var.reshape(B, nimgs, -1, H, W).permute(0, 1, 3, 4, 2)
                sam_feats_0_resized.append(sam_feat_var)

            # Determine which projection features are enabled for the raymap encoder
            if hasattr(raymap_encoder, 'projection_features'):
                projection_features = raymap_encoder.projection_features
            else:
                projection_features = ['pts3d_local', 'pts3d', 'rgb', 'conf', 'sam']

            pts_features_list = []
            if 'pts3d' in projection_features:
                pts_features_list.append(pts3d_0)
            if 'rgb' in projection_features:
                pts_features_list.append(rgb_0)
            if 'conf' in projection_features:
                pts_features_list.append(conf_0.unsqueeze(-1) - 1.0)
            if 'sam' in projection_features or 'sam3' in projection_features:
                pts_features_list.extend(sam_feats_0_resized)

            if len(pts_features_list) > 0:
                pts_features = torch.cat(pts_features_list, dim=-1)
            else:
                pts_features = torch.zeros(B, nimgs, H, W, 0, device=device, dtype=pts3d_0.dtype)

            if gen_novel_poses:
                gen_views = []
                n_intervals = max(nimgs - 1, 0)
                n_gen_views = 0
                recon_poses = img_out_0['c2w']

                if views_per_interval > 0:
                    if n_intervals > 0:
                        gen_poses = generate_intermediate_poses(
                            recon_poses,
                            views_per_interval,
                            device,
                            rotate_angle=gen_rotate_novel_poses_angle,
                            num_seed_rotations=num_seed_rotations,
                            seed_rotation_angle=seed_rotation_angle,
                            seed_translation_distance=seed_translation_distance,
                        )
                    else:
                        gen_poses = generate_novel_straight_rotated_poses(
                            recon_poses,
                            views_per_interval,
                            device,
                            forward=gen_forward_novel_poses_dist,
                            rotate_angle=gen_rotate_novel_poses_angle,
                            lateral_translation=seed_translation_distance if seed_translation_distance is not None else 0.0,
                        )
                

                    n_gen_views = gen_poses.shape[1]

                    for v in range(n_gen_views):
                        view = copy.deepcopy(img_views[0])  # use the first view as template

                        view['camera_pose'] = gen_poses[:, v]
                        ray_map_mask = np.array([1.0], dtype=np.float32)
                        view['ray_map_mask'] = ray_map_mask
                        view['is_raymap'] = True
                        gen_views.append(view)
                        
            raymap_c2w = torch.stack([v['camera_pose'] for v in gen_views], dim=1).to(device)
            cond_features = create_gen_conditioning(pts3d_0, pts_features, focal_0, raymap_c2w, 
                                                    raymap_views=gen_views, visualize=True)

            _, true_shape_raymap, mem_batches, raymap_timesteps = prepare_imgs_or_raymaps_and_true_shape_mem_batches(
                gen_views, device, is_raymap=False)
            
            raymaps = cond_features.permute(0, 1, 4, 2, 3).detach()
            # raymaps = torch.cat([raymaps, cond_features], dim=2)
            B, nraymaps = raymaps.shape[:2]

            x_ray, pos_ray = inference_encoder_raymap(
                encoder=raymap_encoder,
                raymaps=raymaps,
                true_shape_view=true_shape_raymap.view(B * nraymaps, 2),
                max_bs=None,
                requires_grad=True,
                mem=x_img.detach().clone(),
                mem_pos=pos_img.detach().clone(),
                mem_timesteps=img_timesteps,
                timesteps=raymap_timesteps,
                # mem=mem[0][0:6], # mem[0][decoder layer_idx]
                # mem_raymap=img_out['ray_map'].detach().clone(),
                # mem=mem[0][0],
            )
        
            x_img_and_ray = torch.cat([x_img, x_ray], dim=1)
            pos_img_and_ray = torch.cat([pos_img, pos_ray], dim=1)
            true_shape_img_and_raymap = torch.cat([true_shape_img, true_shape_raymap], dim=1)

            
        else:
            x_img_and_ray = x_img
            pos_img_and_ray = pos_img
            true_shape_img_and_raymap = true_shape_img
            nraymaps = 0
            x_ray = None
            

        img_and_raymap_out, pose_img_and_raymap_out, sam_feats_img_and_raymap = inference_render(
            decoder=decoder,
            x=x_img_and_ray,
            pos=pos_img_and_ray,
            true_shape=true_shape_img_and_raymap,
            mem=mem,
            freeze_decoder=False,
            verbose=False,
            # timesteps=timesteps,
        )

    with torch.autocast("cuda", dtype=torch.float32):
        img_out, raymap_out = img_and_raymap_out[:, :nimgs], img_and_raymap_out[:, nimgs:]
        if pose_img_and_raymap_out is not None:
            pose_img_out, pose_raymap_out = pose_img_and_raymap_out[:, :nimgs], pose_img_and_raymap_out[:, nimgs:]
        else:
            pose_img_out, pose_raymap_out = None, None

        img_out = postprocess(img_out, pose_img_out, 
                            pointmaps_activation=pointmaps_activation,
                            compute_cam=True)
        if pred_raymap:
            raymap_out = postprocess(raymap_out, pose_raymap_out, 
                            pointmaps_activation=pointmaps_activation,
                            compute_cam=True)
            raymap_out["c2w_input"] = raymap_c2w
        else:
            raymap_out = None

    return img_out_0, img_out, raymap_out, x_ray, sam_feats_0, sam_feats_img_and_raymap

def create_gen_conditioning(pts3d, pts_features, focal, 
                            raymap_c2w,
                            return_projected_pts3d=False,
                            raymap_views=None, visualize=False,
                            use_raymap_only_conditioning=False,
                            projection_features=None):
    # raymap_views = img_views
    # B, n_raymaps, 4, 4
    # nraymaps = len(raymap_views)
    device = pts3d.device
    proj_dtype = pts3d.dtype
    raymap_c2w = raymap_c2w.to(device=device, dtype=proj_dtype)
    focal = focal.to(device=device, dtype=proj_dtype)
    if pts_features is not None:
        pts_features = pts_features.to(device=device, dtype=proj_dtype)

    B, nraymaps = raymap_c2w.shape[:2]
    H, W = pts3d.shape[2], pts3d.shape[3]
    feature_dim = pts_features.shape[-1]
    
    # If use_raymap_only_conditioning is True, return raymap computed from camera poses
    if use_raymap_only_conditioning:
        # Use focal to compute fxfycxcy
        cx, cy = W / 2.0, H / 2.0
        fxfycxcy = torch.zeros((B, nraymaps, 4), device=device, dtype=raymap_c2w.dtype)
        fxfycxcy[:, :, 0] = focal.unsqueeze(1)  # fx
        fxfycxcy[:, :, 1] = focal.unsqueeze(1)  # fy
        fxfycxcy[:, :, 2] = cx
        fxfycxcy[:, :, 3] = cy
        
        # get_ray_map_lsvm returns [b, v, 6, h, w] (oxd + ray_d)
        ray_map = get_ray_map_lsvm(raymap_c2w, fxfycxcy, H, W, device=device)
        # Rearrange to [B, nraymaps, H, W, 6] to match cond_features format
        ray_map = ray_map.permute(0, 1, 3, 4, 2)  # [B, nraymaps, H, W, 6]
        return ray_map
    # raymap_c2w = torch.stack([v['camera_pose'] for v in raymap_views], dim=1).to(device)
    raymap_w2c = affine_inverse(raymap_c2w)
    
    pts3d = pts3d.reshape(B, -1, 3).unsqueeze(1).expand(-1, nraymaps, -1, -1)
    # rgb_0 = rgb_0.reshape(B, -1, 3).unsqueeze(1).expand(-1, nraymaps, -1, -1)
    # conf_0 = conf_0.reshape(B, -1).unsqueeze(1).expand(-1, nraymaps, -1)
    if feature_dim > 0:
        pts_features = pts_features.reshape(B, -1, feature_dim).unsqueeze(1).expand(-1, nraymaps, -1, -1)
    else:
        pts_features = None
    pts3d_in_raymap_poses = geotrf(raymap_w2c, pts3d)

    # Test with gt camera intrinsics
    # focal =  views[0]['camera_intrinsics'][0, 0, 0]
    # cx = views[0]['camera_intrinsics'][0, 0, 2]
    # cy = views[0]['camera_intrinsics'][0, 1, 2]

    cx = W / 2
    cy = H / 2

    cam_k_estimated = torch.zeros(B, 3, 3, device=device, dtype=pts3d.dtype)
    cam_k_estimated[:, 0, 0] = focal
    cam_k_estimated[:, 1, 1] = focal
    cam_k_estimated[:, 0, 2] = cx
    cam_k_estimated[:, 1, 2] = cy
    cam_k_estimated[:, 2, 2] = 1
    cam_k_estimated = cam_k_estimated.unsqueeze(1).expand(-1, nraymaps, -1, -1)

    pts_cam = torch.einsum("brij, brnj -> brni", cam_k_estimated, pts3d_in_raymap_poses)
    pts_2d = pts_cam[..., :2] / (pts_cam[..., 2:3] + 1e-8)
    cond_features = torch.zeros(B, nraymaps, H, W, 3 + feature_dim, device=device, dtype=pts3d.dtype)
    cond_pointmap = cond_features[..., :3]
    
    

    # Convert to integer coordinates and create validity mask
    pts_2d_int = pts_2d.round().long()
    valid_mask = (
        (pts_2d_int[..., 0] >= 0) & (pts_2d_int[..., 0] < W) &
        (pts_2d_int[..., 1] >= 0) & (pts_2d_int[..., 1] < H) &
        (pts_cam[..., 2] > 0)
    )

    # Apply mask to filter valid points only
    B, nraymaps, N = pts_2d_int.shape[:3]

    # Process each batch and raymap separately to avoid OOM
    for b in range(B):
        for r in range(nraymaps):
            valid_mask_br = valid_mask[b, r]  # [N]
            if not valid_mask_br.any():
                continue
                
            # Get coordinates and values for valid points in this batch/raymap
            pts_2d_valid = pts_2d_int[b, r][valid_mask_br]  # [num_valid, 2]
            pts_3d_valid = pts3d_in_raymap_poses[b, r][valid_mask_br]  # [num_valid, 3]
            if feature_dim > 0:
                features_valid = pts_features[b, r][valid_mask_br]  # [num_valid, feature_dim]
            depth_valid = pts_cam[b, r, :, 2][valid_mask_br]  # [num_valid]
            
            # Calculate linear pixel indices for valid coordinates
            linear_indices = pts_2d_valid[:, 1] * W + pts_2d_valid[:, 0]  # [num_valid]
            
            # Use scatter_min to find closest point per pixel
            total_pixels = H * W
            min_depths, argmin_indices = scatter_min(depth_valid, linear_indices, dim_size=total_pixels)
            
            # Create output mask for pixels that received points
            output_mask = (min_depths < float('inf')) & (min_depths > 0)
            if output_mask.any():
                # Get the 3D points and features corresponding to minimum depths
                closest_points = pts_3d_valid[argmin_indices[output_mask]]

                pixel_indices = output_mask.nonzero(as_tuple=False).squeeze(-1)
                y_coords = torch.div(pixel_indices, W, rounding_mode='floor')
                x_coords = pixel_indices % W
                cond_pointmap[b, r, y_coords, x_coords] = closest_points

                if feature_dim > 0:
                    closest_features = features_valid[argmin_indices[output_mask]]
                    cond_features[b, r, y_coords, x_coords, 3:] = closest_features

    # If 'raymap' is in projection_features, compute raymap and insert it after pts3d_local
    if projection_features is not None and 'raymap' in projection_features:
        # Use focal to compute fxfycxcy
        cx, cy = W / 2.0, H / 2.0
        fxfycxcy = torch.zeros((B, nraymaps, 4), device=device, dtype=raymap_c2w.dtype)
        fxfycxcy[:, :, 0] = focal.unsqueeze(1)  # fx
        fxfycxcy[:, :, 1] = focal.unsqueeze(1)  # fy
        fxfycxcy[:, :, 2] = cx
        fxfycxcy[:, :, 3] = cy
        
        # get_ray_map_lsvm returns [b, v, 6, h, w] (oxd + ray_d)
        ray_map = get_ray_map_lsvm(raymap_c2w, fxfycxcy, H, W, device=device)
        # Rearrange to [B, nraymaps, H, W, 6] to match cond_features format
        ray_map = ray_map.permute(0, 1, 3, 4, 2)  # [B, nraymaps, H, W, 6]
        
        # Concatenate raymap with cond_features
        # cond_features is [B, nraymaps, H, W, 3 + feature_dim] where first 3 is pts3d_local (depth)
        # We insert raymap after the first 3 channels (pts3d_local) to make it [pts3d_local(3), raymap(6), pts_features]
        cond_features = torch.cat([cond_features[..., :3], ray_map, cond_features[..., 3:]], dim=-1)

    if return_projected_pts3d:
        return cond_pointmap

    # visualize = True
    # Visualization code
    if visualize and raymap_views is not None:
        import os
        os.makedirs('demo_data', exist_ok=True)
        for batch_id in range(B):
            pred_depth = cond_pointmap[batch_id][..., 2]
            pred_col = torch.cat([pred_depth[i] for i in range(nraymaps)], dim=0)  # (N*H, W)
            
            # Visualize input images if available
            if 'img' in raymap_views[0]:
                img = torch.cat([v['img'][batch_id] for v in raymap_views], dim=1)
                img = (img.permute(1, 2, 0) + 1.0) / 2 * 255
                img_np = img.cpu().numpy()
            else:
                img_np = None
            
            pred_depth_rgb = depth2rgb(pred_col.detach().cpu().numpy(), 
                                      valid_mask=pred_col.detach().cpu().numpy() > 0, 
                                      min_depth=0.1, max_depth=50)
            
            if img_np is not None:
                combined_rgb = np.concatenate([img_np, pred_depth_rgb], axis=1)
            else:
                combined_rgb = pred_depth_rgb
                
            from PIL import Image
            Image.fromarray((combined_rgb).astype(np.uint8)).save(f"demo_data/depth_{batch_id}.png")
            print(f"Saved visualization to demo_data/depth_{batch_id}.png")


    return cond_features




def loss_of_one_batch_occany_gen(views, raymap_encoder, img_encoder, 
                                 decoder, decoder_gen, 
                             criterion, criterion_gen,
                             device, pointmaps_activation, 
                             symmetrize_batch=False, dtype=torch.float32,
                             not_pred_raymap=False,
                             loss_enc_feat=False,
                             ret=None,
                             distill_criterion=None, 
                             distill_model=None, 
                             is_distill=True,
                             use_raymap_only_conditioning=False,
                             sam_model="SAM2"):

    if raymap_encoder is None:
        pred_raymap = False
    else:
        pred_raymap = not not_pred_raymap
    


    # with torch.cuda.amp.autocast(enabled=bool(use_amp)):
    if isinstance(views[0]['is_raymap'], bool):
        raymap_views = [b for b in views if b['is_raymap']]
        img_views = [b for b in views if not b['is_raymap']]
    else:
        raymap_views = [b for b in views if (b['is_raymap'] == True).all()]
        img_views = [b for b in views if (b['is_raymap'] == False).all()]

    B = img_views[0]['img'].shape[0]
    nraymaps, nimgs = len(raymap_views), len(img_views)
    img_out, raymap_out, x_ray, sam_feats, sam_feats_raymap, _ = inference_occany_gen(
                     img_views, raymap_views, 
                     raymap_encoder, img_encoder, 
                     decoder, decoder_gen,
                     pointmaps_activation,
                     device,
                     pred_raymap=pred_raymap,
                     use_raymap_only_conditioning=use_raymap_only_conditioning,
                     dtype=dtype,
                     sam_model=sam_model)


    with torch.autocast("cuda", dtype=torch.float32):
        # NOTE: This still correct with the loss as in_camera0 is the same for both "views"
        gt_recon = img_views
        pred_recon = img_out
        
        # For visualization 
        combined_gt = img_views + raymap_views
        combined_preds = concat_preds(img_out, raymap_out)
        
        gt_gen = raymap_views
        pred_gen = raymap_out
            
        if criterion is not None:
            # loss_recon = criterion(gt_recon, pred_recon)
            # total_loss = 0.0
            # details = {f"{k}_recon": v for k, v in loss_recon[1].items()}
            # total_loss = loss_recon[0]
            details = {}

        
            loss_gen = criterion_gen(gt_gen, pred_gen)
            details.update({f"{k}_gen": v for k, v in loss_gen[1].items()})
            total_loss = loss_gen[0]
            if loss_enc_feat:
                img_of_raymaps, true_shape_raymap, _, _ = prepare_imgs_or_raymaps_and_true_shape_mem_batches(raymap_views, device, is_raymap=False)
                x_img_of_raymaps, pos_img_of_raymaps = inference_encoder(
                    encoder=img_encoder,
                    imgs=img_of_raymaps,
                    true_shape_view=true_shape_raymap.view(B * nraymaps, 2),
                    max_bs=None,
                    requires_grad=False,
                )
                # # Block gradients only to x_img_of_raymaps branch
                loss_enc_feat = F.mse_loss(x_ray, x_img_of_raymaps.detach().clone())
                details["enc_feat"] = loss_enc_feat
                total_loss = total_loss + loss_enc_feat
                
            if distill_criterion is not None:
                distill_imgs_gen = [b['distill_img'] for b in raymap_views]
                distill_imgs_gen = torch.stack(distill_imgs_gen, dim=1).to(device)
                
                with torch.no_grad():
                    distill_input = distill_imgs_gen.reshape(
                        B * nraymaps, 3, distill_imgs_gen.shape[-2], distill_imgs_gen.shape[-1]
                    )
                    if isinstance(distill_model, Sam3ProcessorWrapper):
                        distill_feats_gen = distill_model.forward_distill(distill_input)
                    else:
                        distill_feats_gen = distill_model.forward(distill_input)
                distill_feats_gen = [distill_feat.view(B, nraymaps, *distill_feat.shape[1:]).detach() for distill_feat in distill_feats_gen]
                # distill_feats_recon = [torch.cat([distill_feat[:, :nimgs], distill_feat[:, :nimgs]], dim=1) for distill_feat in distill_feats]
             
                # sam_feats_recon = [
                #     torch.cat([sam_feats_0[0], sam_feats_img_and_raymap[0][:, :nimgs]], dim=1),
                #     torch.cat([sam_feats_0[1], sam_feats_img_and_raymap[1][:, :nimgs]], dim=1),
                #     torch.cat([sam_feats_0[2], sam_feats_img_and_raymap[2][:, :nimgs]], dim=1),
                # ]
                
                # if distill_criterion.use_conf:
                #     loss_distill_recon = distill_criterion(sam_feats_recon, distill_feats_recon, pred_recon['conf'].detach())
                # else:
                #     loss_distill_recon = distill_criterion(sam_feats_recon, distill_feats_recon)
                # details.update({f"{k}_distill_recon": v for k, v in loss_distill_recon[1].items()})
                # total_loss = total_loss + loss_distill_recon[0]
                # if pred_raymap:
                # sam_feats_gen = [
                #     sam_feats_img_and_raymap[0][:, nimgs:],
                #     sam_feats_img_and_raymap[1][:, nimgs:],
                #     sam_feats_img_and_raymap[2][:, nimgs:],
                # ]
                sam_feats_gen = sam_feats_raymap
                if distill_criterion.use_conf:
                    loss_distill_gen = distill_criterion(sam_feats_gen, distill_feats_gen, pred_gen['conf'].detach())
                else:
                    loss_distill_gen = distill_criterion(sam_feats_gen, distill_feats_gen)
                details.update({f"{k}_distill_gen": v for k, v in loss_distill_gen[1].items()})
                total_loss = total_loss + loss_distill_gen[0]
            loss = (total_loss, details)
        else:
            loss = None
        
        result = dict(loss=loss, views=views, 
                      raymap_preds=raymap_out, 
                    #   img_preds_0=img_out_0,
                      img_preds=img_out,
                      gt_img=img_views,
                      gt_raymap=raymap_views,
                      combined_gt=combined_gt,
                      combined_preds=combined_preds)
    return result

                    

def loss_of_one_batch_occany(views, raymap_encoder, img_encoder, decoder, 
                             criterion, criterion_gen,
                             device, pointmaps_activation, 
                             symmetrize_batch=False, dtype=torch.float32,
                             not_pred_raymap=False,
                             loss_enc_feat=False,
                             ret=None,
                             distill_criterion=None, 
                             distill_model=None, 
                             is_distill=True,
                             sam_model="SAM2",
                             finetune_encoder=False):

    if raymap_encoder is None:
        pred_raymap = False
    else:
        pred_raymap = not not_pred_raymap
    


    # with torch.cuda.amp.autocast(enabled=bool(use_amp)):
    if isinstance(views[0]['is_raymap'], bool):
        raymap_views = [b for b in views if b['is_raymap']]
        img_views = [b for b in views if not b['is_raymap']]
    else:
        raymap_views = [b for b in views if (b['is_raymap'] == True).all()]
        img_views = [b for b in views if (b['is_raymap'] == False).all()]

    B = img_views[0]['img'].shape[0]
    nraymaps, nimgs = len(raymap_views), len(img_views)
    img_out_0, img_out, raymap_out, x_ray, sam_feats_0, sam_feats_img_and_raymap = inference_occany(img_views, raymap_views, 
                     raymap_encoder, img_encoder, decoder, 
                     pointmaps_activation,
                     device,
                     pred_raymap=pred_raymap,
                     dtype=dtype,
                     sam_model=sam_model,
                     encoder_requires_grad=finetune_encoder)


    with torch.autocast("cuda", dtype=torch.float32):
        # NOTE: This still correct with the loss as in_camera0 is the same for both "views"
        gt_recon = img_views + img_views
        pred_recon = concat_preds(img_out_0, img_out)

        # For visualization 
        combined_gt = img_views
        combined_preds = img_out
        if pred_raymap:
            combined_gt = img_views + raymap_views
            combined_preds = concat_preds(img_out, raymap_out)
            
            gt_gen = raymap_views
            pred_gen = raymap_out
            
        if criterion is not None:
            loss_recon = criterion(gt_recon, pred_recon)
            details = {f"{k}_recon": v for k, v in loss_recon[1].items()}
            total_loss = loss_recon[0]

            if pred_raymap:
                loss_gen = criterion_gen(gt_gen, pred_gen)
                details.update({f"{k}_gen": v for k, v in loss_gen[1].items()})
                total_loss = total_loss + loss_gen[0]
                if loss_enc_feat:
                    img_of_raymaps, true_shape_raymap, _, _ = prepare_imgs_or_raymaps_and_true_shape_mem_batches(raymap_views, device, is_raymap=False)
                    x_img_of_raymaps, pos_img_of_raymaps = inference_encoder(
                        encoder=img_encoder,
                        imgs=img_of_raymaps,
                        true_shape_view=true_shape_raymap.view(B * nraymaps, 2),
                        max_bs=None,
                        requires_grad=False,
                    )
                    # # Block gradients only to x_img_of_raymaps branch
                    loss_enc_feat = F.mse_loss(x_ray, x_img_of_raymaps.detach().clone())
                    details["enc_feat"] = loss_enc_feat
                    total_loss = total_loss + loss_enc_feat
            if distill_criterion is not None:
                distill_imgs = [b['distill_img'] for b in img_views + raymap_views]
                distill_imgs = torch.stack(distill_imgs, dim=1).to(device)

                with torch.no_grad():
                    distill_input = distill_imgs.reshape(
                        B * (nimgs + nraymaps), 3, distill_imgs.shape[-2], distill_imgs.shape[-1]
                    )
                    if isinstance(distill_model, Sam3ProcessorWrapper):
                        # SAM3 returns (feat_s0, feat_s1, feat_s2, pre_neck_feat) as tuple
                        distill_feats = distill_model.forward_distill(distill_input)
                    else:
                        distill_feats = distill_model.forward(distill_input)
             
                # Reshape all distill features: (3 for SAM2, 4 for SAM3 with pre_neck_feat)
                distill_feats = [distill_feat.view(B, nimgs + nraymaps, *distill_feat.shape[1:]).detach() for distill_feat in distill_feats]
                distill_feats_recon = [torch.cat([distill_feat[:, :nimgs], distill_feat[:, :nimgs]], dim=1) for distill_feat in distill_feats]
             
                # Combine sam_feats from both passes (handles 3 or 4 features automatically)
                num_feats = len(sam_feats_0)
                sam_feats_recon = [
                    torch.cat([sam_feats_0[i], sam_feats_img_and_raymap[i][:, :nimgs]], dim=1)
                    for i in range(num_feats)
                ]
                
                if distill_criterion.use_conf:
                    loss_distill_recon = distill_criterion(sam_feats_recon, distill_feats_recon, pred_recon['conf'].detach())
                else:
                    loss_distill_recon = distill_criterion(sam_feats_recon, distill_feats_recon)
                details.update({f"{k}_distill_recon": v for k, v in loss_distill_recon[1].items()})
                total_loss = total_loss + loss_distill_recon[0]
                if pred_raymap:
                    distill_feats_gen = [distill_feat[:, nimgs:] for distill_feat in distill_feats]
                    sam_feats_gen = [sam_feats_img_and_raymap[i][:, nimgs:] for i in range(num_feats)]
                    
                    if distill_criterion.use_conf:
                        loss_distill_gen = distill_criterion(sam_feats_gen, distill_feats_gen, pred_gen['conf'].detach())
                    else:
                        loss_distill_gen = distill_criterion(sam_feats_gen, distill_feats_gen)
                    details.update({f"{k}_distill_gen": v for k, v in loss_distill_gen[1].items()})
                    total_loss = total_loss + loss_distill_gen[0]
            loss = (total_loss, details)
        else:
            loss = None
        
        result = dict(loss=loss, views=views, 
                      raymap_preds=raymap_out, 
                      img_preds_0=img_out_0,
                      img_preds=img_out,
                      gt_img=img_views,
                      gt_raymap=raymap_views,
                      combined_gt=combined_gt,
                      combined_preds=combined_preds)
    return result





def concat_preds(*outs):
    if len(outs) < 2:
        raise ValueError("At least two outputs are required for concatenation")
    
    new_out = {}
    first_out = outs[0]
    
    for k in first_out.keys():
        # Check if key exists in all outputs
        if all(k in out for out in outs):
            # try:
            new_out[k] = torch.concatenate([out[k] for out in outs], dim=1)
            # except:
            #     breakpoint()
    
    return new_out
