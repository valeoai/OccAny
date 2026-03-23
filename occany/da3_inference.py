# Copyright (C) 2025-present Naver Corporation. All rights reserved.
import torch
import torch.nn.functional as F
from contextlib import nullcontext
import numpy as np
import itertools
import roma
from occany.model.must3r_blocks.head import ActivationType, apply_activation

from dust3r.post_process import estimate_focal_knowing_depth
from occany.utils.image_util import quaternion_to_matrix, camera_to_pose_encoding
from occany.utils.helpers import get_ray_map_lsvm
from dust3r.utils.geometry import geotrf
from depth_anything_3.utils.geometry import affine_inverse
from occany.utils.helpers import depth2rgb

from torch_scatter import scatter_min






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
        none_mask = [[vl is None for vl in v[i]]
                     for v in value_stacks
                     ]
        # apply "or" on all the different types of values
        none_mask = [any([v[j] for v in none_mask]) for j in range(len(true_shape_stacks[i]))]
        if not any(none_mask) or all(none_mask):
            # there was no None or all were None skip
            continue
        not_none_mask = [not x for x in none_mask]

        def get_filtered_list(lst, local_mask):
            return [v for v, m in zip(lst, local_mask) if m]
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

    

def inference_occany_da3(img_views, model,
                     device,
                     dtype=torch.float32,
                     sam_model="SAM2",
                     pose_from_depth_ray=False,
                     point_from_depth_and_pose=False,
                     **kwargs):
    with torch.autocast("cuda", dtype=dtype):
       
        imgs, true_shape_img, mem_batches, img_timesteps = prepare_imgs_or_raymaps_and_true_shape_mem_batches(img_views, device, is_raymap=False)

        output = model(
            imgs,
            pose_from_depth_ray=pose_from_depth_ray,
            point_from_depth_and_pose=point_from_depth_and_pose,
            **kwargs,
        )

    return output

def inference_occany_da3_gen(
    recon_output,
    img_views,
    gen_views,
    model,
    device,
    dtype=torch.float32,
    projection_features=None,
    pose_from_depth_ray=False,
    point_from_depth_and_pose=False,
    **kwargs,
):
    with torch.autocast("cuda", dtype=dtype):
        return model(
            recon_output=recon_output,
            img_views=img_views,
            gen_views=gen_views,
            projection_features=projection_features,
            pose_from_depth_ray=pose_from_depth_ray,
            point_from_depth_and_pose=point_from_depth_and_pose,
            **kwargs,
        )


def create_gen_conditioning(pts3d, pts_features, focal, 
                            raymap_c2w,
                            return_projected_pts3d=False,
                            gen_views=None, visualize=False,
                            use_raymap_only_conditioning=False,
                            projection_features=None):
    # gen_views = img_views
    # B, n_gen_views, 4, 4
    # n_gen_views = len(gen_views)
    device = pts3d.device
    proj_dtype = pts3d.dtype
    raymap_c2w = raymap_c2w.to(device=device, dtype=proj_dtype)
    focal = focal.to(device=device, dtype=proj_dtype)
    if pts_features is not None:
        pts_features = pts_features.to(device=device, dtype=proj_dtype)

    B, n_gen_views = raymap_c2w.shape[:2]
    H, W = pts3d.shape[2], pts3d.shape[3]
    feature_dim = pts_features.shape[-1]
    
    # If use_raymap_only_conditioning is True, return raymap computed from camera poses
    if use_raymap_only_conditioning:
        # Use focal to compute fxfycxcy
        cx, cy = W / 2.0, H / 2.0
        fxfycxcy = torch.zeros((B, n_gen_views, 4), device=device, dtype=raymap_c2w.dtype)
        fxfycxcy[:, :, 0] = focal.unsqueeze(1)  # fx
        fxfycxcy[:, :, 1] = focal.unsqueeze(1)  # fy
        fxfycxcy[:, :, 2] = cx
        fxfycxcy[:, :, 3] = cy
        
        # get_ray_map_lsvm returns [b, v, 6, h, w] (oxd + ray_d)
        ray_map = get_ray_map_lsvm(raymap_c2w, fxfycxcy, H, W, device=device)
        # Rearrange to [B, n_gen_views, H, W, 6] to match cond_features format
        ray_map = ray_map.permute(0, 1, 3, 4, 2)  # [B, n_gen_views, H, W, 6]
        return ray_map
    # raymap_c2w = torch.stack([v['camera_pose'] for v in gen_views], dim=1).to(device)
    raymap_w2c = affine_inverse(raymap_c2w)
    
    pts3d = pts3d.reshape(B, -1, 3).unsqueeze(1).expand(-1, n_gen_views, -1, -1)
    # rgb_0 = rgb_0.reshape(B, -1, 3).unsqueeze(1).expand(-1, n_gen_views, -1, -1)
    # conf_0 = conf_0.reshape(B, -1).unsqueeze(1).expand(-1, n_gen_views, -1)
    if feature_dim > 0:
        pts_features = pts_features.reshape(B, -1, feature_dim).unsqueeze(1).expand(-1, n_gen_views, -1, -1)
    else:
        pts_features = None
    pts3d_in_raymap_poses = geotrf(raymap_w2c, pts3d)

    # Test with gt camera intrinsics
    # focal =  views[0]['camera_intrinsics'][0, 0, 0]
    # cx = views[0]['camera_intrinsics'][0, 0, 2]
    # cy = views[0]['camera_intrinsics'][0, 1, 2]

    cx = W / 2
    cy = H / 2

    # Use pts3d.dtype for consistency in mixed-precision training
    # Note: 1e-8 epsilon is safe for both fp16 and bf16 (fp16 min: ~6e-8, bf16 min: ~1e-38)
    cam_k_estimated = torch.zeros(B, 3, 3, device=device, dtype=pts3d.dtype)
    cam_k_estimated[:, 0, 0] = focal
    cam_k_estimated[:, 1, 1] = focal
    cam_k_estimated[:, 0, 2] = cx
    cam_k_estimated[:, 1, 2] = cy
    cam_k_estimated[:, 2, 2] = 1
    cam_k_estimated = cam_k_estimated.unsqueeze(1).expand(-1, n_gen_views, -1, -1)

    pts_cam = torch.einsum("brij, brnj -> brni", cam_k_estimated, pts3d_in_raymap_poses)
    pts_2d = pts_cam[..., :2] / (pts_cam[..., 2:3] + 1e-8)
    cond_pointmap = torch.zeros(B, n_gen_views, H, W, 3, device=device, dtype=pts3d.dtype)
    cond_pts_features = torch.zeros(B, n_gen_views, H, W, feature_dim, device=device, dtype=pts3d.dtype)
    
    # Set to a default 0 to avoid error
    # cond_features = torch.cat([cond_pointmap, cond_pts_features], dim=-1)
    
    

    # Convert to integer coordinates and create validity mask
    pts_2d_int = pts_2d.round().long()
    valid_mask = (
        (pts_2d_int[..., 0] >= 0) & (pts_2d_int[..., 0] < W) &
        (pts_2d_int[..., 1] >= 0) & (pts_2d_int[..., 1] < H) &
        (pts_cam[..., 2] > 0)
    )

    # Apply mask to filter valid points only
    B, n_gen_views, N = pts_2d_int.shape[:3]

    # Process each batch and raymap separately to avoid OOM
    for b in range(B):
        for r in range(n_gen_views):
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
                
                # Reshape and assign to output
                cond_pointmap_flat = cond_pointmap[b, r].view(H * W, 3)
                cond_pointmap_flat[output_mask] = closest_points
                cond_pointmap[b, r] = cond_pointmap_flat.view(H, W, 3)

                if feature_dim > 0:
                    closest_features = features_valid[argmin_indices[output_mask]]
                    cond_pts_features_flat = cond_pts_features[b, r].view(H * W, feature_dim)
                    cond_pts_features_flat[output_mask] = closest_features
                    cond_pts_features[b, r] = cond_pts_features_flat.view(H, W, feature_dim)
    
    # Replace the first 3 channels of cond_pts_features (original pts3d_local) with cond_pointmap
    # (pts3d_local in gen view's camera frame) to match feature order with recon_cond_features.
    # This ensures both recon and gen views have: [pts3d_local, pts3d, rgb, conf, sam3, ...]
    # where pts3d_local is in the respective view's camera frame.
    if feature_dim >= 3:
        # cond_pts_features[..., :3] was the original pts3d_local projected to gen view pixels
        # Replace with cond_pointmap which is pts3d_local in gen view's camera frame
        cond_pts_features[..., :3] = cond_pointmap
    cond_features = cond_pts_features

    # If 'raymap' is in projection_features, compute raymap and insert it after pts3d_local
    if projection_features is not None and 'raymap' in projection_features:
        # Use focal to compute fxfycxcy
        cx, cy = W / 2.0, H / 2.0
        fxfycxcy = torch.zeros((B, n_gen_views, 4), device=device, dtype=raymap_c2w.dtype)
        fxfycxcy[:, :, 0] = focal.unsqueeze(1)  # fx
        fxfycxcy[:, :, 1] = focal.unsqueeze(1)  # fy
        fxfycxcy[:, :, 2] = cx
        fxfycxcy[:, :, 3] = cy
        
        # get_ray_map_lsvm returns [b, v, 6, h, w] (oxd + ray_d)
        ray_map = get_ray_map_lsvm(raymap_c2w, fxfycxcy, H, W, device=device)
        # Rearrange to [B, n_gen_views, H, W, 6] to match cond_features format
        ray_map = ray_map.permute(0, 1, 3, 4, 2)  # [B, n_gen_views, H, W, 6]
        
        # Concatenate raymap with cond_features after pts3d_local
        # cond_features is [B, n_gen_views, H, W, feature_dim] where first 3 is pts3d_local
        # Insert raymap after first 3 channels: [pts3d_local(3), raymap(6), rest_of_features]
        cond_features = torch.cat([cond_features[..., :3], ray_map, cond_features[..., 3:]], dim=-1)

    if return_projected_pts3d:
        return cond_pointmap

    # visualize = True
    # Visualization code
    if visualize and gen_views is not None:
        import os
        os.makedirs('demo_data', exist_ok=True)
        for batch_id in range(B):
            pred_depth = cond_pointmap[batch_id][..., 2]
            pred_col = torch.cat([pred_depth[i] for i in range(n_gen_views)], dim=0)  # (N*H, W)
            
            # Visualize input images if available
            if 'img' in gen_views[0]:
                img = torch.cat([v['img'][batch_id] for v in gen_views], dim=1)
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



                    

def loss_of_one_batch_occany_da3(views, model, 
                             device, 
                             dtype=torch.float32,
                             distill_criterion=None, 
                             distill_model=None, 
                             is_distill=False,
                             use_ray_pose=False,
                             sam_model="SAM2",
                             pointmap_criterion=None,
                             depth_criterion=None,
                             raymap_criterion=None,
                             lambda_depth=1.0,
                             lambda_raymap=1.0,
                             lambda_pointmap=1.0,
                             pose_from_depth_ray=False,
                             scale_inv_depth_criterion=None,
                             lambda_scale_inv_depth=1.0):
    """
    Compute loss for one batch with DA3 model.
    
    Args:
        views: List of view dictionaries.
        model: DA3 model.
        criterion: Pointmap loss criterion.
        device: Device to use.
        dtype: Data type for autocast.
        distill_criterion: Distillation criterion (optional).
        distill_model: Distillation model (optional).
        is_distill: Whether to use distillation.
        use_ray_pose: If True, model computes and returns pointmap (for testing/evaluation).
        sam_model: SAM model type.
        depth_criterion: DepthLosses criterion for depth supervision (optional).
        raymap_criterion: RaymapLoss criterion for raymap supervision (optional).
        lambda_depth: Weight for depth loss.
        lambda_raymap: Weight for raymap loss.
        lambda_pointmap: Weight for pointmap loss.
        pose_from_depth_ray: Whether to estimate pose from depth and raymap.
        scale_inv_depth_criterion: Scale-invariant depth loss criterion (optional).
        lambda_scale_inv_depth: Weight for scale-invariant depth loss.
    """

    # with torch.cuda.amp.autocast(enabled=bool(use_amp)):
    if isinstance(views[0]['is_raymap'], bool):
        gen_views = [b for b in views if b['is_raymap']]
        img_views = [b for b in views if not b['is_raymap']]
    else:
        gen_views = [b for b in views if (b['is_raymap'] == True).all()]
        img_views = [b for b in views if (b['is_raymap'] == False).all()]

    B = img_views[0]['img'].shape[0]
    n_gen_views, nimgs = len(gen_views), len(img_views)
    
    # use_ray_pose determines whether to compute pointmap (for testing) or keep raw output (for training)
    # For training, we need raw depth/raymap for loss computation
    # For testing, we compute pointmap for evaluation metrics
    
    output = inference_occany_da3(
                    img_views, model,
                     device,
                     dtype=dtype,
                     sam_model=sam_model,
                     pose_from_depth_ray=pose_from_depth_ray)
    
    with torch.autocast("cuda", dtype=torch.float32):
        depth = output.get('depth')
        depth_conf = output.get('depth_conf')
        ray = output.get('ray')  # (B, T, H, W, 6) when return_depth_and_raymap=True
        ray_conf = output.get('ray_conf')  # (B, T, H, W) when return_depth_and_raymap=True
        
        if depth is not None:
            depth = depth.to(device)
        if depth_conf is not None:
            depth_conf = depth_conf.to(device)
        if ray is not None:
            ray = ray.to(device)
        if ray_conf is not None:
            ray_conf = ray_conf.to(device)

        valid_mask = None
        if 'valid_mask' in img_views[0]:
            valid_mask = torch.stack([b['valid_mask'] for b in img_views], dim=1).to(device)

        details = {}
        total_loss = 0.0
        
        if pointmap_criterion is not None and lambda_pointmap > 0:
            pointmap_gt = torch.stack([b['pts3d'] for b in img_views], dim=1).to(device)
            pointmap = output.get('pointmap', None)
            assert pointmap is not None, "DA3 output does not contain 'pointmap' or 'point_map'"
            pointmap = pointmap.to(device)
            # Prepare confidence for pointmap loss (use depth_conf if available)
            # depth_conf shape is (B, T, H, W), pointmap shape is (B, T, H, W, 3)
            # We pass depth_conf directly - PointmapLoss handles shape matching
            pointmap_conf = depth_conf  # Can be None if not available
            # Pointmap loss (confidence-aware when lambda_c > 0)
            
            loss_pointmap, loss_details = pointmap_criterion(
                pointmap.float(), pointmap_gt.float(), valid_mask, confidence=pointmap_conf
            )
            details.update(loss_details)
            total_loss = total_loss + lambda_pointmap * loss_pointmap
        
        # Depth loss
        if depth_criterion is not None and depth is not None and lambda_depth > 0:
            # Get GT depth from views
            if 'depthmap' in img_views[0]:
                gt_depth = torch.stack([b['depthmap'] for b in img_views], dim=1).to(device)
                # Ensure shapes match: depth is (B, T, H, W), need (B, T, 1, H, W) for DepthLosses
                if gt_depth.ndim == 4:  # (B, T, H, W)
                    gt_depth = gt_depth.unsqueeze(2)  # (B, T, 1, H, W)
                if depth.ndim == 4:  # (B, T, H, W)
                    depth_for_loss = depth.unsqueeze(2)  # (B, T, 1, H, W)
                else:
                    depth_for_loss = depth
                if depth_conf.ndim == 4:  # (B, T, H, W)
                    depth_conf_for_loss = depth_conf.unsqueeze(2)  # (B, T, 1, H, W)
                else:
                    depth_conf_for_loss = depth_conf
                
                # Reshape for DepthLosses which expects (B*T, 1, H, W)
                B_T = depth_for_loss.shape[0] * depth_for_loss.shape[1]
                H, W = depth_for_loss.shape[-2:]
                depth_for_loss = depth_for_loss.reshape(B_T, 1, H, W)
                gt_depth = gt_depth.reshape(B_T, 1, H, W)
                depth_conf_for_loss = depth_conf_for_loss.reshape(B_T, 1, H, W)
                
                depth_mask = None
                if valid_mask is not None:
                    depth_mask = valid_mask.reshape(B_T, 1, H, W)
                
                depth_loss, depth_loss_details = depth_criterion(
                    depth_for_loss.float(), gt_depth.float(), depth_conf_for_loss.float(), depth_mask
                )
                total_loss = total_loss + lambda_depth * depth_loss
                details.update({f"depth_{k}": v for k, v in depth_loss_details.items()})
        
        # Raymap loss
        if raymap_criterion is not None and lambda_raymap > 0:
            # Get GT camera poses (c2w) and intrinsics from views
            assert 'camera_pose' in img_views[0], "camera_pose must be present in views for raymap loss"
            assert 'camera_intrinsics' in img_views[0], "camera_intrinsics must be present in views for raymap loss"
            gt_c2w = torch.stack([b['camera_pose'] for b in img_views], dim=1).to(device)  # (B, T, 4, 4) camera-to-world
            gt_intrinsics = torch.stack([b['camera_intrinsics'] for b in img_views], dim=1).to(device)  # (B, T, 3, 3)
            
            # Use pre-computed gt_raymap from dataset if available
            gt_raymap = None
            if 'gt_raymap' in img_views[0]:
                gt_raymap = torch.stack([b['gt_raymap'] for b in img_views], dim=1).to(device)  # (B, T, H, W, 6)
            
            raymap_loss, raymap_loss_details = raymap_criterion(
                ray.float(), ray_conf.float(), gt_c2w.float(), gt_intrinsics.float(), gt_raymap=gt_raymap
            )
            total_loss = total_loss + lambda_raymap * raymap_loss
            details.update(raymap_loss_details)
        
        # SAM3 distillation loss
        if distill_criterion is not None and distill_model is not None:
            # Use pre-computed sam_feats from output (computed inside model forward for DDP compatibility)
            sam_feats = output.get('sam_feats')
            
            if sam_feats is not None:
                # Get teacher features from SAM3
                # SAM3 expects images normalized differently (not ImageNet normalized)
                # First undo ImageNet normalization, then apply SAM3 preprocessing
                distill_imgs = torch.stack([b['distill_img'] for b in img_views], dim=1).to(device)
                B_distill, T_distill = distill_imgs.shape[:2]
                
                with torch.no_grad():
                    # distill_model.forward_distill returns (feat_s0, feat_s1, feat_s2, pre_neck_feat)
                    distill_feats = distill_model.forward_distill(
                        distill_imgs.reshape(B_distill * T_distill, 3, *distill_imgs.shape[-2:])
                    )
                # Reshape back to (B, T, ...)
                distill_feats = [f.view(B_distill, T_distill, *f.shape[1:]).detach() for f in distill_feats]

                # Compute distillation loss
                if distill_criterion.use_conf:
                    loss_distill, distill_details = distill_criterion(
                        sam_feats, distill_feats, depth_conf.detach()
                    )
                else:
                    loss_distill, distill_details = distill_criterion(sam_feats, distill_feats)
                
                total_loss = total_loss + loss_distill
                details.update({f"distill_{k}": v for k, v in distill_details.items()})
        
        # Scale-invariant depth loss (when aux_outputs is available)
        if scale_inv_depth_criterion is not None and lambda_scale_inv_depth > 0:
            aux_outputs = output.get('aux_outputs')
            if aux_outputs is not None:
                aux_depth = aux_outputs.get('depth')
                if aux_depth is not None and depth is not None:
                    # Get GT depth from views
                    if 'depthmap' in img_views[0]:
                        gt_depth_for_scale = torch.stack([b['depthmap'] for b in img_views], dim=1).to(device)
                        
                        scale_inv_loss, scale_inv_details = scale_inv_depth_criterion(
                            depth.float(),  # trainable depth
                            aux_depth.float(),  # frozen aux depth
                            gt_depth_for_scale.float(),  # GT depth for scale computation
                            valid_mask,  # valid mask
                            depth_conf.float() if depth_conf is not None else None,
                        )
                        total_loss = total_loss + lambda_scale_inv_depth * scale_inv_loss
                        details.update(scale_inv_details)
        
        loss = (total_loss, details)
        
        # Build combined_preds for visualization (matching format expected by test_one_epoch)
        # Get RGB from output if available, otherwise use GT images
        rgb = output.get('rgb')  # (B, T, H, W, 3) if available
        if rgb is None:
            # Use GT images as placeholder for visualization
            rgb = torch.stack([b['img'] for b in img_views], dim=1).to(device)  # (B, T, C, H, W)
            rgb = rgb.permute(0, 1, 3, 4, 2)  # (B, T, H, W, C)
        
        combined_preds = {
            'rgb': rgb,  # (B, T, H, W, 3)
            'depth': depth,  # (B, T, H, W) - predicted depth values
        }
        
        # combined_gt is the list of views for visualization
        combined_gt = img_views
        
        # # Debug: Check if view 0's pts3d is perpendicular to Z
        # v0_pts3d = img_views[0]['pts3d'][2]  # batch_idx=2
        # # Print c2w for view 0
        # print(f"View 0 c2w:\n{img_views[0]['camera_pose'][2]}")
        # # Optionally save ground-truth data for debugging/visualization
        # save_gt_render_data(img_views, "/scratch/project/eu-25-92/ssc_output/da3_gt/00000", batch_idx=2, imagenet_normalize=True)
        
        result = dict(
            loss=loss,
            combined_preds=combined_preds,
            combined_gt=combined_gt,
            img_preds=combined_preds,
            gt_img=img_views,
        )
        
    return result

def concat_preds(*outs):
    if len(outs) < 2:
        raise ValueError("At least two outputs are required for concatenation")
    
    new_out = {}
    first_out = outs[0]
    
    for k in first_out.keys():
        # Check if key exists in all outputs
        if all(k in out for out in outs):
            values = [out[k] for out in outs]
            
            # Check if all values are None
            if all(v is None for v in values):
                print(f"[WARNING] All values {k} are None")
                new_out[k] = None
                continue
            
            # Check if any value is None (but not all)
            if any(v is None for v in values):
                print(f"[WARNING] Some values {k} are None, keep the first non-None value")
                new_out[k] = next((v for v in values if v is not None), None)
                continue
            
            # All values are not None - check the type
            first_value = values[0]
            
            # Handle tuples (e.g., sam_feats which is a tuple of tensors)
            if isinstance(first_value, tuple):
                # Concatenate each element in the tuple separately
                concatenated_tuple = []
                for i in range(len(first_value)):
                    tuple_elements = [v[i] for v in values]
                    if all(isinstance(elem, torch.Tensor) for elem in tuple_elements):
                        concatenated_tuple.append(torch.concatenate(tuple_elements, dim=1))
                    else:
                        print(f"[WARNING] Not all tuple elements are tensors, keep the first one {k}")
                        concatenated_tuple.append(tuple_elements[0])
                new_out[k] = tuple(concatenated_tuple)
            # Handle tensors
            elif isinstance(first_value, torch.Tensor):
                new_out[k] = torch.concatenate(values, dim=1)
            else:
                # For other types, keep the first value
                new_out[k] = first_value
    
    return new_out
    

def loss_of_one_batch_occany_da3_gen(
    views, 
    model,
    device, 
    model_recon=None,  # Frozen model for reconstruction (if None, use model)
    dtype=torch.bfloat16,
    distill_criterion=None, 
    distill_model=None, 
    is_distill=False,
    sam_model="SAM2",
    pointmap_criterion=None,
    depth_criterion=None,
    raymap_criterion=None,
    lambda_depth=1.0,
    lambda_raymap=1.0,
    lambda_pointmap=1.0,
    pose_from_depth_ray=False,
    projection_features='pts3d_local,pts3d,rgb,conf',
    lambda_feat_matching=1.0,
):
    """
    Compute loss for one batch with DA3 model including gen views.
    
    This function:
    1. Runs frozen reconstruction on all views (img + gen) and keeps recon-view outputs for conditioning
    2. Projects features to gen views (gen_views)
    3. Encodes projected features with RaymapEncoderDA3
    4. Passes encoded tokens through trainable model (model_gen) DinoV2 backbone + DualDPT head
    5. Computes losses on gen views, including optional feature matching
    
    Args:
        views: List of view dictionaries (mix of img and raymap views).
        model: DA3 model for generation (trainable, DA3Wrapper).
        model_recon: DA3 model for reconstruction (frozen). If None, uses model.
        device: Device to use.
        dtype: Data type for autocast.
        distill_criterion, distill_model, is_distill: Distillation settings.
        sam_model: SAM model type ("SAM2" or "SAM3").
        pointmap_criterion, depth_criterion, raymap_criterion: Loss criteria.
        criterion_gen: Loss criterion for gen views.
        lambda_depth, lambda_raymap, lambda_pointmap: Loss weights.
        lambda_feat_matching: Weight for encoder feature matching loss.
        pose_from_depth_ray: Whether to estimate pose from depth and raymap.
        projection_features: Comma-separated list of features to project.
    """
    # Use model_recon for reconstruction if provided, otherwise fallback to model
    recon_model = model_recon if model_recon is not None else model
    
    # Separate image views and raymap views
    if isinstance(views[0]['is_raymap'], bool):
        gen_views = [b for b in views if b['is_raymap']]
        img_views = [b for b in views if not b['is_raymap']]
    else:
        gen_views = [b for b in views if (b['is_raymap'] == True).all()]
        img_views = [b for b in views if (b['is_raymap'] == False).all()]

    B = img_views[0]['img'].shape[0]
    n_gen_views, nimgs = len(gen_views), len(img_views)
 

    # === Step 1: Reconstruction on all views using FROZEN model_recon ===
    all_recon_views = img_views + gen_views
    with torch.no_grad():
        all_recon_output = inference_occany_da3(
            all_recon_views, recon_model,  # Use frozen recon_model
            device,
            dtype=dtype,
            sam_model=sam_model,
            pose_from_depth_ray=True,
        )

    # Split the output: recon views are first nimgs, gen views are the rest
    recon_output = {
        k: v[:, :nimgs] if isinstance(v, torch.Tensor) else v
        for k, v in all_recon_output.items()
    }
    if 'sam_feats' in all_recon_output and all_recon_output['sam_feats'] is not None:
        recon_output['sam_feats'] = tuple(f[:, :nimgs] for f in all_recon_output['sam_feats'])

    # Extract target features for gen views from frozen recon model
    target_aux_feats = all_recon_output.get('aux_feats')
    if target_aux_feats is not None:
        target_feats = [
            f[0][:, nimgs:].detach() if isinstance(f, (list, tuple)) else f[:, nimgs:].detach()
            for f in target_aux_feats
        ]
    else:
        target_feats = None

    # Extract reconstruction outputs for visualization/loss bookkeeping
    recon_depth = recon_output.get('depth')
    recon_pointmap = recon_output.get('pointmap')

    gen_output = inference_occany_da3_gen(
        recon_output=recon_output,
        img_views=img_views,
        gen_views=gen_views,
        model=model,
        device=device,
        dtype=dtype,
        projection_features=projection_features,
        pose_from_depth_ray=pose_from_depth_ray,
    )

    # Extract predicted features for gen views from trainable model
    pred_aux_feats = gen_output.get('aux_feats') if gen_output is not None else None
    if pred_aux_feats is not None:
        pred_feats = []
        for f in pred_aux_feats:
            feat = f[0] if isinstance(f, (list, tuple)) else f
            if feat.shape[1] > n_gen_views:
                feat = feat[:, -n_gen_views:]
            pred_feats.append(feat)
    else:
        pred_feats = None

    # Free memory from reconstruction output after generation is computed
    del all_recon_output
        
    
    # === Step 4: Compute losses ===
    with torch.autocast("cuda", dtype=torch.float32):
        details = {}
        total_loss = 0.0
        
        # Gen view losses
        if gen_output is not None:
            gen_pointmap = gen_output.get('pointmap')
            gen_depth = gen_output.get('depth')
            gen_depth_conf = gen_output.get('depth_conf')
            gen_ray = gen_output.get('ray')
            gen_ray_conf = gen_output.get('ray_conf')
            
            # Get GT data for gen views
            gen_pointmap_gt = torch.stack([b['pts3d'] for b in gen_views], dim=1).to(device)
            
            # Valid mask for gen views
            gen_valid_mask = None
            if 'valid_mask' in gen_views[0]:
                gen_valid_mask = torch.stack([b['valid_mask'] for b in gen_views], dim=1).to(device)
            
            # Pointmap loss for gen views
            if pointmap_criterion is not None and lambda_pointmap > 0 and gen_pointmap is not None:
                gen_loss_pointmap, gen_loss_pointmap_details = pointmap_criterion(
                    gen_pointmap.float(), gen_pointmap_gt.float(), gen_valid_mask, confidence=gen_depth_conf
                )
                total_loss = total_loss + lambda_pointmap * gen_loss_pointmap
                details.update({f"{k}_gen": v for k, v in gen_loss_pointmap_details.items()})
            
            # Depth loss for gen views
            if depth_criterion is not None and lambda_depth > 0 and gen_depth is not None:
                assert 'depthmap' in gen_views[0], "GT Depthmap not found in gen views"
                gen_depth_gt = torch.stack([b['depthmap'] for b in gen_views], dim=1).to(device)
                # Ensure shapes match
                if gen_depth_gt.ndim == 4:  # (B, T, H, W)
                    gen_depth_gt = gen_depth_gt.unsqueeze(2)  # (B, T, 1, H, W)
                if gen_depth.ndim == 4:
                    gen_depth_for_loss = gen_depth.unsqueeze(2)
                else:
                    gen_depth_for_loss = gen_depth
                if gen_depth_conf is not None and gen_depth_conf.ndim == 4:
                    gen_depth_conf_for_loss = gen_depth_conf.unsqueeze(2)
                else:
                    gen_depth_conf_for_loss = gen_depth_conf
                
                # Reshape for DepthLosses: (B*T, 1, H, W)
                B_T = gen_depth_for_loss.shape[0] * gen_depth_for_loss.shape[1]
                H, W = gen_depth_for_loss.shape[-2:]
                gen_depth_for_loss = gen_depth_for_loss.reshape(B_T, 1, H, W)
                gen_depth_gt = gen_depth_gt.reshape(B_T, 1, H, W)
                if gen_depth_conf_for_loss is not None:
                    gen_depth_conf_for_loss = gen_depth_conf_for_loss.reshape(B_T, 1, H, W)
                
                gen_depth_mask = None
                if gen_valid_mask is not None:
                    gen_depth_mask = gen_valid_mask.reshape(B_T, 1, H, W)
                
                gen_depth_loss, gen_depth_loss_details = depth_criterion(
                    gen_depth_for_loss.float(), gen_depth_gt.float(), 
                    gen_depth_conf_for_loss.float() if gen_depth_conf_for_loss is not None else None, 
                    gen_depth_mask
                )
                total_loss = total_loss + lambda_depth * gen_depth_loss
                details.update({f"depth_{k}_gen": v for k, v in gen_depth_loss_details.items()})
        
            # Raymap loss for gen views
            if raymap_criterion is not None and lambda_raymap > 0 and gen_ray is not None:
                assert 'camera_pose' in gen_views[0], "camera_pose must be present in gen_views for raymap loss"
                assert 'camera_intrinsics' in gen_views[0], "camera_intrinsics must be present in gen_views for raymap loss"
                gen_c2w = torch.stack([b['camera_pose'] for b in gen_views], dim=1).to(device)  # (B, T, 4, 4)
                gen_intrinsics = torch.stack([b['camera_intrinsics'] for b in gen_views], dim=1).to(device)  # (B, T, 3, 3)
                
                # Use pre-computed gt_raymap if available
                gen_gt_raymap = None
                if 'gt_raymap' in gen_views[0]:
                    gen_gt_raymap = torch.stack([b['gt_raymap'] for b in gen_views], dim=1).to(device)
                
                gen_raymap_loss, gen_raymap_loss_details = raymap_criterion(
                    gen_ray.float(), gen_ray_conf.float(), gen_c2w.float(), gen_intrinsics.float(), gt_raymap=gen_gt_raymap
                )
                total_loss = total_loss + lambda_raymap * gen_raymap_loss
                details.update({f"{k}_gen": v for k, v in gen_raymap_loss_details.items()})
        
            # SAM3 distillation loss for gen views
            if distill_criterion is not None and distill_model is not None:
                sam_feats = gen_output.get('sam_feats')
                
                if sam_feats is not None:
                    # Get teacher features from SAM3 for gen views
                    distill_imgs = torch.stack([b['distill_img'] for b in gen_views], dim=1).to(device)
                    B_distill, T_distill = distill_imgs.shape[:2]
                    
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=dtype):
                            distill_feats = distill_model.forward_distill(
                                distill_imgs.reshape(B_distill * T_distill, 3, *distill_imgs.shape[-2:])
                            )
                    distill_feats = [f.view(B_distill, T_distill, *f.shape[1:]).detach() for f in distill_feats]
                    
                    # Compute distillation loss
                    if distill_criterion.use_conf:
                        loss_distill, distill_details = distill_criterion(
                            sam_feats, distill_feats, gen_depth_conf.detach()
                        )
                    else:
                        loss_distill, distill_details = distill_criterion(sam_feats, distill_feats)
                    
                    total_loss = total_loss + loss_distill
                    details.update({f"distill_{k}_gen": v for k, v in distill_details.items()})

        # Feature matching loss: match only out_layers 11.
        if lambda_feat_matching > 0 and target_feats is not None and pred_feats is not None:
            desired_out_layers = [11]
            feat_indices = [0]

            backbone_out_layers = None
            try:
                backbone_out_layers = list(recon_model.model.backbone.out_layers)
            except Exception:
                backbone_out_layers = None

            if backbone_out_layers is not None:
                missing = [l for l in desired_out_layers if l not in backbone_out_layers]
                if len(missing) > 0:
                    raise ValueError(
                        f"Requested feature matching for out_layers={desired_out_layers}, but backbone out_layers={backbone_out_layers} is missing {missing}."
                    )
                feat_indices = [backbone_out_layers.index(l) for l in desired_out_layers]
            
            max_idx = max(feat_indices)
            if max_idx >= len(pred_feats) or max_idx >= len(target_feats):
                raise ValueError(
                    f"Feature matching indices {feat_indices} out of range for pred_feats={len(pred_feats)} / target_feats={len(target_feats)}."
                )

            feat_loss = 0.0
            for idx in feat_indices:
                feat_loss += F.l1_loss(pred_feats[idx].float(), target_feats[idx].float())
            feat_loss /= float(len(feat_indices))

            total_loss = total_loss + lambda_feat_matching * feat_loss
            details['loss_feat_matching'] = feat_loss
        
        loss = (total_loss, details)
        
        # Build combined_preds for visualization
        recon_rgb = torch.stack([b['img'] for b in img_views], dim=1).to(device).permute(0, 1, 3, 4, 2)
        # recon_depth = recon_output.get('depth')
        
        if gen_output is not None:
            gen_rgb = torch.stack([b['img'] for b in gen_views], dim=1).to(device).permute(0, 1, 3, 4, 2)
            gen_depth = gen_output.get('depth')
            
            combined_preds = {
                'rgb': torch.cat([recon_rgb, gen_rgb], dim=1),
                'depth': torch.cat([recon_depth, gen_depth], dim=1),
            }
            combined_gt = img_views + gen_views
        else:
            combined_preds = {
                'rgb': recon_rgb,
                'depth': recon_depth,
            }
            combined_gt = img_views
        
        result = dict(
            loss=loss,
            combined_preds=combined_preds,
            combined_gt=combined_gt,
            raymap_preds=gen_output,
            gt_raymap=gen_views,
            img_preds={'pointmap': recon_pointmap, 'depth': recon_depth},
            gt_img=img_views,
        )
        
    return result
