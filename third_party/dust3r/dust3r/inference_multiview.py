# --------------------------------------------------------
# utilities needed for the inference
# --------------------------------------------------------
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans, save_semantic
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.viz import SceneViz, auto_cam_size
from dust3r.utils.image import rgb
import os
from occany.occany.datasets.class_mapping import ClassMapping


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor) and value1.ndim == value2.ndim:
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def visualize_semantic(view1, view2, pred1, pred2, save_dir='./tmp'):
    # visualize_type: 'gt' or 'pred'
    pred_semantic1 = pred1['semantic'][0].argmax(dim=-1) + 1
    pred_semantic2 = pred2['semantic'][0].argmax(dim=-1) + 1
    gt_semantic1 = view1['semantic_pseudolabel'][0] + 1
    gt_semantic2 = view2['semantic_pseudolabel'][0] + 1

    view1_name = view1['label'][0]
    view2_name = view2['label'][0]

    class_mapping = ClassMapping()
    save_semantic(pred_semantic1.cpu().numpy(), os.path.join(save_dir, f"semantic_pred1_{view1_name}.png"), class_mapping)
    save_semantic(pred_semantic2.cpu().numpy(), os.path.join(save_dir, f"semantic_pred2_{view2_name}.png"), class_mapping)
    save_semantic(gt_semantic1.cpu().numpy(), os.path.join(save_dir, f"semantic_gt1_{view1_name}.png"), class_mapping)
    save_semantic(gt_semantic2.cpu().numpy(), os.path.join(save_dir, f"semantic_gt2_{view2_name}.png"), class_mapping)





def visualize_results(view1, view2, pred1, pred2, save_dir='./tmp', save_name=None, visualize_type='gt'):
    # visualize_type: 'gt' or 'pred'
    viz = SceneViz()
    views = [view1, view2]
    poses = [views[view_idx]['camera_pose'][0] for view_idx in [0, 1]]
    cam_size = max(auto_cam_size(poses), 0.5)
    if visualize_type == 'pred':
        cam_size *= 0.1
        views[0]['pts3d'] = geotrf(poses[0], pred1['pts3d']) # convert from X_camera1 to X_world
        views[1]['pts3d'] = geotrf(poses[0], pred2['pts3d_in_other_view'])
    for view_idx in [0, 1]:
        pts3d = views[view_idx]['pts3d'][0]
        valid_mask = views[view_idx]['valid_mask'][0]
        colors = rgb(views[view_idx]['img'][0])
        viz.add_pointcloud(pts3d, colors, valid_mask)
        viz.add_camera(pose_c2w=views[view_idx]['camera_pose'][0],
                    focal=views[view_idx]['camera_intrinsics'][0, 0],
                    color=(view_idx * 255, (1 - view_idx) * 255, 0),
                    image=colors,
                    cam_size=cam_size)
    if save_name is None:
        save_name = f'{views[0]["dataset"][0]}_{views[0]["label"][0]}_{views[0]["instance"][0]}_{views[1]["instance"][0]}_{visualize_type}'
    save_path = save_dir+'/'+save_name+'.glb'
    print(f'Saving visualization to {save_path}')
    return viz.save_glb(save_path)



def loss_of_one_batch(
    batch, model, criterion, device, precision, ret=None, profiling=False,
):
    """
    Args:
        batch (list[dict]): a list of views, each view is a dict of tensors, the tensors are batched
    """
    for view in batch:
        for (
            name
        ) in (
            "img pts3d valid_mask camera_pose camera_intrinsics F_matrix corres".split()
        ):  # pseudo_focal
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    views = batch

    autocast_dict = dict(device_type=device.type)
    if precision == "32":
        autocast_dict["enabled"] = False
    elif precision == "16-mixed":
        autocast_dict["dtype"] = torch.float16
    elif precision in ["bf16-mixed", "bf16-mixed-no-grad-scaling"]:
        autocast_dict["dtype"] = torch.bfloat16
    elif precision == torch.bfloat16:
        autocast_dict["dtype"] = torch.bfloat16


    with torch.autocast(**autocast_dict):
        if profiling:
            preds, profiling_info = model(views, profiling=profiling)
        else:
            preds = model(views, profiling=profiling)

        # loss is supposed to be symmetric
        loss = (
            criterion(views, preds) if criterion is not None else None
        )


    result = dict(views=views, preds=preds, loss=loss)
    if profiling:
        result["profiling_info"] = profiling_info

    return result[ret] if ret else result


@torch.no_grad()
def inference(multiple_views_in_one_sample, model,
              device, precision='32', verbose=True, profiling=False):
    assert precision in ['32', '16-mixed', 'bf16-mixed', 'bf16-mixed-no-grad-scaling'], f"Precision {precision} not supported"
    if verbose:
        print(f">> Inference with model on {len(multiple_views_in_one_sample)} images")
    result = []

    # first, check if all images have the same size
    multiple_shapes = not (check_if_same_size(multiple_views_in_one_sample))
    if multiple_shapes:  # force bs=1
        batch_size = 1

    # Get the result from loss_of_one_batch
    res = loss_of_one_batch(
        collate_with_cat([tuple(multiple_views_in_one_sample)]), model, None, device, precision, profiling=profiling
    )


    # Extract profiling_info before to_cpu if it exists
    profiling_info = None
    if profiling and "profiling_info" in res:
        profiling_info = res.pop("profiling_info")

    # Process the result without profiling_info
    result.append(to_cpu(res))
    result = collate_with_cat(result, lists=multiple_shapes)

    # Return the result with profiling_info if requested
    if profiling and profiling_info is not None:
        return result, profiling_info

    return result

def check_if_same_size(imgs):
    shapes = [img["img"].shape[-2:] for img in imgs]
    return all(shape == shapes[0] for shape in shapes)


def get_pred_pts3d(gt, pred, use_pose=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:
        # pts3d from my camera
        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred["pts3d_in_other_view"]  # return!

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    # concat the pointcloud
    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):
        # scaling = (all_pr / all_gt).view(B, -1).mean(dim=1)
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):
        # init scaling with l2 closed form
        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip_(min=1e-8).reciprocal()
            # update the scaling with the new weights
            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)
    # assert scaling.isfinite().all(), bb()
    return scaling
