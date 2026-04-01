from scipy.spatial import KDTree
from typing import Dict as TDict, Optional, Union
import numpy as np
import torch

def nn_correspondance(verts1: np.ndarray, verts2: np.ndarray) -> np.ndarray:
    """
    Compute nearest neighbor distances from verts2 to verts1 using KDTree.

    Args:
        verts1: Reference point cloud [N, 3]
        verts2: Query point cloud [M, 3]

    Returns:
        Distance array [M,] - distance from each point in verts2 to nearest in verts1
    """
    if len(verts1) == 0 or len(verts2) == 0:
        return np.array([])

    kdtree = KDTree(verts1)
    distances, _ = kdtree.query(verts2)
    return distances.reshape(-1)


def evaluate_3d_reconstruction(
    verts_pred: np.ndarray,
    verts_trgt: np.ndarray,
    threshold: float = 0.05,
) -> TDict[str, float]:
    """
    Evaluate 3D reconstruction quality using standard metrics.

    This function computes:
    - Accuracy: Mean distance from predicted points to GT surface
    - Completeness: Mean distance from GT points to predicted surface
    - Overall: Average of accuracy and completeness
    - Precision: Fraction of predicted points within threshold of GT
    - Recall: Fraction of GT points within threshold of prediction
    - F-score: Harmonic mean of precision and recall

    Args:
        verts_pred: Predicted point cloud (numpy array)
        verts_trgt: Ground truth point cloud (numpy array)
        threshold: Distance threshold for precision/recall (meters)

    Returns:
        Dict with metrics: acc, comp, overall, precision, recall, fscore
    """
    
    # Handle empty point clouds
    if len(verts_pred) == 0 or len(verts_trgt) == 0:
        return {
            "acc": float("inf"),
            "comp": float("inf"),
            "overall": float("inf"),
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
        }

    # Compute distances
    dist_pred_to_gt = nn_correspondance(verts_trgt, verts_pred)  # Accuracy
    dist_gt_to_pred = nn_correspondance(verts_pred, verts_trgt)  # Completeness

    # Compute metrics
    accuracy = float(np.mean(dist_pred_to_gt))
    completeness = float(np.mean(dist_gt_to_pred))
    overall = (accuracy + completeness) / 2

    precision = float(np.mean((dist_pred_to_gt < threshold).astype(float)))
    recall = float(np.mean((dist_gt_to_pred < threshold).astype(float)))

    if precision + recall > 0:
        fscore = 2 * precision * recall / (precision + recall)
    else:
        fscore = 0.0

    return {
        "acc": accuracy,
        "comp": completeness,
        "overall": overall,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
    }



def evaluate_depth_metrics(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    max_depth: float = 50.0,
) -> TDict[str, Optional[float]]:
    """
    Compute standard monocular depth evaluation metrics (no alignment).

    Metrics follow the conventions from Eigen et al. and dust3r/depth_eval.py:
    Abs Rel, Sq Rel, RMSE, Log RMSE, δ < 1.25, δ < 1.25², δ < 1.25³.

    Args:
        pred_depth: Predicted depth, any shape (will be flattened).
        gt_depth:   Ground-truth depth, same shape as *pred_depth*.
        max_depth:  Upper bound on valid GT depth (metres).

    Returns:
        Dict with keys abs_rel, sq_rel, rmse, log_rmse, delta_1, delta_2,
        delta_3.  All values are ``None`` when no valid pixel exists.
    """
    _none = {
        "abs_rel": None,
        "sq_rel": None,
        "rmse": None,
        "log_rmse": None,
        "delta_1": None,
        "delta_2": None,
        "delta_3": None,
    }

    pred = pred_depth.reshape(-1).float()
    gt = gt_depth.reshape(-1).float()

    valid = (
        (gt > 0)
        & (gt < max_depth)
        & (pred > 0)
        & torch.isfinite(gt)
        & torch.isfinite(pred)
    )
    if valid.sum() == 0:
        return _none

    pred = pred[valid]
    gt = gt[valid]

    abs_rel = torch.mean(torch.abs(pred - gt) / gt).item()
    sq_rel = torch.mean(((pred - gt) ** 2) / gt).item()
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()

    pred_log = torch.clamp(pred, min=1e-5)
    log_rmse = torch.sqrt(torch.mean((torch.log(pred_log) - torch.log(gt)) ** 2)).item()

    ratio = torch.maximum(pred / gt, gt / pred)
    delta_1 = torch.mean((ratio < 1.25).float()).item()
    delta_2 = torch.mean((ratio < 1.25 ** 2).float()).item()
    delta_3 = torch.mean((ratio < 1.25 ** 3).float()).item()

    return {
        "abs_rel": abs_rel,
        "sq_rel": sq_rel,
        "rmse": rmse,
        "log_rmse": log_rmse,
        "delta_1": delta_1,
        "delta_2": delta_2,
        "delta_3": delta_3,
    }


def evaluate_chamfer_distance(
    pred_pc: np.ndarray,
    gt_pc: np.ndarray,
    device: str = "cuda",
) -> TDict[str, Optional[float]]:
    """
    Compute Chamfer distance between predicted and GT point clouds.

    Uses the CUDA-accelerated ``ChamferDistance`` module from
    ``third_party/pyTorchChamferDistance``.

    Args:
        pred_pc: Predicted point cloud ``(N, 3)`` numpy array.
        gt_pc:   Ground-truth point cloud ``(M, 3)`` numpy array.
        device:  Torch device string (default ``"cuda"``).

    Returns:
        Dict with chamfer_pred2gt (mean L2 pred→gt), chamfer_gt2pred
        (mean L2 gt→pred), and chamfer_dist (sum of both directions).
        All ``None`` when either cloud is empty.
    """
    _none = {"chamfer_pred2gt": None, "chamfer_gt2pred": None, "chamfer_dist": None}

    if pred_pc.shape[0] == 0 or gt_pc.shape[0] == 0:
        return _none

    from chamfer_distance import ChamferDistance

    chamfer_fn = ChamferDistance()

    pred_t = torch.from_numpy(pred_pc).float().unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt_pc).float().unsqueeze(0).to(device)

    with torch.no_grad():
        sq_pred2gt, sq_gt2pred = chamfer_fn(pred_t, gt_t)

    mean_pred2gt = sq_pred2gt.sqrt().mean().item()
    mean_gt2pred = sq_gt2pred.sqrt().mean().item()
    chamfer_dist = mean_pred2gt + mean_gt2pred

    return {
        "chamfer_pred2gt": mean_pred2gt,
        "chamfer_gt2pred": mean_gt2pred,
        "chamfer_dist": chamfer_dist,
    }


def evaluate_all_metrics(
    pred_pc: np.ndarray,
    gt_pc: np.ndarray,
    threshold: float = 0.05,
    device: str = "cuda",
) -> TDict[str, TDict[str, Optional[float]]]:
    """Compute 3D reconstruction + Chamfer metrics in a single GPU pass.

    The CUDA Chamfer kernel returns **squared** L2 NN distances for both
    directions (pred→gt and gt→pred). We take ``sqrt`` to obtain mean L2
    distances, matching the standard Chamfer definition used by the CPU
    implementation and reusing the same distances for accuracy /
    completeness / precision / recall / F-score.

    Returns a dict with keys ``"recon"`` and ``"chamfer"``, each containing
    the same metric dicts as ``evaluate_3d_reconstruction`` and
    ``evaluate_chamfer_distance``.
    """
    _empty_recon = {
        "acc": float("inf"), "comp": float("inf"), "overall": float("inf"),
        "precision": 0.0, "recall": 0.0, "fscore": 0.0,
    }
    _empty_chamfer = {
        "chamfer_pred2gt": None, "chamfer_gt2pred": None, "chamfer_dist": None,
    }

    if pred_pc.shape[0] == 0 or gt_pc.shape[0] == 0:
        return {"recon": _empty_recon, "chamfer": _empty_chamfer}

    from chamfer_distance import ChamferDistance
    chamfer_fn = ChamferDistance()

    pred_t = torch.from_numpy(pred_pc).float().unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt_pc).float().unsqueeze(0).to(device)

    with torch.no_grad():
        sq_pred2gt, sq_gt2pred = chamfer_fn(pred_t, gt_t)

    l2_pred2gt = sq_pred2gt.squeeze(0).sqrt()  # (N,)
    l2_gt2pred = sq_gt2pred.squeeze(0).sqrt()  # (M,)

    accuracy = l2_pred2gt.mean().item()
    completeness = l2_gt2pred.mean().item()
    overall = (accuracy + completeness) / 2.0

    precision = (l2_pred2gt < threshold).float().mean().item()
    recall = (l2_gt2pred < threshold).float().mean().item()
    fscore = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "recon": {
            "acc": accuracy,
            "comp": completeness,
            "overall": overall,
            "precision": precision,
            "recall": recall,
            "fscore": fscore,
        },
    }

