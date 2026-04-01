from scipy.spatial import KDTree
from typing import Dict as TDict, Optional, Union
import numpy as np


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

