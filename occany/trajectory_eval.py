"""Trajectory evaluation utilities: ADE metrics, coordinate transforms, plotting."""

import json
import os

import numpy as np


def camera_to_ego_basis_matrix():
    # Camera convention: x right, y down, z forward
    # Ego/BEV convention: x forward, y left, z up
    return np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _to_homogeneous_4x4(c2w_np):
    if c2w_np.shape[-2:] == (4, 4):
        return c2w_np
    if c2w_np.shape[-2:] == (3, 4):
        c2w_h = np.zeros(c2w_np.shape[:-2] + (4, 4), dtype=c2w_np.dtype)
        c2w_h[..., :3, :4] = c2w_np
        c2w_h[..., 3, 3] = 1.0
        return c2w_h
    raise ValueError(f"Unsupported c2w shape: {c2w_np.shape}. Expected (...,3,4) or (...,4,4)")


def wrap_to_pi(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def get_output_c2w(output):
    c2w = output.get("c2w")
    if c2w is not None:
        return c2w

    c2w_pose = output.get("c2w_pose")
    if c2w_pose is not None:
        return c2w_pose

    pose_rotmat = output.get("pose_rotmat")
    pose_trans = output.get("pose_trans")
    if pose_rotmat is not None and pose_trans is not None:
        c2w = np.eye(4, dtype=np.float32)
        c2w = np.broadcast_to(c2w, pose_rotmat.shape[:-2] + (4, 4)).copy()
        c2w = pose_rotmat.new_tensor(c2w)
        c2w[..., :3, :3] = pose_rotmat
        c2w[..., :3, 3] = pose_trans
        return c2w

    available_keys = ", ".join(sorted(output.keys()))
    raise RuntimeError(
        "Output does not contain camera poses for trajectory evaluation. "
        f"Expected one of: c2w, c2w_pose, or pose_rotmat+pose_trans. Available keys: {available_keys}"
    )


def c2w_to_trajectory(c2w):
    """Convert camera-to-world poses to XY+heading trajectory.

    Args:
        c2w: torch.Tensor of shape (B, T, 3, 4) or (B, T, 4, 4).

    Returns:
        np.ndarray of shape (B, T, 3) with [x, z, theta] per frame,
        relative to the first frame.
    """
    if c2w is None:
        raise RuntimeError("Output does not contain `c2w` poses.")

    c2w_np = c2w.detach().float().cpu().numpy()
    c2w_h = _to_homogeneous_4x4(c2w_np)

    bsz, num_frames = c2w_h.shape[:2]
    traj_xy_theta = np.zeros((bsz, num_frames, 3), dtype=np.float32)

    for b in range(bsz):
        pose0_inv = np.linalg.inv(c2w_h[b, 0])
        for i in range(num_frames):
            local_pose = np.matmul(pose0_inv, c2w_h[b, i])
            origin = np.asarray(local_pose[:3, 3], dtype=np.float32)
            # Camera convention: x right, y down, z forward.
            forward_vec = local_pose[:3, :3][:, 2]
            theta = wrap_to_pi(np.arctan2(forward_vec[2], forward_vec[0]))
            traj_xy_theta[b, i, :] = [origin[0], origin[2], theta]

    return traj_xy_theta


def trajectory_path_length(traj_xy):
    if traj_xy is None or len(traj_xy) < 2:
        return 0.0
    deltas = np.diff(traj_xy[:, :2], axis=0)
    if len(deltas) == 0:
        return 0.0
    return float(np.linalg.norm(deltas, axis=1).sum())


# ─── Plotting ───────────────────────────────────────────────────────────────


def save_trajectory_plot(
    gt_traj,
    gen_traj_xy_theta,
    ade,
    sample_index,
    save_dir=None,
    out_path=None,
):
    """Save a comparison plot of GT vs predicted trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"traj_{sample_index:06d}.png")

    gen_traj = np.array(gen_traj_xy_theta[:, :2], copy=True)
    gt_plot = np.array(gt_traj[:, :2], copy=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        gt_plot[:, 0], gt_plot[:, 1],
        color="tab:green", linewidth=2.0, marker="o", markersize=2.0,
        label="GT trajectory",
    )
    ax.plot(
        gen_traj[:, 0], gen_traj[:, 1],
        color="tab:blue", linewidth=2.0, marker="o", markersize=2.0,
        label="Predicted trajectory",
    )

    ax.scatter(gt_plot[0, 0], gt_plot[0, 1], color="black", marker="s", s=36, label="Start")
    ax.scatter(gt_plot[-1, 0], gt_plot[-1, 1], color="tab:green", marker="x", s=64, label="GT end")
    ax.scatter(gen_traj[-1, 0], gen_traj[-1, 1], color="tab:blue", marker="x", s=64, label="Pred end")

    ax.set_title(f"Sample {sample_index:06d} | ADE={ade:.3f}")
    ax.set_xlabel("x (right)")
    ax.set_ylabel("y (forward)")
    ax.grid(True, linestyle="--", alpha=0.4)

    all_coords = np.concatenate([gt_plot, gen_traj], axis=0)
    x_min, y_min = all_coords.min(axis=0)
    x_max, y_max = all_coords.max(axis=0)
    ax.set_xlim(min(-5, x_min - 2.0), max(5, x_max + 2.0))
    ax.set_ylim(min(-5, y_min - 2.0), max(5, y_max + 2.0))
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ─── Batch evaluation ───────────────────────────────────────────────────────


def evaluate_trajectory_batch(
    batch,
    output,
    evaluated_samples,
    plot_every,
    trajectory_plot_dir,
):
    """Compute ADE for a batch and optionally save trajectory plots.

    Returns:
        (mean_ade, list_of_per_sample_ade)  or  (None, []) when GT is absent.
    """
    if "gt_trajectory" not in batch:
        return None, []

    gt_traj = batch["gt_trajectory"].detach().cpu().numpy()
    gen_traj_xy_theta = c2w_to_trajectory(get_output_c2w(output))
    gen_traj = gen_traj_xy_theta[:, :, :2]

    if gt_traj.shape != gen_traj.shape:
        raise RuntimeError(
            f"GT trajectory shape {gt_traj.shape} != predicted shape {gen_traj.shape}"
        )

    ade_list = []
    for b in range(gt_traj.shape[0]):
        sample_index = evaluated_samples + b
        eval_gt = gt_traj[b]
        ade = float(np.mean(np.linalg.norm(eval_gt - gen_traj[b], axis=1)))
        ade_list.append(ade)

        if plot_every > 0 and sample_index % plot_every == 0:
            save_trajectory_plot(
                gt_traj=eval_gt,
                gen_traj_xy_theta=gen_traj_xy_theta[b],
                ade=ade,
                sample_index=sample_index,
                save_dir=trajectory_plot_dir,
            )

    return float(np.mean(ade_list)), ade_list


# ─── Metrics I/O ────────────────────────────────────────────────────────────


def save_trajectory_metrics(metrics_path, ade_scores, n_frames, height, width, data_root, anno_file):
    if len(ade_scores) == 0:
        return

    metrics = {
        "num_samples": len(ade_scores),
        "mean_ADE": float(np.mean(ade_scores)),
        "ade_per_sample": ade_scores,
        "n_frames": n_frames,
        "height": height,
        "width": width,
        "data_root": data_root,
        "anno_file": anno_file,
    }

    out_dir = os.path.dirname(metrics_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
