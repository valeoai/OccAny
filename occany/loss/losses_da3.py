from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from occany.utils.helpers import intrinsics_c2w_to_raymap



class L21Loss(nn.Module):
    """Euclidean distance between 3d points (L2 norm loss)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape and a.ndim >= 2, f"Bad shape = {a.shape}"
        dist = torch.norm(a - b, dim=-1)  # L2 distance, one dimension less
        assert dist.ndim == a.ndim - 1

        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")


class L11Loss(nn.Module):
    """Manhattan distance between 3d points (L1 norm loss)."""

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Invalid reduction mode: {reduction}")
        self.reduction = reduction

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        assert a.shape == b.shape and a.ndim >= 2, f"Bad shape = {a.shape}"
        dist = torch.abs(a - b).sum(dim=-1)  # L1 distance, one dimension less
        assert dist.ndim == a.ndim - 1

        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")


class PointmapLoss(nn.Module):
    def __init__(self, reduction: str = "mean", lambda_c: float = 0.0, 
                 norm_mode: str = "avg_dis", gt_scale: bool = False, loss_type: str = 'L2'):
        """
        Args:
            reduction: 'mean' or 'sum' for final loss aggregation.
            lambda_c (float): Weight for the log confidence term. 
                              When lambda_c=0.0, confidence is not used.
                              When lambda_c>0.0, confidence-aware loss is computed:
                              conf * loss - lambda_c * log(conf)
            norm_mode (str): Normalization mode for pointmaps (e.g., 'avg_dis', 'median_dis').
            gt_scale (bool): If True, do not normalize pointmaps (evaluate at GT scale).
            loss_type (str): 'L1' or 'L2' loss.
        """
        super().__init__()
        if loss_type == 'L1':
            self.criterion = L11Loss(reduction="none")
        elif loss_type == 'L2':
            self.criterion = L21Loss(reduction="none")
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}. Expected 'L1' or 'L2'.")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"bad {reduction=} mode")
        self.reduction = reduction
        self.lambda_c = lambda_c
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def forward(
        self, 
        pred_pointmap: torch.Tensor, 
        gt_pointmap: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        confidence: Optional[torch.Tensor] = None
    ):
        """
        Computes pointmap loss, optionally with confidence weighting.
        
        Args:
            pred_pointmap: Predicted pointmap (..., 3).
            gt_pointmap: Ground truth pointmap (..., 3).
            mask: Valid pixel mask (1 for valid, 0 for invalid).
            confidence: Confidence map for predictions. Required if lambda_c > 0.
        
        Returns:
            loss: Scalar loss value.
            logs: Dictionary with loss breakdown.
        """
        if pred_pointmap.shape != gt_pointmap.shape:
            raise ValueError(f"Pointmap shape mismatch: pred={tuple(pred_pointmap.shape)} gt={tuple(gt_pointmap.shape)}")

        if pred_pointmap.ndim < 2 or pred_pointmap.shape[-1] != 3:
            raise ValueError(f"Expected pointmap of shape (..., 3), got {tuple(pred_pointmap.shape)}")

        if self.gt_scale:
            norm_factor = 1.0
        else:
            # Per-scene normalization: compute average distance per sample
            # gt_pointmap: (B, H, W, 3) or (B, T, H, W, 3)
            gt_dis = gt_pointmap.norm(dim=-1)  
            
            # Spatial dimensions are all except the first (Batch)
            spatial_dims = tuple(range(1, gt_dis.ndim))
            
            if mask is not None:
                mask_f = mask.to(dtype=gt_dis.dtype)
                # Expand mask to match gt_dis if needed
                if mask_f.ndim == gt_dis.ndim + 1 and mask_f.shape[-1] == 1:
                    mask_f = mask_f[..., 0]
                
                masked_sum = (gt_dis * mask_f).sum(dim=spatial_dims, keepdim=True)
                valid_count = mask_f.sum(dim=spatial_dims, keepdim=True).clamp(min=1e-8)
                norm_factor = masked_sum / valid_count
            else:
                norm_factor = gt_dis.mean(dim=spatial_dims, keepdim=True)
            
            norm_factor = norm_factor.clamp(min=1e-8)
            # Expand for broadcasting with (..., 3) pointmap
            norm_factor = norm_factor.unsqueeze(-1)
        
        # Apply the same normalization to both pred and gt
        gt_pointmap_norm = gt_pointmap / norm_factor
        pred_pointmap_norm = pred_pointmap / norm_factor
        
        per_point = self.criterion(pred_pointmap_norm, gt_pointmap_norm)
        
        # Apply confidence weighting if lambda_c > 0
        if self.lambda_c > 0.0 and confidence is not None:
            # Ensure confidence has the right shape (should match per_point)
            if confidence.ndim == per_point.ndim + 1 and confidence.shape[-1] == 1:
                confidence = confidence[..., 0]
            if confidence.shape != per_point.shape:
                raise ValueError(f"Confidence shape mismatch: conf={tuple(confidence.shape)} per_point={tuple(per_point.shape)}")
            
            # Avoid log(0) by clamping
            eps = 1e-6
            conf_safe = torch.clamp(confidence, min=eps)
            # Confidence-weighted loss: conf * loss - lambda_c * log(conf)
            per_point = confidence * per_point - self.lambda_c * torch.log(conf_safe)
            
        if mask is None:
            loss = per_point.mean() if per_point.numel() > 0 else per_point.new_zeros(())
            return loss, {"loss_pointmap": float(loss)}

        if mask.ndim == per_point.ndim + 1 and mask.shape[-1] == 1:
            mask = mask[..., 0]
        if mask.shape != per_point.shape:
            raise ValueError(f"Mask shape mismatch: mask={tuple(mask.shape)} per_point={tuple(per_point.shape)}")

        mask_f = mask.to(dtype=per_point.dtype)
        denom = mask_f.sum().clamp(min=1e-8)
        masked_sum = (per_point * mask_f).sum()

        if self.reduction == "sum":
            loss = masked_sum
        else:
            loss = masked_sum / denom

        return loss, {"loss_pointmap": float(loss)}


class DepthLosses(nn.Module):
    def __init__(self, lambda_c=1.0, alpha=1.0, detach_confidence=False, gt_scale=False):
        """
        Args:
            lambda_c (float): Weight for the log confidence term in L_D.
                              When lambda_c=0.0, confidence is not used (simple L1 loss).
                              When lambda_c>0.0, confidence-aware loss is computed.
            alpha (float): Weight for the gradient loss term in total loss.
            detach_confidence (bool): If True, detach confidence from the computation graph.
            gt_scale (bool): If True, do not normalize depth (evaluate at GT scale).
        """
        super().__init__()
        self.lambda_c = lambda_c
        self.alpha = alpha
        self.detach_confidence = detach_confidence
        self.gt_scale = gt_scale

    def forward(self, pred_depth, gt_depth, confidence, mask=None):
        """
        Computes the composite depth loss.
        
        Args:
            pred_depth (Tensor): Predicted depth map [B, 1, H, W].
            gt_depth (Tensor): Ground truth depth map [B, 1, H, W].
            confidence (Tensor): Confidence map D_c [B, 1, H, W].
            mask (Tensor, optional): Valid pixel mask (1 for valid, 0 for invalid) [B, 1, H, W].
                                     If None, assumes all pixels are valid.
        
        Returns:
            total_loss (Tensor): The weighted sum of L_D and L_grad.
            logs (dict): Dictionary containing individual loss values for logging.
        """
        # Create mask of ones if not provided
        if mask is None:
            mask = torch.ones_like(pred_depth)

        if self.gt_scale:
            norm_factor = 1.0
        else:
            # Per-scene normalization: compute average depth per sample
            # gt_depth: (B, 1, H, W)
            spatial_dims = (2, 3)
            
            if mask is not None:
                mask_f = mask.to(dtype=gt_depth.dtype)
                masked_sum = (gt_depth * mask_f).sum(dim=spatial_dims, keepdim=True)
                valid_count = mask_f.sum(dim=spatial_dims, keepdim=True).clamp(min=1e-8)
                norm_factor = masked_sum / valid_count
            else:
                norm_factor = gt_depth.mean(dim=spatial_dims, keepdim=True)
            
            norm_factor = norm_factor.clamp(min=1e-8)
        # 1. Compute Confidence-aware Depth Loss (L_D)
        l_depth = self.confidence_depth_loss(pred_depth/norm_factor, gt_depth/norm_factor, confidence, mask)

        total_loss, logs = l_depth, {"loss_depth": l_depth}
        
        if self.alpha > 0.0:
            # 2. Compute Gradient Loss (L_grad)
            l_grad = self.gradient_loss(pred_depth, gt_depth, mask)
            total_loss += self.alpha * l_grad
            logs["loss_grad"] = l_grad

        return total_loss, logs

    def confidence_depth_loss(self, pred, target, confidence, mask):
        """
        Implements Equation: 
        L_D = (1/Z) * sum( mask * ( confidence * |pred - target| - lambda_c * log(confidence) ) )
        If lambda_c=0, implements simple L1 loss:
        L_D = (1/Z) * sum( mask * |pred - target| )
        """
        # L1 absolute difference
        abs_diff = torch.abs(pred - target)
        
        if self.lambda_c > 0.0 and confidence is not None:
            # Detach confidence from computation graph if specified
            if self.detach_confidence:
                confidence = confidence.detach()
            
            # Avoid log(0) numerical instability by clamping confidence or adding epsilon
            eps = 1e-6
            confidence_safe = torch.clamp(confidence, min=eps)
            
            # Calculate per-pixel loss term
            # Term 1: D_{c,p} * |pred - target|
            # Term 2: lambda_c * log(D_{c,p})
            pixel_loss = (confidence * abs_diff) - (self.lambda_c * torch.log(confidence_safe))

            # Apply mask
            masked_loss = pixel_loss * mask
        else:
            # Simple L1 loss
            masked_loss = abs_diff * mask
        
        # Normalize by Z_omega (number of valid pixels in mask)
        # We use a small epsilon in divisor to avoid division by zero
        normalization = torch.sum(mask) + 1e-8
        
        loss = torch.sum(masked_loss) / normalization
        return loss

    def gradient_loss(self, pred, target, mask):
        """
        Implements Equation (3):
        L_grad = || grad_x(pred) - grad_x(target) ||_1 + || grad_y(pred) - grad_y(target) ||_1
        """
        # Compute gradients using finite difference
        pred_dx, pred_dy = self._compute_gradients(pred)
        target_dx, target_dy = self._compute_gradients(target)
        
        # Adjust mask for gradients (gradients reduce spatial dim by 1)
        # We crop the mask to match the gradient shapes
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

        # L1 Loss on gradients
        loss_dx = torch.abs(pred_dx - target_dx) * mask_dx
        loss_dy = torch.abs(pred_dy - target_dy) * mask_dy
        
        # Normalize
        # Here we normalize by the number of valid gradient pixels for stability.
        norm_dx = torch.sum(mask_dx) + 1e-8
        norm_dy = torch.sum(mask_dy) + 1e-8
        
        return (torch.sum(loss_dx) / norm_dx) + (torch.sum(loss_dy) / norm_dy)

    def _compute_gradients(self, img):
        """
        Computes horizontal (dx) and vertical (dy) finite differences.
        Returns:
            dx: Tensor of shape [B, C, H, W-1]
            dy: Tensor of shape [B, C, H-1, W]
        """
        # Horizontal difference: img[:, :, :, i+1] - img[:, :, :, i]
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        
        # Vertical difference: img[:, :, i+1, :] - img[:, :, i, :]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        
        return dx, dy

class RaymapLoss(nn.Module):
    """Loss for raymap prediction.
    
    Converts GT camera poses (extrinsics) and intrinsics to world-space rays,
    then computes confidence-weighted L2 loss against predicted ray field.
    
    The predicted ray field from DA3 has shape (B, T, H, W, 6) where:
    - ray[..., :3] = ray directions (normalized)
    - ray[..., 3:] = ray origins
    """
    def __init__(self, lambda_c: float = 1.0, reduction: str = "mean", 
                 gt_scale: bool = False, loss_type: str = 'L2'):
        """
        Args:
            lambda_c: Weight for the log confidence term (similar to DepthLosses).
            reduction: 'mean' or 'sum' for final loss aggregation.
            gt_scale: If True, do not normalize origins (evaluate at GT scale).
            loss_type: 'L1' or 'L2' loss.
        """
        super().__init__()
        self.lambda_c = lambda_c
        if loss_type == 'L1':
            self.criterion = L11Loss(reduction="none")
        elif loss_type == 'L2':
            self.criterion = L21Loss(reduction="none")
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}. Expected 'L1' or 'L2'.")
        if reduction not in {"mean", "sum"}:
            raise ValueError(f"bad {reduction=} mode")
        self.reduction = reduction
        self.gt_scale = gt_scale

    def forward(
        self,
        pred_ray: torch.Tensor,       # (B, T, H, W, 6) - [directions(3), origins(3)]
        pred_ray_conf: torch.Tensor,  # (B, T, H, W)
        gt_c2w: torch.Tensor,         # (B, T, 4, 4) - camera-to-world (NOT world-to-camera!)
        gt_intrinsics: torch.Tensor,  # (B, T, 3, 3)
        gt_raymap: torch.Tensor,  # (B, T, H, W, 6) - optional pre-computed GT raymap
    ):
        """
        Computes raymap loss between predicted and GT rays.
        
        Args:
            pred_ray: Predicted ray field from DA3 model.
            pred_ray_conf: Confidence map for ray predictions.
            gt_c2w: Ground truth camera-to-world matrices (NOT w2c!).
            gt_intrinsics: Ground truth camera intrinsics.
            gt_raymap: Optional pre-computed GT raymap. If provided, this is used directly
                       instead of computing from gt_c2w and gt_intrinsics. This allows
                       verification that the dataset's raymap matches the pts3d computation.
            
        Returns:
            total_loss: Scalar loss value.
            logs: Dictionary with loss breakdown.
        """
        B, T, H, W, _ = pred_ray.shape
        dtype = pred_ray.dtype

        
        # Use pre-computed GT raymap if provided, otherwise compute from intrinsics and c2w
        # if gt_raymap is not None:
        #     # Use pre-computed raymap from dataset
        assert gt_raymap.shape == pred_ray.shape, f"gt_raymap shape {gt_raymap.shape} != pred_ray shape {pred_ray.shape}"
        # else:
        # Compute GT raymap from intrinsics and c2w
        gt_raymap_torch = intrinsics_c2w_to_raymap(
            gt_intrinsics,  # (B, T, 3, 3)
            gt_c2w,         # (B, T, 4, 4) - camera-to-world
            H,
            W,
        )  # (B, T, H, W, 6)
        assert (gt_raymap_torch - gt_raymap).abs().max() < 1e-3, "gt_raymap and gt_raymap_torch do not match"
        # Extract GT directions and origins from raymap
        gt_directions = gt_raymap[..., :3]  # (B, T, H, W, 3)
        gt_origins = gt_raymap[..., 3:]     # (B, T, H, W, 3)

        # Extract predicted directions and origins
        pred_directions = pred_ray[..., :3]  # (B, T, H, W, 3)
        pred_origins = pred_ray[..., 3:]     # (B, T, H, W, 3)

        if self.gt_scale:
            norm_factor = 1.0
        else:
            # Per-scene normalization: compute average origin distance per sample
            # gt_origins: (B, T, H, W, 3)
            gt_origin_norms = gt_origins.norm(dim=-1)
            
            # Spatial and temporal dimensions are all except the first (Batch)
            spatial_dims = tuple(range(1, gt_origin_norms.ndim))
            
            norm_factor = gt_origin_norms.mean(dim=spatial_dims, keepdim=True).clamp(min=1e-8)
            # Expand for broadcasting with (..., 3) origins
            norm_factor = norm_factor.unsqueeze(-1)
       
        pred_origins_normalized = pred_origins / norm_factor
        gt_origins_normalized = gt_origins / norm_factor

        
        # Compute L2 distance for directions and origins separately
        dir_loss_per_pixel = self.criterion(pred_directions, gt_directions.to(dtype))  # (B, T, H, W)
        origin_loss_per_pixel = self.criterion(pred_origins_normalized, gt_origins_normalized.to(dtype))     # (B, T, H, W)
        
        # Combined per-pixel loss
        per_pixel_loss = 10 * dir_loss_per_pixel + origin_loss_per_pixel
      
        # Apply confidence weighting if lambda_c > 0
        if self.lambda_c > 0.0:
            eps = 1e-6
            conf_safe = torch.clamp(pred_ray_conf, min=eps)
            
            # Confidence-weighted loss: conf * loss - lambda_c * log(conf)
            weighted_loss = (conf_safe * per_pixel_loss) - (self.lambda_c * torch.log(conf_safe))
        else:
            # Simple loss without confidence weighting
            weighted_loss = per_pixel_loss
        
        # Apply reduction
        if self.reduction == "sum":
            total_loss = weighted_loss.sum()
        else:
            total_loss = weighted_loss.mean()
        
        # Compute unweighted losses for logging
        dir_loss = dir_loss_per_pixel.mean()
        origin_loss = origin_loss_per_pixel.mean()
        
        logs = {
            "loss_raymap": float(total_loss),
            "loss_ray_dir": float(dir_loss),
            "loss_ray_origin": float(origin_loss),
        }
        
        return total_loss, logs



class DistillLoss(nn.Module):
    """
    Distillation loss for multi-scale feature maps.
    Unlike the implementation in dust3r, this inherits from nn.Module 
    to support standard PyTorch losses without the BaseCriterion restriction.
    """
    def __init__(self, criterion, use_conf=False, scale_weights=None):
        super().__init__()
        self.criterion = criterion
        self.use_conf = use_conf
        # Default weights to balance contribution based on typical loss values:
        # high_res: 0.045, mid_res: 1.435, low_res: 0.73, pre_neck: 1.64
        # self.scale_weights = scale_weights or [4.0, 1.0, 1.0, 0.5] # Might use later
        self.scale_weights = scale_weights or [1.0, 1.0, 1.0, 1.0]


        # If using confidence, the underlying criterion should return unreduced loss
        if self.use_conf and hasattr(self.criterion, 'reduction'):
            self.criterion.reduction = 'none'

    def forward(self, pred_feats, gt_feats, conf=None):
        """
        Compute distillation loss for feature maps.
        
        Args:
            pred_feats: List/tuple of predicted features (e.g., [s0, s1, s2, pre_neck])
            gt_feats: List/tuple of teacher features
            conf: Optional confidence map for spatial weighting
        """
        details = {}
        num_feats = len(pred_feats)
        
        # Feature names for logging
        feat_names = ["distill_loss_high_res", "distill_loss_mid_res", "distill_loss_low_res", "distill_loss_pre_neck"]
        
        loss = 0
        unweighted_total_loss = 0
        
        # Ensure we don't index beyond feat_names
        def get_feat_name(idx):
            if idx < len(feat_names):
                return feat_names[idx]
            return f"distill_loss_feat_{idx}"

        use_conf_weighting = self.use_conf and conf is not None
        if use_conf_weighting:
            # Flatten B and T dimensions if needed: (B, T, H, W) -> (B*T, 1, H, W)
            if conf.ndim == 4:
                conf_spatial = conf.view(conf.shape[0] * conf.shape[1], 1, conf.shape[2], conf.shape[3])
            else:
                conf_spatial = conf
                if conf_spatial.ndim == 3:
                    conf_spatial = conf_spatial.unsqueeze(1)

        for i in range(num_feats):
            p_feat = pred_feats[i]
            g_feat = gt_feats[i]
            
            # Flatten B and T/V dimensions: (B, T, C, H, W) -> (B*T, C, H, W)
            if p_feat.ndim == 5:
                p_feat = p_feat.flatten(0, 1)
            if g_feat.ndim == 5:
                g_feat = g_feat.flatten(0, 1)

            if use_conf_weighting:
                # Compute raw pixel-level loss
                pixel_loss = self.criterion(p_feat, g_feat)
                
                # Average over channel dimension if criterion didn't
                if pixel_loss.ndim == 4:
                    pixel_loss = pixel_loss.mean(dim=1, keepdim=True)
                
                # Interpolate loss to match confidence map resolution
                if pixel_loss.shape[-2:] != conf_spatial.shape[-2:]:
                    pixel_loss = F.interpolate(pixel_loss, size=conf_spatial.shape[-2:], mode="bilinear", align_corners=False)
                
                # Weight by confidence
                weighted_loss_spatial = (pixel_loss * conf_spatial).sum(dim=(1, 2, 3)) / conf_spatial.sum(dim=(1, 2, 3)).clamp(min=1e-8)
                item_loss = weighted_loss_spatial.mean()
            else:
                item_loss = self.criterion(p_feat, g_feat)
                if item_loss.numel() > 1:
                    item_loss = item_loss.mean()
            
            # Unified scale weight application and logging
            weight = self.scale_weights[i] if i < len(self.scale_weights) else 1.0
            weighted_item_loss = item_loss * weight
            
            loss += weighted_item_loss
            unweighted_total_loss += item_loss
            
            # Log both raw (for comparison) and weighted (actual) values
            feat_name = get_feat_name(i)
            details[feat_name] = float(item_loss)
            details[feat_name + "_weighted"] = float(weighted_item_loss)

        details["distill_loss"] = float(loss)
        details["distill_loss_unweighted"] = float(unweighted_total_loss)
        return loss, details



class ScaleInvariantDepthLoss(nn.Module):
    """
    Scale-invariant depth loss using median alignment.
    1. Compute scale from GT: scale = median(gt_depth / frozen_depth)
    2. Create pseudo-GT: pseudo_gt = frozen_depth * scale
    3. Supervise trainable: loss = L1(trainable_depth, pseudo_gt)
    """
    def __init__(self, min_depth=0.01, max_depth=50.0, max_scale=100.0, alpha=1.0,
                 l1_weight=1.0, max_error_after=2.5, use_ransac=True, ransac_iters=100,
                 ransac_inlier_thresh=0.1, ransac_sample_size=50):
        super().__init__()
        self.min_depth = min_depth  # Filter out depth values smaller than this
        self.max_depth = max_depth  # Filter out depth values larger than this
        self.max_scale = max_scale  # Clamp scale to this maximum
        self.alpha = alpha  # Weight for the gradient loss term
        self.use_ransac = use_ransac  # Whether to use RANSAC for scale computation
        self.ransac_iters = ransac_iters  # Number of RANSAC iterations
        self.ransac_inlier_thresh = ransac_inlier_thresh  # Inlier threshold (relative error)
        self.ransac_sample_size = ransac_sample_size  # Number of samples per RANSAC iteration

    def compute_scale_median(self, gt, pred, mask=None):
        """
        Compute per-sample scale: scale = median(gt / pred)
        With numerical stability safeguards.
        """
        B = gt.shape[0]
        gt_flat = gt.reshape(B, -1)
        pred_flat = pred.reshape(B, -1)
        
        # Create a combined validity mask:
        # 1. Provided mask (if any)
        # 2. Depth values must be above min_depth threshold
        # 3. Depth values must be below max_depth threshold
        valid = (pred_flat > self.min_depth) & (gt_flat > self.min_depth)
        valid = valid & (gt_flat < self.max_depth)
        if mask is not None:
            mask_flat = mask.reshape(B, -1) > 0
            valid = valid & mask_flat
        
        # Compute per-pixel ratio only for valid pixels
        ratio = torch.where(valid, gt_flat / (pred_flat + 1e-8), torch.tensor(float('nan'), device=gt.device))
        
        # Compute nanmedian of ratios
        scale = torch.nanmedian(ratio, dim=1).values
        
        # Handle cases where all pixels are invalid (all NaN)
        all_invalid = ~valid.any(dim=1)
        scale = torch.where(all_invalid, torch.ones_like(scale), scale)
        
        # Clamp scale to reasonable bounds
        scale = scale.clamp(min=1.0 / self.max_scale, max=self.max_scale)
            
        return scale

    def compute_scale_ransac(self, gt, pred, mask=None):
        """
        Compute per-sample scale using RANSAC for robustness against outliers.
        Fully vectorized implementation - parallelizes both batch and RANSAC iterations.
        
        Args:
            gt: (B, N) ground truth depth (flattened)
            pred: (B, N) predicted/auxiliary depth (flattened)
            mask: (B, N) optional validity mask (flattened)
            
        Returns:
            scale: (B,) per-sample scale factors
        """
        B = gt.shape[0]
        device = gt.device
        dtype = gt.dtype
        
        gt_flat = gt.reshape(B, -1)
        pred_flat = pred.reshape(B, -1)
        N = gt_flat.shape[1]  # Number of pixels per sample
        
        # Create validity mask
        valid = (pred_flat > self.min_depth) & (gt_flat > self.min_depth) & (gt_flat < self.max_depth)
        if mask is not None:
            mask_flat = mask.reshape(B, -1) > 0
            valid = valid & mask_flat
        
        # Compute ratios for all pixels (NaN where invalid)
        ratio = torch.where(valid, gt_flat / (pred_flat + 1e-8), 
                            torch.full_like(gt_flat, float('nan')))
        
        # Count valid pixels per batch
        n_valid = valid.sum(dim=1)  # (B,)
        
        # Fallback scales using median (for samples with too few valid pixels)
        fallback_scales = torch.nanmedian(ratio, dim=-1).values  # (B,)
        
        # Check if any batch has enough valid pixels for RANSAC
        has_enough = n_valid >= self.ransac_sample_size
        if not has_enough.any():
            # All batches use fallback
            best_scales = torch.where(torch.isnan(fallback_scales), 
                                       torch.ones_like(fallback_scales), fallback_scales)
            return best_scales.clamp(min=1.0 / self.max_scale, max=self.max_scale)
        
        # Sample candidate indices from valid pixels only.
        # Invalid pixels are assigned low random scores and excluded when enough valid points exist.
        valid_expanded = valid.unsqueeze(1).expand(-1, self.ransac_iters, -1)  # (B, ransac_iters, N)
        rand_scores = torch.rand(B, self.ransac_iters, N, device=device, dtype=dtype)
        rand_scores = rand_scores.masked_fill(~valid_expanded, -1.0)
        sample_idx = rand_scores.topk(k=self.ransac_sample_size, dim=-1).indices  # (B, ransac_iters, sample_size)
        
        # Gather sample ratios: (B, ransac_iters, sample_size)
        ratio_expanded = ratio.unsqueeze(1).expand(-1, self.ransac_iters, -1)  # (B, ransac_iters, N)
        sample_ratios = torch.gather(ratio_expanded, dim=2, index=sample_idx)
        sample_valid = torch.gather(valid_expanded, dim=2, index=sample_idx)
        sample_ratios = torch.where(sample_valid, sample_ratios, torch.full_like(sample_ratios, float('nan')))
        
        # Compute candidate scales from samples: (B, ransac_iters)
        candidate_scales = torch.nanmedian(sample_ratios, dim=-1).values
        
        # Compute relative errors for all pixels against all candidate scales
        # ratio: (B, N) -> (B, 1, N), candidate_scales: (B, ransac_iters) -> (B, ransac_iters, 1)
        relative_error = torch.abs(ratio.unsqueeze(1) - candidate_scales.unsqueeze(-1)) / (
            candidate_scales.unsqueeze(-1) + 1e-8
        )  # (B, ransac_iters, N)
        
        # Count inliers per iteration (only where valid)
        inliers = (relative_error < self.ransac_inlier_thresh) & valid.unsqueeze(1)  # (B, ransac_iters, N)
        inlier_counts = inliers.sum(dim=-1)  # (B, ransac_iters)
        
        # Handle NaN candidate scales (set their inlier count to 0)
        inlier_counts = torch.where(torch.isnan(candidate_scales), 
                                     torch.zeros_like(inlier_counts), inlier_counts)
        
        # Find best iteration per batch
        best_iter = inlier_counts.argmax(dim=1)  # (B,)
        
        # Gather the best inlier mask for each batch: (B, N)
        batch_indices = torch.arange(B, device=device).view(B, 1, 1)
        iter_indices = best_iter.view(B, 1, 1)
        pixel_indices = torch.arange(N, device=device).view(1, 1, N).expand(B, 1, N)
        best_inlier_mask = inliers[batch_indices, iter_indices, pixel_indices].squeeze(1)  # (B, N)
        
        # Compute refined scale from inliers
        inlier_ratios = torch.where(best_inlier_mask, ratio, 
                                     torch.full_like(ratio, float('nan')))  # (B, N)
        best_scales = torch.nanmedian(inlier_ratios, dim=-1).values  # (B,)
        
        # Use fallback for samples with too few valid pixels
        best_scales = torch.where(has_enough, best_scales, fallback_scales)
        
        # Handle NaN scales: use fallback first, then default to 1.0.
        best_scales = torch.where(torch.isnan(best_scales), fallback_scales, best_scales)
        best_scales = torch.where(torch.isnan(best_scales), 
                                   torch.ones_like(best_scales), best_scales)
        
        # Clamp scale to reasonable bounds
        best_scales = best_scales.clamp(min=1.0 / self.max_scale, max=self.max_scale)
        
        return best_scales

    def forward(self, train_depth, aux_depth, gt_depth, mask=None, confidence=None):
        """
        Args:
            train_depth: (B, T, H, W) or (B, H, W) - trainable
            aux_depth: (B, T, H, W) or (B, H, W) - frozen
            gt_depth: (B, T, H, W) or (B, H, W) - ground truth
            mask: (B, T, H, W) or (B, H, W) - valid mask
            confidence: Optional confidence map (unused, kept for call-site compatibility)
        """
        # Ensure 4D (B, T, H, W)
        if train_depth.ndim == 3:
            train_depth = train_depth.unsqueeze(1)
        if aux_depth.ndim == 3:
            aux_depth = aux_depth.unsqueeze(1)
        if gt_depth.ndim == 3:
            gt_depth = gt_depth.unsqueeze(1)
        if mask is not None and mask.ndim == 3:
            mask = mask.unsqueeze(1)
            
        B, T, H, W = train_depth.shape
        # Flatten T, H, W to compute one scale per batch item (shared across all views)
        train_flat = train_depth.reshape(B, T * H * W)
        aux_flat = aux_depth.reshape(B, T * H * W)
        gt_flat = gt_depth.reshape(B, T * H * W)
        mask_flat = mask.reshape(B, T * H * W) if mask is not None else None
        # print(train_flat.mean())
        # print(aux_flat.mean())
        with torch.no_grad():
            if self.use_ransac:
                scales = self.compute_scale_ransac(gt_flat, aux_flat, mask_flat)  # (B,)
            else:
                scales = self.compute_scale_median(gt_flat, aux_flat, mask_flat)  # (B,)
        
        # Reshape back to (B, T, H, W) for loss computation
        train_flat = train_depth.reshape(B * T, H, W)
        aux_flat = aux_depth.reshape(B * T, H, W)
        mask_flat = mask.reshape(B * T, H, W) if mask is not None else None
        
        # Repeat scales for each view: (B,) -> (B*T,)
        scales = scales.repeat_interleave(T)
        
        # Pseudo-GT = aux * scale
        pseudo_gt = aux_flat * scales.view(-1, 1, 1)

        # DEBUG: save pseudo_gt + gt_depth to test.png
        # import numpy as np
        # from occany.utils.helpers import depth2rgb
        # _pg = pseudo_gt[0].detach().cpu().numpy()
        # _gt = gt_depth[0, 0].detach().cpu().numpy()
        # _pg_rgb = depth2rgb(_pg, valid_mask=_pg > 0)
        # _gt_rgb = depth2rgb(_gt, valid_mask=_gt > 0)
        # import cv2
        # cv2.imwrite("test.png", np.concatenate([_pg_rgb, _gt_rgb], axis=1))
        # breakpoint()
        
        # Create loss mask: valid mask AND depth within [min_depth, max_depth]
        loss_mask = (train_flat > self.min_depth)
        loss_mask = loss_mask & (pseudo_gt > self.min_depth) & (pseudo_gt < self.max_depth)
        if mask_flat is not None:
            loss_mask = loss_mask & (mask_flat > 0)
        
        # Loss = L1(trainable, pseudo_gt)
        if not loss_mask.any():
            return train_depth.sum() * 0.0, {
                "loss_scale_inv_depth": 0.0, 
                "loss_scale_inv_grad": 0.0,
                "avg_scale": float(scales.mean())
            }
        
        l1_loss = F.l1_loss(train_flat[loss_mask], pseudo_gt[loss_mask])
        total_loss = l1_loss
        logs = {"loss_scale_inv_depth": float(l1_loss), "avg_scale": float(scales.mean())}
        
        # Compute gradient loss between pred and pseudo_gt
        if self.alpha > 0.0:
            grad_loss = self._gradient_loss(train_flat, pseudo_gt, loss_mask)
            total_loss = total_loss + self.alpha * grad_loss
            logs["loss_scale_inv_grad"] = float(grad_loss)
            
        return total_loss, logs
    
    def _gradient_loss(self, pred, target, mask):
        """
        Computes gradient loss: L1 difference of gradients.
        Only computes gradients where both neighboring pixels are valid.
        Args:
            pred: (B, H, W)
            target: (B, H, W)
            mask: (B, H, W) valid mask (boolean or 0/1)
        """
        # Add channel dimension for gradient computation: (B, 1, H, W)
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)
        mask = mask.unsqueeze(1).float()
        
        # Create valid gradient masks BEFORE computing gradients
        # A gradient is only valid if BOTH neighboring pixels are valid
        mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]  # (B, 1, H, W-1)
        mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]  # (B, 1, H-1, W)
        
        # Compute gradients using finite difference
        # Only compute where mask is valid to avoid meaningless values
        pred_dx, pred_dy = self._compute_gradients_masked(pred, mask_dx, mask_dy)
        target_dx, target_dy = self._compute_gradients_masked(target, mask_dx, mask_dy)

        # L1 Loss on gradients (only on valid gradient locations)
        loss_dx = torch.abs(pred_dx - target_dx) * mask_dx
        loss_dy = torch.abs(pred_dy - target_dy) * mask_dy
        
        # Normalize by number of valid gradient pixels
        norm_dx = mask_dx.sum().clamp(min=1e-8)
        norm_dy = mask_dy.sum().clamp(min=1e-8)
        
        return (loss_dx.sum() / norm_dx) + (loss_dy.sum() / norm_dy)
    
    def _compute_gradients_masked(self, img, mask_dx, mask_dy):
        """
        Computes horizontal (dx) and vertical (dy) finite differences.
        Only computes gradients where the mask indicates both neighbors are valid.
        Args:
            img: (B, C, H, W)
            mask_dx: (B, C, H, W-1) valid mask for horizontal gradients
            mask_dy: (B, C, H-1, W) valid mask for vertical gradients
        Returns:
            dx: Tensor of shape [B, C, H, W-1], zeroed where invalid
            dy: Tensor of shape [B, C, H-1, W], zeroed where invalid
        """
        # Compute raw gradients
        dx = img[:, :, :, 1:] - img[:, :, :, :-1]
        dy = img[:, :, 1:, :] - img[:, :, :-1, :]
        
        # Zero out invalid gradients (where either neighbor is invalid)
        # This prevents large spurious gradients at mask boundaries
        dx = dx * mask_dx
        dy = dy * mask_dy
        
        return dx, dy


