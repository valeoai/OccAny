from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import L1Loss, MSELoss

from dust3r.losses import *  # noqa: F401,F403


class DistillLoss(nn.Module):
    def __init__(self, criterion: nn.Module, use_conf: bool = False):
        super().__init__()
        self.criterion = criterion
        self.use_conf = use_conf

    def _elementwise_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(self.criterion, nn.MSELoss):
            return F.mse_loss(prediction, target, reduction="none")
        if isinstance(self.criterion, nn.L1Loss):
            return F.l1_loss(prediction, target, reduction="none")
        raise TypeError(
            f"DistillLoss only supports nn.MSELoss and nn.L1Loss for confidence-aware reduction, got {type(self.criterion)!r}"
        )

    def _resize_confidence(self, confidence: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        conf = confidence
        if conf.ndim == reference.ndim - 1:
            conf = conf.unsqueeze(2)
        if conf.shape[-2:] != reference.shape[-2:]:
            flat = conf.reshape(-1, conf.shape[-3], conf.shape[-2], conf.shape[-1])
            flat = F.interpolate(flat, size=reference.shape[-2:], mode="bilinear", align_corners=False)
            conf = flat.reshape(*conf.shape[:-3], flat.shape[-3], flat.shape[-2], flat.shape[-1])
        if conf.ndim == reference.ndim - 1:
            conf = conf.unsqueeze(2)
        return conf

    def forward(
        self,
        student_feats: list[torch.Tensor] | tuple[torch.Tensor, ...],
        teacher_feats: list[torch.Tensor] | tuple[torch.Tensor, ...],
        confidence: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if len(student_feats) != len(teacher_feats):
            raise ValueError(f"Expected the same number of feature levels, got {len(student_feats)} and {len(teacher_feats)}")

        total_loss = None
        details: dict[str, float] = {}

        for feat_idx, (student_feat, teacher_feat) in enumerate(zip(student_feats, teacher_feats)):
            student_feat = student_feat.float()
            teacher_feat = teacher_feat.float()
            if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
                teacher_feat = F.interpolate(
                    teacher_feat.reshape(-1, teacher_feat.shape[-3], teacher_feat.shape[-2], teacher_feat.shape[-1]),
                    size=student_feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(*teacher_feat.shape[:-3], teacher_feat.shape[-3], student_feat.shape[-2], student_feat.shape[-1])
            if student_feat.shape != teacher_feat.shape:
                raise ValueError(
                    f"Feature level {feat_idx} shape mismatch after interpolation: {student_feat.shape} vs {teacher_feat.shape}"
                )

            if self.use_conf:
                if confidence is None:
                    raise ValueError("Confidence tensor is required when use_conf=True")
                loss_map = self._elementwise_loss(student_feat, teacher_feat).mean(dim=2, keepdim=True)
                conf = self._resize_confidence(confidence.float(), loss_map).clamp_min(1e-6)
                feature_loss = (loss_map * conf).mean()
            else:
                feature_loss = self.criterion(student_feat, teacher_feat)
                if not isinstance(feature_loss, torch.Tensor):
                    feature_loss = torch.as_tensor(feature_loss, device=student_feat.device, dtype=student_feat.dtype)

            details[f"feat_{feat_idx}"] = float(feature_loss.detach())
            total_loss = feature_loss if total_loss is None else total_loss + feature_loss

        if total_loss is None:
            total_loss = torch.zeros((), device=confidence.device if confidence is not None else "cpu")
        details["total"] = float(total_loss.detach())
        return total_loss, details
