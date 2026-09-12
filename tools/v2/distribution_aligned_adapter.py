"""Distribution-anchored RGB/IR/depth adapter for a frozen YOLO detector.

The detector sees the exact original RGB/soft-fusion path at initialization. During
adapter training, fused P2/P3 channel moments are softly anchored to the detector's
pre-injection features to limit feature drift.
"""
from __future__ import annotations

import torch

from adapter_model import AdapterDetectionModel


class DistributionAlignedAdapterModel(AdapterDetectionModel):
    def attach_distribution_aligned_adapters(self, align_weight=0.05):
        super().attach_adapters(use_ir=True, use_depth=True)
        self.align_weight = float(align_weight)
        self._alignment_terms = []
        return self

    def _begin_residual_pass(self):
        self._alignment_terms = []

    def _apply_residual(self, anchor, residual, scale):
        fused = anchor + residual
        if self.training and torch.is_grad_enabled():
            dims = (2, 3)
            anchor_mean = anchor.detach().mean(dims)
            anchor_std = anchor.detach().var(dims, unbiased=False).add(1e-6).sqrt()
            fused_mean = fused.mean(dims)
            fused_std = fused.var(dims, unbiased=False).add(1e-6).sqrt()
            mean_shift = ((fused_mean - anchor_mean) / anchor_std).abs().mean()
            std_shift = (torch.log(fused_std) - torch.log(anchor_std)).abs().mean()
            self._alignment_terms.append(0.5 * (mean_shift + std_shift))
        return fused

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(batch["img"])
        loss, loss_items = self.criterion(preds, batch)
        if self._alignment_terms and self.align_weight > 0:
            alignment = torch.stack(self._alignment_terms).mean()
            loss = loss + alignment * self.align_weight * batch["img"].shape[0]
        return loss, loss_items


def attach_distribution_aligned_adapters(
    model: AdapterDetectionModel, align_weight=0.05
) -> DistributionAlignedAdapterModel:
    if isinstance(model, DistributionAlignedAdapterModel):
        model.align_weight = float(align_weight)
        return model
    model.__class__ = DistributionAlignedAdapterModel
    return model.attach_distribution_aligned_adapters(align_weight=align_weight)
