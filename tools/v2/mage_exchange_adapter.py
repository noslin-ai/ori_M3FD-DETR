"""MAGE/CSSA-inspired gated exchange on top of the winning IR adapter.

The added P2/P3 refiners are zero initialized at their output projections, so
attaching them preserves every prediction from the source checkpoint exactly.
"""
from __future__ import annotations

import torch
from torch import nn

from distribution_aligned_adapter import DistributionAlignedAdapterModel


class CrossModalExchangeRefiner(nn.Module):
    """Joint channel/spatial gating followed by a lightweight local refiner."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        gate_channels = max(channels // reduction, 16)
        hidden = max(channels // 4, 32)

        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels * 2, gate_channels, 1, bias=True),
            nn.SiLU(inplace=True),
            nn.Conv2d(gate_channels, channels * 2, 1, bias=True),
        )
        self.spatial_gate = nn.Conv2d(4, 2, 7, padding=3, bias=True)
        self.reduce = nn.Conv2d(channels * 2, hidden, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.SiLU(inplace=True),
        )
        self.project = nn.Conv2d(hidden, channels, 1, bias=True)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, anchor: torch.Tensor, auxiliary: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            (anchor.mean((2, 3), keepdim=True), auxiliary.mean((2, 3), keepdim=True)),
            dim=1,
        )
        anchor_channel, auxiliary_channel = self.channel_gate(pooled).chunk(2, dim=1)
        channel_gates = (anchor_channel.sigmoid(), auxiliary_channel.sigmoid())

        spatial_descriptor = torch.cat(
            (
                anchor.mean(1, keepdim=True),
                anchor.amax(1, keepdim=True),
                auxiliary.mean(1, keepdim=True),
                auxiliary.amax(1, keepdim=True),
            ),
            dim=1,
        )
        anchor_spatial, auxiliary_spatial = self.spatial_gate(spatial_descriptor).chunk(2, dim=1)
        exchanged = torch.cat(
            (
                anchor * channel_gates[1] * auxiliary_spatial.sigmoid(),
                auxiliary * channel_gates[0] * anchor_spatial.sigmoid(),
            ),
            dim=1,
        )
        return self.project(self.refine(self.reduce(exchanged)))


class MAGEExchangeAdapterModel(DistributionAlignedAdapterModel):
    """Frozen winning detector plus trainable gated P2/P3 exchange refiners."""

    def attach_exchange_refiners(self, align_weight: float = 0.02):
        if not getattr(self, "use_ir", False) or getattr(self, "use_depth", False):
            raise TypeError("MAGE exchange expects the winning IR-only adapter checkpoint")
        reference = next(self.parameters())
        self.exchange_refiners = nn.ModuleDict(
            {
                "p2": CrossModalExchangeRefiner(256),
                "p3": CrossModalExchangeRefiner(512),
            }
        ).to(device=reference.device, dtype=reference.dtype)
        self.align_weight = float(align_weight)
        self.lock_pretrained_bn = True
        self.criterion = None
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "lock_pretrained_bn", False):
            for module in self.model.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
            self.ir_adapter.eval()
        return self

    def _apply_residual(self, anchor, residual, scale):
        fused = anchor + residual + self.exchange_refiners[scale](anchor, residual)
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


def attach_exchange_refiners(
    model: DistributionAlignedAdapterModel, align_weight: float = 0.02
) -> MAGEExchangeAdapterModel:
    if isinstance(model, MAGEExchangeAdapterModel):
        model.align_weight = float(align_weight)
        return model
    if not isinstance(model, DistributionAlignedAdapterModel):
        raise TypeError("expected a DistributionAlignedAdapterModel checkpoint")
    model.__class__ = MAGEExchangeAdapterModel
    return model.attach_exchange_refiners(align_weight=align_weight)
