"""Lightweight residual IR/depth adapters for a loaded YOLO11 DetectionModel.

The RGB checkpoint object is preserved verbatim. Auxiliary branches inject zero-initialized
residuals after backbone layers 2 (P2, stride 4) and 4 (P3, stride 8).
"""
from __future__ import annotations

import torch
from torch import nn
from ultralytics.nn.tasks import DetectionModel


class AuxBranch(nn.Module):
    def __init__(self, cin: int):
        super().__init__()
        self.s2 = self._block(cin, 32)
        self.s4 = self._block(32, 64)
        self.s8 = self._block(64, 128)

    @staticmethod
    def _block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, 2, 1, bias=False),
            nn.BatchNorm2d(cout),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x = self.s2(x)
        p2 = self.s4(x)
        p3 = self.s8(p2)
        return p2, p3


def _zero_conv(cin: int, cout: int) -> nn.Conv2d:
    m = nn.Conv2d(cin, cout, 1, bias=True)
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class AdapterDetectionModel(DetectionModel):
    """DetectionModel that accepts RGB3+IR1+log-depth1+valid1 tensors."""

    def attach_adapters(self, use_ir=True, use_depth=False):
        self.use_ir = bool(use_ir)
        self.use_depth = bool(use_depth)
        if self.use_ir:
            self.ir_adapter = AuxBranch(1)
            self.ir_inject_p2 = _zero_conv(64, 256)
            self.ir_inject_p3 = _zero_conv(128, 512)
        if self.use_depth:
            self.depth_adapter = AuxBranch(2)
            self.depth_inject_p2 = _zero_conv(64, 256)
            self.depth_inject_p3 = _zero_conv(128, 512)
        self.criterion = None
        return self

    def _aux_residuals(self, x):
        d2 = d3 = None
        if self.use_ir:
            p2, p3 = self.ir_adapter(x[:, 3:4])
            d2 = self.ir_inject_p2(p2)
            d3 = self.ir_inject_p3(p3)
        if self.use_depth:
            p2, p3 = self.depth_adapter(x[:, 4:6])
            z2 = self.depth_inject_p2(p2)
            z3 = self.depth_inject_p3(p3)
            d2 = z2 if d2 is None else d2 + z2
            d3 = z3 if d3 is None else d3 + z3
        return d2, d3

    def _predict_once(self, x, profile=False, embed=None):
        if x.shape[1] == 3:  # stride/AMP probes and RGB-only smoke tests
            return super()._predict_once(x, profile, embed)
        if x.shape[1] != 6:
            raise ValueError(f"Adapter model expects 3 or 6 channels, got {x.shape[1]}")
        d2, d3 = self._aux_residuals(x)
        x = x[:, :3]
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            if m.i == 2 and d2 is not None:
                if x.shape != d2.shape:
                    raise RuntimeError(f"P2 mismatch: rgb={x.shape}, aux={d2.shape}")
                x = x + d2
            elif m.i == 4 and d3 is not None:
                if x.shape != d3.shape:
                    raise RuntimeError(f"P3 mismatch: rgb={x.shape}, aux={d3.shape}")
                x = x + d3
            y.append(x if m.i in self.save else None)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def loss(self, batch, preds=None):
        if getattr(self, "criterion", None) is None:
            self.criterion = self.init_criterion()
        if preds is None:
            preds = self.predict(batch["img"])
        return self.criterion(preds, batch)


def attach_adapters(model: DetectionModel, use_ir=True, use_depth=False) -> AdapterDetectionModel:
    if isinstance(model, AdapterDetectionModel):
        return model
    model.__class__ = AdapterDetectionModel
    return model.attach_adapters(use_ir=use_ir, use_depth=use_depth)
