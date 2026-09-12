"""Ultralytics-compatible online RGB/IR/depth dataset and trainer.

No site-packages modifications and no materialized six-channel images. RGB photometric
augmentations stay RGB-only; affine/flip/letterbox parameters are shared with auxiliary maps.
"""
from __future__ import annotations

import glob
import os
from copy import copy

import cv2
import numpy as np
import torch

from ultralytics.data.augment import Compose, Format, LetterBox, RandomPerspective
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import colorstr
from ultralytics.data.utils import get_hash
from ultralytics.utils.torch_utils import unwrap_model

from adapter_model import AdapterDetectionModel


class TriModalRandomPerspective(RandomPerspective):
    def apply_depth(self, labels, params=None):
        aux = labels.get("depth")
        if aux is None:
            return labels
        M, size = params["M"], params["size"]
        if (size[0] != aux.shape[1] or size[1] != aux.shape[0]) or (M != np.eye(3)).any():
            warp = cv2.warpPerspective if self.perspective else cv2.warpAffine
            matrix = M if self.perspective else M[:2]
            ir = warp(aux[..., 0], matrix, dsize=size, flags=cv2.INTER_LINEAR, borderValue=0)
            dep = warp(aux[..., 1], matrix, dsize=size, flags=cv2.INTER_NEAREST, borderValue=0)
            valid = warp(aux[..., 2], matrix, dsize=size, flags=cv2.INTER_NEAREST, borderValue=0)
            aux = np.stack((ir, dep, valid), axis=-1)
        labels["depth"] = np.ascontiguousarray(aux)
        return labels


class TriModalLetterBox(LetterBox):
    def __call__(self, labels=None, image=None):
        labels = {} if labels is None else labels
        return_image_only = len(labels) == 0
        if image is not None:
            labels["img"] = image
        params = self.get_params(labels)
        labels = self.apply_image(labels, params)
        if not return_image_only:
            labels = self.apply_instances(labels, params)
        labels = self.apply_semantic(labels, params)
        labels = self.apply_depth(labels, params)
        return labels["img"] if return_image_only else labels

    def apply_depth(self, labels, params=None):
        aux = labels.get("depth")
        if aux is None:
            return labels
        new_unpad = params["new_unpad"]
        if aux.shape[:2][::-1] != new_unpad:
            ir = cv2.resize(aux[..., 0], new_unpad, interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(aux[..., 1], new_unpad, interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(aux[..., 2], new_unpad, interpolation=cv2.INTER_NEAREST)
            aux = np.stack((ir, dep, valid), axis=-1)
        t, b, l, r = params["top"], params["bottom"], params["left"], params["right"]
        aux = cv2.copyMakeBorder(aux, t, b, l, r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        labels["depth"] = np.ascontiguousarray(aux)
        return labels


class TriModalFormat(Format):
    def apply_depth(self, labels, params=None):
        aux = labels.pop("depth", None)
        if aux is None:
            raise RuntimeError("paired auxiliary tensor missing before Format")
        aux = torch.from_numpy(np.ascontiguousarray(aux.transpose(2, 0, 1))).float()
        rgb = labels["img"].float()
        if rgb.shape[1:] != aux.shape[1:]:
            raise RuntimeError(f"RGB/aux transform mismatch: {rgb.shape} vs {aux.shape}")
        labels["img"] = torch.cat((rgb, aux), dim=0)
        return labels


def _clone_transform(t):
    if isinstance(t, Compose):
        return Compose([_clone_transform(x) for x in t.transforms])
    if type(t) is RandomPerspective:
        return TriModalRandomPerspective(
            degrees=t.degrees, translate=t.translate, scale=t.scale, shear=t.shear,
            perspective=t.perspective, size=t.size, preserve_obb=t.preserve_obb,
        )
    if type(t) is LetterBox:
        return TriModalLetterBox(
            new_shape=t.new_shape, auto=t.auto, scale_fill=t.scale_fill, scaleup=t.scaleup,
            center=t.center, stride=t.stride, padding_value=t.padding_value, interpolation=t.interpolation,
        )
    if type(t) is Format:
        return TriModalFormat(
            bbox_format=t.bbox_format, normalize=t.normalize, return_mask=t.return_mask,
            return_keypoint=t.return_keypoint, return_obb=t.return_obb, mask_ratio=t.mask_ratio,
            mask_overlap=t.mask_overlap, batch_idx=t.batch_idx, bgr=t.bgr,
        )
    return t


class TriModalYOLODataset(YOLODataset):
    def __init__(self, *args, data=None, **kwargs):
        self.modal_root = (data or {}).get("modal_root", "/root/autodl-tmp/aic_race/M3F-DETR/data/train")
        self._ir_map = self._files(f"{self.modal_root}/infrared")
        self._depth_map = self._files(f"{self.modal_root}/depth")
        self._label_map = self._files(f"{self.modal_root}/labels", exts=("*.txt",))
        super().__init__(*args, data=data, **kwargs)

    @staticmethod
    def _files(path, exts=("*.png", "*.jpg", "*.jpeg")):
        out = {}
        for ext in exts:
            for p in glob.glob(os.path.join(path, ext)):
                stem = os.path.splitext(os.path.basename(p))[0]
                if stem in out:
                    raise RuntimeError(f"duplicate stem {stem} in {path}")
                out[stem] = p
        return out

    def get_label_files(self):
        stems = [os.path.splitext(os.path.basename(p))[0] for p in self.im_files]
        missing = [s for s in stems if s not in self._label_map or s not in self._ir_map or s not in self._depth_map]
        if missing:
            raise FileNotFoundError(f"missing paired files for {len(missing)} stems, first={missing[:5]}")
        self.label_files = [self._label_map[s] for s in stems]
        self.ir_files = [self._ir_map[s] for s in stems]
        self.depth_files = [self._depth_map[s] for s in stems]
        return self.label_files

    def get_cache_hash(self):
        return get_hash(self.label_files + self.im_files + self.ir_files + self.depth_files + [__file__])

    def get_image_and_label(self, index):
        labels = super().get_image_and_label(index)
        stem = os.path.splitext(os.path.basename(labels["im_file"]))[0]
        ir = cv2.imread(self._ir_map[stem], cv2.IMREAD_UNCHANGED)
        dep = cv2.imread(self._depth_map[stem], cv2.IMREAD_UNCHANGED)
        if ir is None or dep is None:
            raise FileNotFoundError(f"failed reading pair for {stem}")
        if ir.ndim == 3:
            ir = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)
        if dep.ndim == 3:
            dep = dep[..., 0]
        if ir.shape != tuple(labels["ori_shape"]) or dep.shape != tuple(labels["ori_shape"]):
            raise RuntimeError(f"shape mismatch {stem}: rgb={labels['ori_shape']} ir={ir.shape} depth={dep.shape}")
        rh, rw = labels["resized_shape"]
        if ir.shape != (rh, rw):
            ir = cv2.resize(ir, (rw, rh), interpolation=cv2.INTER_LINEAR)
            dep = cv2.resize(dep, (rw, rh), interpolation=cv2.INTER_NEAREST)
        if dep.dtype == np.uint16:
            valid = dep > 0
            d = np.zeros(dep.shape, np.float32)
            d[valid] = np.log1p(np.clip(dep[valid].astype(np.float32), 0, 20000)) / np.log1p(20000.0)
            d *= 255.0
            v = valid.astype(np.float32) * 255.0
        else:
            # JPEG depth has lost metric calibration; preserve normalized value plus support mask.
            valid = dep > 0
            d = dep.astype(np.float32)
            v = valid.astype(np.float32) * 255.0
        labels["depth"] = np.stack((ir.astype(np.float32), d, v), axis=-1)
        return labels

    def build_transforms(self, hyp=None):
        hyp.mosaic = hyp.mixup = hyp.cutmix = hyp.copy_paste = 0.0
        return _clone_transform(super().build_transforms(hyp))


class TriModalDetectionTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode="train", batch=None):
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return TriModalYOLODataset(
            img_path=img_path, imgsz=self.args.imgsz, batch_size=batch, augment=mode == "train",
            hyp=copy(self.args), rect=self.args.rect or mode == "val", cache=None,
            single_cls=self.args.single_cls or False, stride=gs, pad=0.0 if mode == "train" else 0.5,
            prefix=colorstr(f"{mode}: "), task=self.args.task, classes=self.args.classes,
            data=self.data, fraction=self.args.fraction,
        )

    def get_model(self, cfg=None, weights=None, verbose=True):
        if isinstance(weights, AdapterDetectionModel):
            return weights
        return super().get_model(cfg=cfg, weights=weights, verbose=verbose)
