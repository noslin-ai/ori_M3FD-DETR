import random

import cv2
import numpy as np


class MultiModalTransform:
    """三模态（RGB / IR / Depth）同步数据增强。

    确保三模态在空间上保持一致：同步 resize、同步水平翻转。
    兼容测试集（boxes 可为空 tensor）。
    """

    def __init__(self, size=(384, 640), train=True):
        self.size = size
        self.train = train

    def __call__(self, rgb, ir, depth, boxes):
        h, w = self.size
        old_h, old_w = rgb.shape[:2]

        # ---- 同步 resize ----
        rgb = cv2.resize(rgb, (w, h))
        ir = cv2.resize(ir, (w, h))

        # depth: (3, H, W) -> (H, W, 3) -> resize -> (3, H, W)
        depth = np.transpose(depth, (1, 2, 0))
        depth = cv2.resize(depth, (w, h))
        depth = np.transpose(depth, (2, 0, 1))

        scale_x = w / old_w
        scale_y = h / old_h

        if boxes.shape[0] > 0:
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        # ---- 同步水平翻转 ----
        if self.train:
            if random.random() < 0.5:
                rgb = np.fliplr(rgb)
                ir = np.fliplr(ir)
                depth = np.flip(depth, axis=2)

                if boxes.shape[0] > 0:
                    xmin = boxes[:, 0].clone()
                    xmax = boxes[:, 2].clone()

                    boxes[:, 0] = w - xmax
                    boxes[:, 2] = w - xmin

        return rgb, ir, depth, boxes
