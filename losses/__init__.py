"""M3F-DETR 损失函数 — 统一从 models.losses 重导出，避免重复定义。"""

from models.losses import FocalLoss, DINOLoss
from models.losses.box_loss import box_l1_loss, giou_loss

__all__ = ["FocalLoss", "box_l1_loss", "giou_loss", "DINOLoss"]
