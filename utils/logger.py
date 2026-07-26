"""Logger — 日志系统。

支持:
    1. 终端输出
    2. 文件日志
    3. TensorBoard (可选)
"""

import os
import time
from collections import defaultdict


class AverageMeter:
    """滑动平均计算器。"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """训练日志记录器。

    Args:
        log_dir: 日志目录
        use_tensorboard: 是否使用 TensorBoard
    """

    def __init__(self, log_dir="logs", use_tensorboard=False):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.meters = defaultdict(AverageMeter)

        os.makedirs(log_dir, exist_ok=True)

        # 文件日志
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"train_{timestamp}.log")

        # TensorBoard
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = os.path.join(log_dir, "tensorboard")
                self.writer = SummaryWriter(tb_dir)
            except ImportError:
                print("  ⚠ TensorBoard 未安装，跳过")
                self.use_tensorboard = False

    def update(self, name, value, n=1):
        """更新指标。"""
        self.meters[name].update(value, n)

    def log(self, epoch, step, log_dict=None, prefix="train"):
        """记录日志。

        Args:
            epoch: 当前 epoch
            step: 当前 step
            log_dict: 额外日志
            prefix: 前缀
        """
        # 格式化输出
        parts = [f"[{prefix}] e{epoch} s{step}"]
        for name, meter in self.meters.items():
            parts.append(f"{name}={meter.avg:.4f}")

        if log_dict:
            for k, v in log_dict.items():
                parts.append(f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}")

        msg = " ".join(parts)
        print(msg)

        # 写入文件
        with open(self.log_file, "a") as f:
            f.write(msg + "\n")

        # TensorBoard
        if self.writer and self.use_tensorboard:
            for name, meter in self.meters.items():
                self.writer.add_scalar(f"{prefix}/{name}", meter.avg, step)

    def reset(self):
        """重置所有 meter。"""
        for meter in self.meters.values():
            meter.reset()

    def add_scalar(self, tag, value, step):
        """写入 TensorBoard 标量。"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def close(self):
        """关闭 logger。"""
        if self.writer:
            self.writer.close()
