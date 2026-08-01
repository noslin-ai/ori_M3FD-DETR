"""随机种子 — 保证实验可复现。"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """设置所有随机种子。

    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 性能优化（RTX 5090 Blackwell SM 120）
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # TF32 加速矩阵乘法（Blackwell 原生 8x matmul 吞吐）
    if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    print(f"  Random seed set to: {seed}")


def worker_init_fn(worker_id):
    """DataLoader worker 种子初始化。

    用法:
        DataLoader(..., worker_init_fn=worker_init_fn)
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
