import cv2
import numpy as np

# 深度传感器有效范围（毫米）— 赛题规定: 30cm~20m (300~19999mm)
DEPTH_MIN = 300
DEPTH_MAX = 19999


def process_depth(depth_path):
    """处理深度图，兼容 uint16 PNG 和 uint8 JPG 两种格式。

    PNG (16bit):
        原始深度值，单位毫米，0 为无效。
        处理: 裁剪到 300~20000mm → 归一化 → 计算梯度。

    JPG (8bit):
        有损压缩后的深度可视化图，值域 0~255。
        0 为无效，1~255 线性映射到 300~20000mm。
        精度有所损失但几何结构仍可保留。

    Returns:
        depth3: np.ndarray, shape=(3, H, W), dtype=float32
            channel 0: 归一化深度
            channel 1: x 方向 Sobel 梯度
            channel 2: y 方向 Sobel 梯度
    """
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise FileNotFoundError(f"无法读取深度图: {depth_path}")

    # JPG 图像会被读取为 3 通道 (H, W, 3)，取第一通道并转为灰度
    if depth.ndim == 3:
        depth = depth[:, :, 0]  # JPG 三通道取第一个

    depth = depth.astype(np.float32)

    # 无效区域掩码
    mask = depth <= 0

    # ---- uint16 PNG：原始深度值（毫米）----
    if depth.max() > 255:
        # 裁剪到传感器有效范围
        depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX)
        # 归一化到 [0, 1]
        depth = (depth - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
    # ---- uint8 JPG：有损可视化深度 ----
    else:
        # 值域 1~255 → 映射到 300~20000
        valid = depth > 0
        if valid.any():
            depth[valid] = (depth[valid] - 1) / 254.0  # 归一化到 [0, 1]

    # 恢复无效区域
    depth[mask] = 0

    # ---- 梯度几何信息 ----
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)

    depth3 = np.stack([depth, gx, gy], axis=0)

    return depth3
