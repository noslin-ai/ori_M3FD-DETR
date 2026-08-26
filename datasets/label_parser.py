import torch


def load_yolo_label(label_path, img_width, img_height):
    """加载YOLO格式标签，转换为DETR所需的像素坐标框。

    官方标签格式（归一化）:
        class_id center_x center_y width height

    转换后（像素坐标）:
        xmin ymin xmax ymax
    """
    boxes = []
    labels = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if line.strip() == "":
            continue

        data = line.strip().split()

        cls = int(data[0])
        cx = float(data[1])
        cy = float(data[2])
        w = float(data[3])
        h = float(data[4])

        # 少量官方标注会略微越界，训练前裁剪到合法图像范围。
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        w = min(max(w, 1e-6), 1.0)
        h = min(max(h, 1e-6), 1.0)

        # 归一化坐标 -> 像素坐标
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height

        xmin = max(cx - w / 2, 0.0)
        ymin = max(cy - h / 2, 0.0)
        xmax = min(cx + w / 2, float(img_width))
        ymax = min(cy + h / 2, float(img_height))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    return boxes, labels
