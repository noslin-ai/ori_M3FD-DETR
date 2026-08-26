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

        # 归一化坐标 -> 像素坐标
        cx *= img_width
        cy *= img_height
        w *= img_width
        h *= img_height

        xmin = cx - w / 2
        ymin = cy - h / 2
        xmax = cx + w / 2
        ymax = cy + h / 2

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(cls)

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    return boxes, labels
