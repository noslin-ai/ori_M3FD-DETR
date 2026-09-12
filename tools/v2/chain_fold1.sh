#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate race
cd /root/autodl-tmp/aic_race/M3F-DETR
mkdir -p runs/detect/runs/s4/logs
python tools/v2/build_fold_view.py --fold 1
DATA="$PWD/data/v2_fold1/data.yaml"
COMMON="epochs=40 cache=ram device=0 workers=12 project=runs/s4 optimizer=AdamW lr0=0.001 lrf=0.05 cos_lr=True warmup_epochs=3.0 momentum=0.937 weight_decay=0.0005 seed=42 deterministic=True patience=999 amp=True hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 degrees=0.0 translate=0.1 scale=0.15 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 bgr=0.0 mosaic=0.0 mixup=0.0 copy_paste=0.0 auto_augment=randaugment erasing=0.2 val=True plots=False exist_ok=False"
yolo detect train model=yolo11m.pt data="$DATA" name=fold1_rgb1280 imgsz=1280 batch=8 $COMMON > runs/detect/runs/s4/logs/fold1_rgb1280.log 2>&1
test "$(wc -l < runs/detect/runs/s4/fold1_rgb1280/results.csv)" -eq 41
python -u tools/v2/train_ir_adapter.py --base "$PWD/runs/detect/runs/s4/fold1_rgb1280/weights/best.pt" --data "$DATA" --project runs/s4 --name fold1_ir_adapter --epochs 20 --batch 8 > runs/detect/runs/s4/logs/fold1_ir_adapter.log 2>&1
test "$(wc -l < runs/detect/runs/s4/fold1_ir_adapter/results.csv)" -eq 21
python -u tools/v2/train_trimodal_adapter.py --base "$PWD/runs/detect/runs/s4/fold1_ir_adapter/weights/best.pt" --data "$DATA" --project runs/s4 --name fold1_trimodal_adapter --epochs 20 --batch 8 > runs/detect/runs/s4/logs/fold1_trimodal_adapter.log 2>&1
test "$(wc -l < runs/detect/runs/s4/fold1_trimodal_adapter/results.csv)" -eq 21
echo "FOLD1_CHAIN_DONE $(date -Is)"
