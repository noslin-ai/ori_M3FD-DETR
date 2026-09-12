#!/bin/bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate race
cd /root/autodl-tmp/aic_race/M3F-DETR
SUM=/tmp/s2_fullscale_summary.txt
: > "$SUM"
COMMON="epochs=40 cache=ram device=0 workers=12 project=runs/s2 optimizer=AdamW lr0=0.001 lrf=0.05 cos_lr=True warmup_epochs=3.0 momentum=0.937 weight_decay=0.0005 seed=42 deterministic=True patience=999 amp=True hsv_h=0.015 hsv_s=0.4 hsv_v=0.3 degrees=0.0 translate=0.1 scale=0.15 shear=0.0 perspective=0.0 flipud=0.0 fliplr=0.5 bgr=0.0 mosaic=0.0 mixup=0.0 copy_paste=0.0 auto_augment=randaugment erasing=0.2 val=True plots=False exist_ok=False"
run () {
  local name=$1 imgsz=$2 batch=$3
  echo "=== $name start $(date) ===" | tee -a "$SUM"
  yolo detect train model=yolo11m.pt data=data/s2_fullscale/data.yaml name="$name" imgsz="$imgsz" batch="$batch" $COMMON > "/tmp/${name}.log" 2>&1
  local csv="runs/detect/runs/s2/$name/results.csv"
  test "$(wc -l < "$csv")" -eq 41
  stat -c '%y %s %n' "runs/detect/runs/s2/$name/weights/best.pt" | tee -a "$SUM"
  tr '\r' '\n' < "/tmp/${name}.log" | grep -aE "all +[0-9]+ +[0-9]+" | tail -1 | tee -a "$SUM"
}
run full1280_v2 1280 8
run full1920_v2 1920 4
echo "TRAINING_DONE $(date)" | tee -a "$SUM"
