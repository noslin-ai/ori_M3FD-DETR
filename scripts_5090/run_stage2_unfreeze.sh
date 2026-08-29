#!/usr/bin/env bash
# 5090 全开训练 Stage 2: 从 stage1 last.pt 续跑，解冻全部，低 lr 精修
set -e
cd /root/autodl-tmp/aic_race/M3F-DETR
source /root/miniconda3/etc/profile.d/conda.sh
conda activate race
export OMP_NUM_THREADS=16

C1=runs/detect/runs/yolo11m_1024/stage1_frozen
/root/miniconda3/envs/race/bin/yolo detect train \
  model=$C1/weights/last.pt \
  data=data/yolo_sar_v3/data.yaml \
  imgsz=1024 \
  batch=20 \
  epochs=120 \
  optimizer=AdamW \
  lr0=0.0005 \
  weight_decay=0.0005 \
  cos_lr=True \
  close_mosaic=10 \
  mixup=0.15 \
  copy_paste=0.1 \
  cache=ram \
  workers=16 \
  amp=True \
  deterministic=False \
  patience=50 \
  seed=42 \
  project=runs/yolo11m_1024 \
  name=stage2_unfreeze \
  > /root/autodl-tmp/aic_race/M3F-DETR/logs/stage2_unfreeze.log 2>&1
echo "STAGE2 DONE exit=$?"
