#!/usr/bin/env bash
# 5090 全开训练 Stage 1: 冻结前 10 层，只训 neck+head（保护预训练特征）
set -e
cd /root/autodl-tmp/aic_race/M3F-DETR
source /root/miniconda3/etc/profile.d/conda.sh
conda activate race
export OMP_NUM_THREADS=16

/root/miniconda3/envs/race/bin/yolo detect train \
  model=yolo11m.pt \
  data=data/yolo_sar_v3/data.yaml \
  imgsz=1024 \
  batch=20 \
  epochs=60 \
  freeze=[0,1,2,3,4,5,6,7,8,9] \
  optimizer=AdamW \
  lr0=0.001 \
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
  name=stage1_frozen \
  > /tmp/stage1_frozen.log 2>&1
echo "STAGE1 DONE exit=$?"
