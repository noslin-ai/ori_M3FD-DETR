#!/usr/bin/env bash
# =============================================================================
# One-shot submission builder with per-class confidence A/B.
#
# For a given trained model, this:
#   1. [GPU, one pass] scans the fold1-val split to find each class's best
#      confidence threshold (mAP@50-95 equal-weight per class).
#   2. [GPU]  runs full-image TTA inference on the TEST images at conf=0.001
#      (baseline)  -> submission_<tag>_ab00_baseline.zip
#   3. [GPU]  re-runs the SAME inference but applies the per-class thresholds
#              -> submission_<tag>_ab01_perclass.zip
# Both zips are produced so you can A/B them on the platform.
#
# It never reads test labels, never writes train data, never pseudo-labels.
#
# Usage (example, soft-fusion model on soft test images):
#   bash scripts_5090/run_submission_ab.sh \
#     --weights runs/detect/runs/native_m_trimodal/soft768_labelrefresh_from_sar_best/weights/best.pt \
#     --val-images data/yolo_trimodal_soft_m/val/images \
#     --val-labels  data/yolo_trimodal_soft_m/val/labels \
#     --test-root   data/test_trimodal_soft \
#     --tag soft_labelrefresh_fulltta \
#     --imgsz 768 --tta
# =============================================================================
set -euo pipefail

PY=/root/miniconda3/envs/race/bin/python
ROOT=/root/autodl-tmp/aic_race/M3F-DETR
cd "$ROOT"

WEIGHTS=""; VALIMG=""; VALLBL=""; TESTROOT=""; TAG="scan"
IMGSZ=768; TILE=0; OVERLAP=0.25; CONF=0.001; FUSE=0.55; MAXDET=100; BATCH=8; DEVICE=cuda
TTA=""
SCAN_ARGS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2;;
    --val-images) VALIMG="$2"; shift 2;;
    --val-labels) VALLBL="$2"; shift 2;;
    --test-root) TESTROOT="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --imgsz) IMGSZ="$2"; shift 2;;
    --tile) TILE="$2"; shift 2;;
    --conf) CONF="$2"; shift 2;;
    --batch) BATCH="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --tta) TTA="--tta"; shift;;
    *) echo "unknown arg: $1"; exit 1;;
  esac
done

for v in WEIGHTS VALIMG VALLBL TESTROOT; do
  [ -n "${!v}" ] || { echo "missing --$v"; exit 1; }
done

[ -e /dev/nvidiactl ] || { echo "GPU not mounted"; exit 1; }

echo "================================================================"
echo " submission builder: tag=$TAG  weights=$WEIGHTS"
date
echo "================================================================"

CACHE="$ROOT/.scan_${TAG}.npz"
JSON="$ROOT/.scan_${TAG}.json"

echo ">> [1/3] per-class confidence scan on val ..."
"$PY" tools/scan_conf_per_class.py \
    --weights "$WEIGHTS" --val-images "$VALIMG" --val-labels "$VALLBL" \
    --cache "$CACHE" --imgsz "$IMGSZ" --batch "$BATCH" --device "$DEVICE" \
    $TTA --out-json "$JSON"

echo ">> [2/3] baseline submission (global conf=$CONF) ..."
"$PY" tools/infer_ultra_tiled.py \
    --weights "$WEIGHTS" --data-root "$TESTROOT" \
    --output "submission_${TAG}_ab00_baseline" \
    --zip "submission_${TAG}_ab00_baseline.zip" \
    --imgsz "$IMGSZ" --tile "$TILE" --overlap "$OVERLAP" \
    --conf "$CONF" --iou 0.6 --fuse-iou "$FUSE" --max-det "$MAXDET" \
    --batch "$BATCH" --device "$DEVICE" $TTA

echo ">> [3/3] per-class threshold submission ..."
"$PY" tools/infer_ultra_tiled.py \
    --weights "$WEIGHTS" --data-root "$TESTROOT" \
    --output "submission_${TAG}_ab01_perclass" \
    --zip "submission_${TAG}_ab01_perclass.zip" \
    --imgsz "$IMGSZ" --tile "$TILE" --overlap "$OVERLAP" \
    --conf "$CONF" --perclass-conf "$JSON" --iou 0.6 --fuse-iou "$FUSE" --max-det "$MAXDET" \
    --batch "$BATCH" --device "$DEVICE" $TTA

echo
echo "Done. A/B candidates:"
echo "  submission_${TAG}_ab00_baseline.zip"
echo "  submission_${TAG}_ab01_perclass.zip"
