#!/usr/bin/env bash
# =============================================================================
# One-shot parallel launcher for the next two YOLO11m training tracks
# (GPU must be attached = AutoDL 开卡 / paid GPU mode).
#
# Track A (gated):  data already generated (data/yolo_trimodal_gated_m,
#                   test_trimodal_gated) -- config from CHANGELOG v0.17.1.
# Track B (rareos): clean rare-class oversampling of the soft-fusion data.
#
# Both start from the SAME platform-best SAR weight and share the identical
# optimizer/hyper-params, so together they give a clean A/B of the fusion
# strategy (gated local-injection) vs. data-balancing (rare-class dup).
#
# Pre-flight:
#   - GPU mounted (nvidia visible) & enough free disk (checked here).
#   - If free disk < NEEDED, this script STOPS and tells you what to delete;
#     it will NOT delete anything by itself.
# =============================================================================
set -euo pipefail

RACE=/root/miniconda3/envs/race/bin/python
CONDA=/root/miniconda3/etc/profile.d/conda.sh
ROOT=/root/autodl-tmp/aic_race/M3F-DETR
cd "$ROOT"

export OMP_NUM_THREADS=8

# ---- config ----
GATED_CFG=configs/yolo_native_m_trimodal_gated_labelrefresh.yaml
RAREOS_CFG=configs/yolo_native_m_trimodal_rareos_labelrefresh.yaml
GATED_DATA=data/yolo_trimodal_gated_m
RAREOS_DATA=data/yolo_trimodal_soft_m_rareos
GATED_RUN=runs/detect/runs/native_m_trimodal/gated768_labelrefresh_from_sar_best
RAREOS_RUN=runs/detect/runs/native_m_trimodal/rareos768_labelrefresh_from_sar_best

echo "================================================================"
echo " parallel YOLO11m launcher  (gated + rareos)"
date
echo "================================================================"

# ---- 0. sanity: GPU attached ----
if [ ! -e /dev/nvidiactl ]; then
  echo "ERROR: GPU not mounted (/dev/nvidiactl missing). Open-card first."
  echo "  AutoDL: 实例 -> 开机(计费) 以挂载 GPU。"
  exit 1
fi
"$RACE" -c 'import torch;assert torch.cuda.is_available();print("GPU ok:",torch.cuda.device_count())'

# ---- 1. disk check ----
FREE_KB=$(df --output=avail -k "$ROOT" | tail -1)
FREE_GB=$(( FREE_KB / 1024 / 1024 ))
echo "free disk: ${FREE_GB}G on $(df -P "$ROOT" | tail -1 | awk '{print $1}')"

# Rough worst-case: each run saves best/last + epoch*.pt every save_period=5.
# YOLO11m best.pt ~40MB, full epoch pt ~161MB. With save_period=5 over 80 ep:
#   ~16 intermediate + best/last  =>  up to ~2.7G per run if NOT cleaned.
NEEDED=8   # two runs + slack
if [ "$FREE_GB" -lt "$NEEDED" ]; then
  echo
  echo "!! Free disk ${FREE_GB}G < ${NEEDED}G needed for two parallel runs."
  echo "!! Delete intermediate epoch weights first, e.g.:"
  echo "    # native_x_sar YOLO11x (each 435MB) -- keep only best/last:"
  echo "    find runs/detect/runs/native_x_sar -name 'epoch*.pt' -delete"
  echo "    # native_m_trimodal old fusion/soft runs -- keep soft_labelrefresh best:"
  echo "    find runs/detect/runs/native_m_trimodal/fusion768_from_sar_best -name 'epoch*.pt' -delete"
  echo "    find runs/detect/runs/native_m_trimodal/soft768_from_sar_best -name 'epoch*.pt' -delete"
  echo "  Then re-run this script. Nothing was deleted."
  exit 2
fi

# ---- 2. optional data prep for rareos (Track B) if missing ----
if [ ! -f "$RAREOS_DATA/data.yaml" ]; then
  echo ">> preparing rare-class oversampled data ..."
  "$RACE" tools/oversample_rare_clean.py \
      --src data/yolo_trimodal_soft_m \
      --dst "$RAREOS_DATA" \
      --mult 1:2,7:2,11:4
  find "$RAREOS_DATA" -name "*.cache" -delete
else
  echo ">> rareos data already present: $RAREOS_DATA"
fi

# ---- 3. clear stale caches (labels were refreshed 2026-09-02) ----
find "$GATED_DATA" "$RAREOS_DATA" -name "*.cache" -delete

# ---- 4. launch both in parallel screens ----
echo ">> launching gated  (device 0, screen yolo_m_gated_v0171) ..."
screen -dmS yolo_m_gated_v0171 bash -lc \
  "source $CONDA && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=$GATED_CFG device=0 > yolo_m_trimodal_gated_labelrefresh_v0171_train.log 2>&1"

echo ">> launching rareos (device 1, screen yolo_m_rareos_v018) ..."
screen -dmS yolo_m_rareos_v018 bash -lc \
  "source $CONDA && conda activate race && OMP_NUM_THREADS=8 yolo detect train cfg=$RAREOS_CFG device=1 > yolo_m_trimodal_rareos_labelrefresh_v018_train.log 2>&1"

sleep 5
echo ">> screens now:"
screen -ls || true

# Verify both actually started (weights files begin to appear):
sleep 20
for RUN in "$GATED_RUN" "$RAREOS_RUN"; do
  if [ -d "$RUN" ] && ls "$RUN"/weights/*.pt >/dev/null 2>&1; then
    echo "OK  $RUN is training"
  else
    echo "WARN $RUN no weights yet -- check its log"
  fi
done

echo
echo "monitor:  tail -f yolo_m_trimodal_gated_labelrefresh_v0171_train.log"
echo "          tail -f yolo_m_trimodal_rareos_labelrefresh_v018_train.log"
