#!/bin/bash
# Tile TTA on compliant best single model (1280refine).
# Rationale: 845/1000 test images are 1920x1080 -> full-image pass at imgsz=1280
# downscales them 0.67x and throws away small-object pixels. 640px tiles upscaled
# to imgsz=1280 give a 2.0x view = 3.0x relative gain. Single model, single weight
# file, multi-view inference -> compliant (no multi-model ensembling).
# The 155 small images (640x360) get no tiles (whole-image window is skipped) -> unaffected.
set -e
cd /root/autodl-tmp/aic_race/M3F-DETR
source /root/miniconda3/etc/profile.d/conda.sh
conda activate race
export OMP_NUM_THREADS=12
W=runs/detect/runs/native_m_trimodal/full2000cont_1280_refine/weights/best.pt

echo "### STAGE 1: smoke test (5 images) ###"
python tools/infer_ultra_tiled.py \
  --weights "$W" --data-root data/test_trimodal_soft \
  --output /tmp/tile_smoke --imgsz 1280 --tile 640 --overlap 0.25 \
  --conf 0.3 --iou 0.6 --fuse-iou 0.55 --max-det 100 --tta --limit 5

echo "### STAGE 2: full run (1000 images, full image + 8 tiles + TTA) ###"
python tools/infer_ultra_tiled.py \
  --weights "$W" --data-root data/test_trimodal_soft \
  --output submission_1280_tile640_raw --zip submission_1280_tile640_raw.zip \
  --imgsz 1280 --tile 640 --overlap 0.25 \
  --conf 0.3 --iou 0.6 --fuse-iou 0.55 --max-det 100 --tta

echo "### STAGE 3: conf sweep ###"
python - <<'PY'
import glob, os, shutil
src = "submission_1280_tile640_raw"
for t in [0.4, 0.45, 0.5, 0.55]:
    tgt = f"submission_1280_tile640_conf{t}"
    os.makedirs(tgt, exist_ok=True)
    n = 0
    for f in glob.glob(src + "/*.txt"):
        stem = os.path.basename(f); keep = []
        for line in open(f):
            p = line.strip().split()
            if len(p) >= 6 and float(p[5]) >= t:
                keep.append(line); n += 1
        open(tgt + "/" + stem, "w").writelines(keep)
    shutil.make_archive(tgt, "zip", root_dir=tgt)
    print(f"tile640 conf{t}: {n} boxes")
PY
echo "### DONE ###"
