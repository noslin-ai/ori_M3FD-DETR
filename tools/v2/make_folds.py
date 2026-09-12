"""Step 0b — stratified 5-fold split over the 2000 labelled images for out-of-fold evaluation.

Why: the old protocol used a single 200-image val where tricycle had THREE ground truths, so the
local signal was dominated by noise (and anti-correlated with the platform). Out-of-fold
prediction over all 2000 images gives each rare class its full instance count, which is the only
way a local number can be trusted for model-change decisions.

Stratification: greedy — process classes from rarest to most common; for each class, assign the
images containing it (fewest-instance images last) to whichever fold currently has the fewest
instances of that class. Balances rare classes without breaking the common ones.
"""
import os, glob, collections, json

ROOT = "/root/autodl-tmp/aic_race/M3F-DETR"
LBL = f"{ROOT}/data/train/labels"
OUT = "/root/autodl-tmp/aic_race/M3F-DETR/data/folds5_v2"
K = 5
NAMES = ["person","boat","animal","seat","sign","bicycle","car","ball","light","garbage can","uav","tricycle"]

os.makedirs(OUT, exist_ok=True)

img = {}
cls_of = collections.defaultdict(list)
tot = collections.Counter()
for f in sorted(glob.glob(LBL + "/*.txt")):
    stem = os.path.splitext(os.path.basename(f))[0]
    rows = []
    for ln in open(f):
        q = ln.split()
        if len(q) >= 5:
            rows.append(int(float(q[0])))
            tot[int(float(q[0]))] += 1
    img[stem] = collections.Counter(rows)
    for c in set(rows):
        cls_of[c].append(stem)

fold = {s: None for s in img}
fold_cnt = [collections.Counter() for _ in range(K)]
imgs_per_fold = [0] * K

# 从最稀有类开始,按"该类实例数"做贪心均衡
for c in sorted(tot, key=lambda k: tot[k]):
    stems = sorted(cls_of[c], key=lambda s: -img[s][c])
    for s in stems:
        if fold[s] is not None:
            continue
        best = min(range(K), key=lambda k: (fold_cnt[k][c], imgs_per_fold[k]))
        fold[s] = best
        # The image is now irrevocably assigned: update EVERY class it
        # contains.  Updating only class c makes later balancing decisions
        # and the printed per-fold totals silently wrong.
        for cc, n in img[s].items():
            fold_cnt[best][cc] += n
        imgs_per_fold[best] += 1
for s in img:                       # 含稀有类之外的图(纯常见类)均匀填
    if fold[s] is None:
        best = min(range(K), key=lambda k: imgs_per_fold[k])
        fold[s] = best
        imgs_per_fold[best] += 1
        for c, n in img[s].items():
            fold_cnt[best][c] += n

for k in range(K):
    ss = sorted(s for s in img if fold[s] == k)
    with open(f"{OUT}/fold{k}.txt", "w") as fh:
        fh.write("\n".join(ss) + "\n")

print(f"总图数 {len(img)}  每折图数 {imgs_per_fold}")
print(f"\n{'class':<12}{'total':>7}" + "".join(f"{'fold'+str(k):>8}" for k in range(K)) + f"{'max/min':>9}")
for c in range(12):
    vals = [fold_cnt[k][c] for k in range(K)]
    ratio = (max(vals) / min(vals)) if min(vals) > 0 else float("inf")
    print(f"{NAMES[c]:<12}{tot[c]:>7}" + "".join(f"{v:>8}" for v in vals) + f"{ratio:>9.2f}")
json.dump({str(k): {c: fold_cnt[k][c] for c in range(12)} for k in range(K)},
          open(f"{OUT}/counts.json", "w"), indent=1)
print(f"\n写出 {OUT}/fold0..4.txt")
