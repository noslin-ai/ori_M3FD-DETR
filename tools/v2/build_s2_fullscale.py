"""Build zero-copy fold5_v2 manifests with split-isolated Ultralytics caches."""
import os, glob, shutil
ROOT='/root/autodl-tmp/aic_race/M3F-DETR'
OUT=f'{ROOT}/data/s2_fullscale'
SRC=f'{ROOT}/data/train'
fold0=set(open(f'{ROOT}/data/folds5_v2/fold0.txt').read().split())
# Rebuild only this derived view; source data remain untouched.
if os.path.lexists(OUT): shutil.rmtree(OUT)
os.makedirs(OUT)
source={}
for ext in ('*.png','*.jpg','*.jpeg'):
    for p in glob.glob(f'{SRC}/visible/{ext}'):
        source[os.path.splitext(os.path.basename(p))[0]]=p
assert len(source)==2000, len(source)
for split,stems in [('train',sorted(s for s in source if s not in fold0)),('val',sorted(fold0))]:
    base=f'{OUT}/{split}'
    os.makedirs(base)
    os.symlink(f'{SRC}/visible',f'{base}/images')
    os.symlink(f'{SRC}/labels',f'{base}/labels')
    # Paths deliberately pass through split/images, producing separate
    # train/labels.cache and val/labels.cache files.
    with open(f'{OUT}/{split}.txt','w') as f:
        for s in stems:
            f.write(f'{base}/images/{os.path.basename(source[s])}\n')
    print(split,len(stems),f'cache={base}/labels.cache')
names=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
with open(f'{OUT}/data.yaml','w') as f:
    f.write(f'path: {OUT}\ntrain: train.txt\nval: val.txt\nnc: 12\nnames:\n')
    for i,n in enumerate(names): f.write(f'  {i}: {n}\n')
