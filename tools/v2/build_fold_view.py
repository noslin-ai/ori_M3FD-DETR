"""Build zero-copy train/val manifests for any folds5_v2 fold."""
import argparse,glob,os,shutil
R='/root/autodl-tmp/aic_race/M3F-DETR'; SRC=f'{R}/data/train'
p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);a=p.parse_args();assert 0<=a.fold<5
out=f'{R}/data/v2_fold{a.fold}'; val=set(open(f'{R}/data/folds5_v2/fold{a.fold}.txt').read().split())
if os.path.lexists(out):shutil.rmtree(out)
os.makedirs(out);src={}
for e in ('*.png','*.jpg','*.jpeg'):
 for z in glob.glob(f'{SRC}/visible/{e}'):src[os.path.splitext(os.path.basename(z))[0]]=z
assert len(src)==2000 and val<=set(src)
for split,stems in [('train',sorted(set(src)-val)),('val',sorted(val))]:
 b=f'{out}/{split}';os.makedirs(b);os.symlink(f'{SRC}/visible',f'{b}/images');os.symlink(f'{SRC}/labels',f'{b}/labels')
 with open(f'{out}/{split}.txt','w') as f:
  for s in stems:f.write(f'{b}/images/{os.path.basename(src[s])}\n')
names=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
with open(f'{out}/data.yaml','w') as f:
 f.write(f'path: {out}\ntrain: train.txt\nval: val.txt\nnc: 12\nnames:\n')
 for i,n in enumerate(names):f.write(f'  {i}: {n}\n')
print(out,'train',2000-len(val),'val',len(val))
