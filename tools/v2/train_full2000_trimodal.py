"""Fixed-budget full-2000 continuation of the tested trimodal model."""
import glob,os,sys
R='/root/autodl-tmp/aic_race/M3F-DETR';sys.path.insert(0,f'{R}/tools/v2')
import adapter_model
from ultralytics import YOLO
from adapter_dataset import TriModalDetectionTrainer

def build_view():
 out=f'{R}/data/v2_full2000';os.makedirs(out,exist_ok=True);src={}
 for e in ('*.png','*.jpg','*.jpeg'):
  for p in glob.glob(f'{R}/data/train/visible/{e}'):src[os.path.splitext(os.path.basename(p))[0]]=p
 assert len(src)==2000
 for split,stems in [('train',sorted(src)),('val',sorted(open(f'{R}/data/folds5_v2/fold0.txt').read().split()))]:
  b=f'{out}/{split}';os.makedirs(b,exist_ok=True)
  for n,t in [('images',f'{R}/data/train/visible'),('labels',f'{R}/data/train/labels')]:
   p=f'{b}/{n}'
   if not os.path.lexists(p):os.symlink(t,p)
  with open(f'{out}/{split}.txt','w') as f:
   for s in stems:f.write(f'{b}/images/{os.path.basename(src[s])}\n')
 names=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
 with open(f'{out}/data.yaml','w') as f:
  f.write(f'path: {out}\ntrain: train.txt\nval: val.txt\nnc: 12\nnames:\n')
  for i,n in enumerate(names):f.write(f'  {i}: {n}\n')
 return f'{out}/data.yaml'

def main():
 data=build_view();base=f'{R}/runs/detect/runs/s3/trimodal_adapter_p23_frozen/weights/best.pt';y=YOLO(base);assert y.model.use_ir and y.model.use_depth
 y.train(trainer=TriModalDetectionTrainer,data=data,project='runs/s5',name='trimodal_full2000_cont10',epochs=10,batch=8,imgsz=1280,fraction=1.0,cache=False,device=0,workers=8,freeze=24,optimizer='AdamW',lr0=1e-4,lrf=0.1,cos_lr=True,warmup_epochs=1.0,momentum=0.937,weight_decay=5e-4,seed=44,deterministic=True,patience=999,amp=True,plots=False,hsv_h=0.015,hsv_s=0.4,hsv_v=0.3,degrees=0.0,translate=0.1,scale=0.15,shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.5,bgr=0.0,mosaic=0.0,mixup=0.0,copy_paste=0.0,auto_augment='randaugment',erasing=0.2,multi_scale=0.0,exist_ok=False)
if __name__=='__main__':main()
