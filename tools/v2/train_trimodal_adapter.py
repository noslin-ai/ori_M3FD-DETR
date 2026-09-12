"""Continue the learned IR Adapter as an RGB+IR+depth single model."""
import argparse, os, sys
import torch
sys.path.insert(0, os.path.dirname(__file__))
import adapter_model
from ultralytics import YOLO
from adapter_model import add_depth_adapter
from adapter_dataset import TriModalDetectionTrainer
ROOT='/root/autodl-tmp/aic_race/M3F-DETR'
BASE=f'{ROOT}/runs/detect/runs/s3/ir_adapter_p23_frozen/weights/best.pt'

def flat(x):
    if torch.is_tensor(x): return [x]
    if isinstance(x,dict): return sum((flat(x[k]) for k in sorted(x)),[])
    if isinstance(x,(tuple,list)): return sum((flat(v) for v in x),[])
    return []

def main():
    p=argparse.ArgumentParser(); p.add_argument('--name',default='trimodal_adapter_p23_frozen'); p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=8)
    p.add_argument('--base',default=BASE); p.add_argument('--data',default=f'{ROOT}/data/s2_fullscale/data.yaml'); p.add_argument('--project',default='runs/s3')
    a=p.parse_args()
    ref=YOLO(a.base).model.float().cuda().eval()
    y=YOLO(a.base); y.model=add_depth_adapter(y.model).float().cuda().eval()
    x=torch.rand(1,6,128,128,device='cuda')
    with torch.inference_mode():
        r=flat(ref(x)); z=flat(y.model(x)); diff=max((u-v).abs().max().item() for u,v in zip(r,z))
    print('DEPTH_ZERO_IDENTITY_MAX_DIFF',diff,flush=True); assert diff==0.0
    y.train(
        trainer=TriModalDetectionTrainer, data=a.data,
        project=a.project, name=a.name, epochs=a.epochs, batch=a.batch, imgsz=1280,
        fraction=1.0, cache=False, device=0, workers=8, freeze=24,
        optimizer='AdamW', lr0=5e-4, lrf=0.05, cos_lr=True, warmup_epochs=2.0,
        momentum=0.937, weight_decay=5e-4, seed=42, deterministic=True,
        patience=999, amp=True, plots=False, hsv_h=0.015, hsv_s=0.4, hsv_v=0.3,
        degrees=0.0, translate=0.1, scale=0.15, shear=0.0, perspective=0.0,
        flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=0.0, mixup=0.0,
        copy_paste=0.0, auto_augment='randaugment', erasing=0.2, multi_scale=0.0,
        exist_ok=False,
    )
if __name__=='__main__': main()
