"""Train the V2 IR residual adapter from the verified RGB-1280 fold0 checkpoint."""
import argparse, os, sys
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(__file__))
from adapter_model import attach_adapters
from adapter_dataset import TriModalDetectionTrainer

ROOT='/root/autodl-tmp/aic_race/M3F-DETR'
BASE=f'{ROOT}/runs/detect/runs/s2/full1280_v2/weights/best.pt'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',default='ir_adapter_p23_frozen')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--fraction',type=float,default=1.0)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--base',default=BASE)
    p.add_argument('--data',default=f'{ROOT}/data/s2_fullscale/data.yaml')
    p.add_argument('--project',default='runs/s3')
    a=p.parse_args()
    y=YOLO(a.base)
    y.model=attach_adapters(y.model,use_ir=True,use_depth=False)
    y.train(
        trainer=TriModalDetectionTrainer,
        data=a.data, project=a.project, name=a.name,
        epochs=a.epochs, fraction=a.fraction, batch=a.batch, imgsz=1280, cache=False,
        device=0, workers=8, freeze=24, optimizer='AdamW', lr0=1e-3, lrf=0.05,
        cos_lr=True, warmup_epochs=2.0, momentum=0.937, weight_decay=5e-4,
        seed=42, deterministic=True, patience=999, amp=True, plots=False,
        hsv_h=0.015, hsv_s=0.4, hsv_v=0.3, degrees=0.0, translate=0.1,
        scale=0.15, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
        bgr=0.0, mosaic=0.0, mixup=0.0, copy_paste=0.0,
        auto_augment='randaugment', erasing=0.2, multi_scale=0.0,
        exist_ok=False,
    )

if __name__=='__main__': main()
