"""Low-LR continuation of the converging RGB+IR+depth Adapter."""
import argparse,os,sys
sys.path.insert(0,os.path.dirname(__file__))
import adapter_model
from ultralytics import YOLO
from adapter_dataset import TriModalDetectionTrainer
ROOT='/root/autodl-tmp/aic_race/M3F-DETR'; BASE=f'{ROOT}/runs/detect/runs/s3/trimodal_adapter_p23_frozen/weights/best.pt'
def main():
 p=argparse.ArgumentParser(); p.add_argument('--name',default='trimodal_adapter_p23_frozen_cont'); p.add_argument('--epochs',type=int,default=10); a=p.parse_args()
 y=YOLO(BASE); assert y.model.use_ir and y.model.use_depth
 y.train(trainer=TriModalDetectionTrainer,data=f'{ROOT}/data/s2_fullscale/data.yaml',project='runs/s3',name=a.name,epochs=a.epochs,batch=8,imgsz=1280,fraction=1.0,cache=False,device=0,workers=8,freeze=24,optimizer='AdamW',lr0=2e-4,lrf=0.1,cos_lr=True,warmup_epochs=1.0,momentum=0.937,weight_decay=5e-4,seed=43,deterministic=True,patience=999,amp=True,plots=False,hsv_h=0.015,hsv_s=0.4,hsv_v=0.3,degrees=0.0,translate=0.1,scale=0.15,shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.5,bgr=0.0,mosaic=0.0,mixup=0.0,copy_paste=0.0,auto_augment='randaugment',erasing=0.2,multi_scale=0.0,exist_ok=False)
if __name__=='__main__': main()
