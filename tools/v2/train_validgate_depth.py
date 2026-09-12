"""Train a valid-mask-gated depth Adapter from the learned fold0 IR model."""
import os,sys,torch
R='/root/autodl-tmp/aic_race/M3F-DETR';sys.path.insert(0,f'{R}/tools/v2')
import adapter_model
from adapter_model import add_depth_adapter
from adapter_dataset import TriModalDetectionTrainer
from ultralytics import YOLO

def flat(x):
 if torch.is_tensor(x):return [x]
 if isinstance(x,dict):return sum((flat(x[k]) for k in sorted(x)),[])
 if isinstance(x,(tuple,list)):return sum((flat(v) for v in x),[])
 return []
def main():
 base=f'{R}/runs/detect/runs/s3/ir_adapter_p23_frozen/weights/best.pt';ref=YOLO(base).model.float().cuda().eval();y=YOLO(base);y.model=add_depth_adapter(y.model).float().cuda().eval();y.model.depth_valid_gate=True
 x=torch.rand(1,6,128,128,device='cuda');x[:,5:6]=(x[:,5:6]>.3).float()
 with torch.inference_mode():a=flat(ref(x));b=flat(y.model(x));d=max((u-v).abs().max().item() for u,v in zip(a,b))
 print('GATED_DEPTH_ZERO_IDENTITY_MAX_DIFF',d,flush=True);assert d==0.0
 y.train(trainer=TriModalDetectionTrainer,data=f'{R}/data/s2_fullscale/data.yaml',project='runs/s6',name='fold0_trimodal_validgate',epochs=20,batch=8,imgsz=1280,fraction=1.0,cache=False,device=0,workers=8,freeze=24,optimizer='AdamW',lr0=5e-4,lrf=.05,cos_lr=True,warmup_epochs=2.0,momentum=.937,weight_decay=5e-4,seed=42,deterministic=True,patience=999,amp=True,plots=False,hsv_h=.015,hsv_s=.4,hsv_v=.3,degrees=0.0,translate=.1,scale=.15,shear=0.0,perspective=0.0,flipud=0.0,fliplr=.5,bgr=0.0,mosaic=0.0,mixup=0.0,copy_paste=0.0,auto_augment='randaugment',erasing=.2,multi_scale=0.0,exist_ok=False)
if __name__=='__main__':main()
