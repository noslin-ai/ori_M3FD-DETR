"""Add a zero-init depth residual to the 57.180 IR Adapter and train depth only."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT=Path('/root/autodl-tmp/aic_race/M3F-DETR')
sys.path.insert(0,str(ROOT/'tools/v2'))
import adapter_model  # noqa: F401
import distribution_aligned_adapter  # noqa: F401
from adapter_model import add_depth_adapter, AdapterDetectionModel
from adapter_dataset import TriModalDetectionTrainer
from ultralytics.utils.torch_utils import unwrap_model

class DepthOnlyTrainer(TriModalDetectionTrainer):
    def _setup_train(self):
        super()._setup_train()
        model=unwrap_model(self.model)
        trainable=[]
        for name,p in model.named_parameters():
            keep=name.startswith(('depth_adapter.','depth_inject_p2.','depth_inject_p3.'))
            p.requires_grad_(keep)
            if keep: trainable.append(name)
        if not trainable or any(not n.startswith('depth_') for n in trainable):
            raise RuntimeError(f'invalid depth-only trainable set: {trainable}')
        print(f'DEPTH_ONLY_TRAINABLE tensors={len(trainable)} params={sum(p.numel() for p in model.parameters() if p.requires_grad)}',flush=True)
        print('\n'.join(trainable),flush=True)

def flat(x):
    if torch.is_tensor(x): return [x]
    if isinstance(x,dict): return sum((flat(x[k]) for k in sorted(x)),[])
    if isinstance(x,(tuple,list)): return sum((flat(v) for v in x),[])
    return []

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default=str(ROOT/'runs/detect/runs/s5/champion_da_ir_p23/weights/best.pt'))
    p.add_argument('--data',default=str(ROOT/'data/yolo_full2000_soft/data.yaml'))
    p.add_argument('--project',default=str(ROOT/'runs/detect/runs/s7'))
    p.add_argument('--name',default='champion_ir_depthonly_p23')
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch',type=int,default=8)
    a=p.parse_args()

    ref=YOLO(a.base).model.float().cuda().eval()
    if not isinstance(ref,AdapterDetectionModel) or not getattr(ref,'use_ir',False) or getattr(ref,'use_depth',False):
        raise TypeError('base must be the IR-only Adapter checkpoint')
    y=YOLO(a.base)
    y.model=add_depth_adapter(y.model).float().cuda().eval()
    y.model.depth_valid_gate=False
    x=torch.rand(1,6,128,128,device='cuda'); x[:,5:6]=(x[:,5:6]>.3).float()
    with torch.inference_mode():
        before=flat(ref(x)); after=flat(y.model(x))
        diff=max((u-v).abs().max().item() for u,v in zip(before,after))
    print('DEPTH_ZERO_IDENTITY_MAX_DIFF',diff,flush=True)
    if diff != 0.0: raise RuntimeError('zero depth branch changed the winning IR model')

    y.train(
        trainer=DepthOnlyTrainer,data=a.data,project=a.project,name=a.name,
        epochs=a.epochs,batch=a.batch,imgsz=1280,fraction=1.0,cache=False,
        device=0,workers=8,freeze=24,optimizer='AdamW',lr0=5e-4,lrf=0.05,
        cos_lr=True,warmup_epochs=2.0,momentum=0.937,weight_decay=5e-4,
        seed=42,deterministic=True,patience=999,amp=True,plots=False,
        hsv_h=0.002,hsv_s=0.06,hsv_v=0.06,degrees=0.0,translate=0.03,
        scale=0.15,shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.5,bgr=0.0,
        mosaic=0.0,mixup=0.0,copy_paste=0.0,auto_augment=None,erasing=0.0,
        multi_scale=0.0,exist_ok=False,
    )

if __name__=='__main__': main()
