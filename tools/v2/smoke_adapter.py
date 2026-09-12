"""Smoke tests for the V2 residual adapter model (no optimizer training)."""
import copy
import sys
import torch
from ultralytics import YOLO
sys.path.insert(0, '/root/autodl-tmp/aic_race/M3F-DETR/tools/v2')
from adapter_model import attach_adapters

BASE='/root/autodl-tmp/aic_race/M3F-DETR/runs/detect/runs/s2/full1280_v2/weights/best.pt'

def tensors(x):
    if torch.is_tensor(x): return [x]
    if isinstance(x, dict):
        z=[]
        for k in sorted(x): z += tensors(x[k])
        return z
    if isinstance(x, (tuple,list)):
        z=[]
        for v in x: z += tensors(v)
        return z
    return []

ref=YOLO(BASE).model.float().cuda().eval()
adapt=YOLO(BASE).model.float()
adapt=attach_adapters(adapt,use_ir=True,use_depth=False).cuda().eval()
x3=torch.rand(1,3,128,128,device='cuda')
x6=torch.cat((x3,torch.rand(1,3,128,128,device='cuda')),1)
with torch.inference_mode():
    a=tensors(ref(x3)); b=tensors(adapt(x6))
assert len(a)==len(b) and a
mx=max((u-v).abs().max().item() for u,v in zip(a,b))
print('identity_tensor_count',len(a),'max_abs_diff',mx)
assert mx==0.0

adapt.train(); adapt.zero_grad(set_to_none=True)
o=adapt.predict(x6)
feat=o['feats'] if isinstance(o,dict) else o
loss=sum(v.float().square().mean() for v in feat)
loss.backward()
for name in ('ir_inject_p2','ir_inject_p3'):
    m=getattr(adapt,name)
    g=sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(name,'grad_sum',g)
    assert g>0
base_n=sum(p.numel() for p in ref.parameters())
adapt_n=sum(p.numel() for p in adapt.parameters())
print('base_params',base_n,'adapter_params',adapt_n-base_n,'pct',100*(adapt_n-base_n)/base_n)
print('SMOKE_MODEL_OK')
