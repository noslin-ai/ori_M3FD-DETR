"""Official fold0 evaluation for the six-channel IR residual Adapter."""
import os, sys, glob, json
import cv2
import numpy as np
import torch
ROOT='/root/autodl-tmp/aic_race/M3F-DETR'
sys.path.insert(0,f'{ROOT}/tools/v2')
import adapter_model  # required before loading custom checkpoint
from official_metric import evaluate
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

VIS=f'{ROOT}/data/train/visible'; IR=f'{ROOT}/data/train/infrared'; DEP=f'{ROOT}/data/train/depth'; LBL=f'{ROOT}/data/train/labels'
FOLD=f'{ROOT}/data/folds5_v2/fold0.txt'; OUT=f'{ROOT}/runs/detect/runs/s3/eval'
NAMES=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
MAX_DET=100; os.makedirs(OUT,exist_ok=True)

def path_map(d):
    out={}
    for e in ('*.png','*.jpg','*.jpeg'):
        for p in glob.glob(f'{d}/{e}'):
            s=os.path.splitext(os.path.basename(p))[0]
            if s in out: raise RuntimeError(f'duplicate stem {s}')
            out[s]=p
    return out
PM,IM,DM=path_map(VIS),path_map(IR),path_map(DEP)
STEMS=sorted(open(FOLD).read().split())
assert len(STEMS)==395 and all(s in PM and s in IM and s in DM for s in STEMS)

def gt(stem):
    a=[]; p=f'{LBL}/{stem}.txt'
    if os.path.exists(p):
        for ln in open(p):
            q=ln.split()
            if len(q)>=5: a.append((int(float(q[0])),*[float(x) for x in q[1:5]]))
    return a
GTS={s:gt(s) for s in STEMS}
SIZES={s:tuple(reversed(cv2.imread(PM[s]).shape[:2])) for s in STEMS}

def make_aux(stem):
    ir=cv2.imread(IM[stem],cv2.IMREAD_UNCHANGED)
    dep=cv2.imread(DM[stem],cv2.IMREAD_UNCHANGED)
    if ir.ndim==3: ir=cv2.cvtColor(ir,cv2.COLOR_BGR2GRAY)
    if dep.ndim==3: dep=dep[...,0]
    if dep.dtype==np.uint16:
        valid=dep>0; d=np.zeros(dep.shape,np.float32)
        d[valid]=np.log1p(np.clip(dep[valid].astype(np.float32),0,20000))/np.log1p(20000.0)
        d*=255.; v=valid.astype(np.float32)*255.
    else:
        valid=dep>0; d=dep.astype(np.float32); v=valid.astype(np.float32)*255.
    return np.stack((ir.astype(np.float32),d,v),-1)

def letterbox_pair(rgb,aux,lb):
    p=lb.get_params({'img':rgb})
    rgb=lb.apply_image({'img':rgb},p)['img']
    nu=p['new_unpad']
    if aux.shape[:2][::-1]!=nu:
        aux=np.stack((cv2.resize(aux[...,0],nu,interpolation=cv2.INTER_LINEAR),
                      cv2.resize(aux[...,1],nu,interpolation=cv2.INTER_NEAREST),
                      cv2.resize(aux[...,2],nu,interpolation=cv2.INTER_NEAREST)),-1)
    aux=cv2.copyMakeBorder(aux,p['top'],p['bottom'],p['left'],p['right'],cv2.BORDER_CONSTANT,value=(0,0,0))
    return rgb,np.ascontiguousarray(aux)

def infer(weights,imgsz,tag):
    cache=f'{OUT}/{tag}.json'
    if os.path.exists(cache):
        raw=json.load(open(cache)); return {s:[tuple(x) for x in a] for s,a in raw.items()}
    y=YOLO(weights); net=y.model.to('cuda').half().eval()
    lb=LetterBox(new_shape=(imgsz,imgsz),auto=True,stride=int(net.stride.max()))
    predout={}
    with torch.inference_mode():
        for n,s in enumerate(STEMS,1):
            im0=cv2.imread(PM[s],cv2.IMREAD_COLOR); aux=make_aux(s)
            im,aux=letterbox_pair(im0,aux,lb)
            rgb=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)).astype(np.float32)
            au=np.ascontiguousarray(aux.transpose(2,0,1))
            x=torch.from_numpy(np.concatenate((rgb,au),0)).to('cuda').half().div_(255).unsqueeze(0)
            raw=net(x); pred=raw[0] if isinstance(raw,(tuple,list)) else raw
            d=non_max_suppression(pred,conf_thres=.001,iou_thres=.7,max_det=MAX_DET)[0]
            rows=[]
            if len(d):
                d[:,:4]=ops.scale_boxes(x.shape[2:],d[:,:4],im0.shape)
                z=ops.xyxy2xywh(d[:,:4]); z[:,[0,2]]/=im0.shape[1]; z[:,[1,3]]/=im0.shape[0]
                rows=[(int(d[i,5]),*[float(v) for v in z[i]],float(d[i,4])) for i in range(len(d))]
            predout[s]=rows
            if n%50==0: print(tag,n,'/',len(STEMS),flush=True)
    json.dump(predout,open(cache,'w')); return predout

def score(preds,stems):
    return evaluate({s:preds[s] for s in stems},{s:GTS[s] for s in stems},names=NAMES)

def main(weights,imgsz,tag):
    p=infer(weights,imgsz,tag)
    groups={'all':STEMS,'large':[s for s in STEMS if max(SIZES[s])>=1280],'small':[s for s in STEMS if max(SIZES[s])<1280]}
    rec={'tag':tag,'weights':weights,'imgsz':imgsz,'boxes':sum(map(len,p.values())),'groups':{}}
    for name,ss in groups.items():
        val,pc,_=score(p,ss)
        rec['groups'][name]={'n':len(ss),'score':val,'per_class':{NAMES[c]:v*100 for c,v in pc.items()}}
        print(f'### {tag} {name} n={len(ss)} score={val:.4f}')
        print({NAMES[c]:round(v*100,2) for c,v in pc.items()})
    json.dump(rec,open(f'{OUT}/{tag}_metrics.json','w'),indent=2)

if __name__=='__main__': main(sys.argv[1],int(sys.argv[2]),sys.argv[3])
