#!/usr/bin/env python3
import argparse, json, math, os, sys, zipfile
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('-c','--config',required=True)
    p.add_argument('-r','--resume',required=True)
    p.add_argument('--images',required=True)
    p.add_argument('--output-root',required=True)
    p.add_argument('--cache',required=True)
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--batch',type=int,default=11)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--thresholds',type=float,nargs='*',default=[0.3,0.4,0.45,0.47,0.5])
    p.add_argument('--count-match',type=int,default=0)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--dfine-root',default='/root/autodl-tmp/aic_race/D-FINE')
    return p.parse_args()

class Images(Dataset):
    def __init__(self,root,size):
        self.files=sorted(p for p in Path(root).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'})
        self.size=size
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        p=self.files[i]
        im=Image.open(p).convert('RGB')
        w,h=im.size
        x=TF.to_tensor(TF.resize(im,[self.size,self.size],antialias=True))
        return x,torch.tensor([w,h]),p.stem

def validate_row(row):
    c,cx,cy,w,h,s=row
    return int(c)==c and 0<=c<12 and all(math.isfinite(v) for v in row) and 0<=cx<=1 and 0<=cy<=1 and 0<w<=1 and 0<h<=1 and 0<=s<=1

def emit(cache, stems, out_root, tag, thr):
    out=Path(out_root)/tag
    out.mkdir(parents=True,exist_ok=True)
    total=0
    for stem in stems:
        rows=[r for r in cache[stem] if r[5]>=thr][:100]
        if not all(validate_row(r) for r in rows): raise RuntimeError(f'invalid row: {stem}')
        with (out/f'{stem}.txt').open('w',newline='\n') as f:
            for c,cx,cy,w,h,s in rows:
                f.write(f'{int(c)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {s:.6f}\n')
        total+=len(rows)
    z=Path(out_root)/(tag+'.zip')
    with zipfile.ZipFile(z,'w',zipfile.ZIP_DEFLATED) as q:
        for stem in stems:q.write(out/f'{stem}.txt',f'{stem}.txt')
    with zipfile.ZipFile(z) as q:
        if len(q.namelist())!=len(stems) or any('/' in n for n in q.namelist()): raise RuntimeError('bad zip layout')
    print('PACKAGE',tag,'thr',thr,'files',len(stems),'boxes',total,'zip',z,flush=True)
    return total,z

def main():
    a=parse_args()
    repo=Path(a.dfine_root)
    sys.path.insert(0,str(repo))
    from src.core import YAMLConfig
    cfg=YAMLConfig(a.config,resume=a.resume)
    if 'HGNetv2' in cfg.yaml_cfg: cfg.yaml_cfg['HGNetv2']['pretrained']=False
    ck=torch.load(a.resume,map_location='cpu',weights_only=False)
    state=ck['ema']['module'] if ck.get('ema') else ck['model']
    cfg.model.load_state_dict(state)
    model=cfg.model.deploy().to(a.device).eval()
    post=cfg.postprocessor.deploy().to(a.device).eval()
    ds=Images(a.images,a.imgsz)
    if len(ds)!=1000: raise RuntimeError(f'expected 1000 images, got {len(ds)}')
    dl=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=a.workers>0)
    cache={}
    with torch.inference_mode():
        for bi,(x,sizes,stems) in enumerate(dl):
            x=x.to(a.device,non_blocking=True)
            sizes=sizes.to(a.device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.float16):
                labels,boxes,scores=post(model(x),sizes)
            for j,stem in enumerate(stems):
                w0,h0=map(float,sizes[j].tolist())
                rows=[]
                order=torch.argsort(scores[j],descending=True)
                for k in order.tolist():
                    s=float(scores[j,k]); c=int(labels[j,k])
                    x1,y1,x2,y2=map(float,boxes[j,k].tolist())
                    x1=max(0.,min(w0,x1)); x2=max(0.,min(w0,x2)); y1=max(0.,min(h0,y1)); y2=max(0.,min(h0,y2))
                    bw=x2-x1; bh=y2-y1
                    if c<0 or c>=12 or bw<=0 or bh<=0 or not math.isfinite(s): continue
                    row=[c,(x1+x2)/(2*w0),(y1+y2)/(2*h0),bw/w0,bh/h0,s]
                    if validate_row(row): rows.append(row)
                cache[stem]=rows
            if (bi+1)%10==0 or bi+1==len(dl): print('INFER',bi+1,'/',len(dl),flush=True)
    Path(a.cache).parent.mkdir(parents=True,exist_ok=True)
    Path(a.cache).write_text(json.dumps(cache,separators=(',',':')))
    stems=[p.stem for p in ds.files]
    flat=sorted((r[5] for stem in stems for r in cache[stem][:100]),reverse=True)
    print('CACHE',a.cache,'images',len(cache),'top100_scores',len(flat),'score_quantiles',[flat[int((len(flat)-1)*q)] for q in [0,.25,.5,.75,.9,.95,.99]],flush=True)
    for t in a.thresholds: emit(cache,stems,a.output_root,f'dfine_l_soft1280_b11_conf{t:g}',t)
    if a.count_match:
        if a.count_match>len(flat): raise RuntimeError('count match exceeds predictions')
        t=flat[a.count_match-1]
        emit(cache,stems,a.output_root,f'dfine_l_soft1280_b11_countmatch{a.count_match}',t)

if __name__=='__main__': main()
