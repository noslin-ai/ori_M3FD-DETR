"""Paired bootstrap: original trimodal vs neck/head fine-tune."""
import json,os,random,sys
from multiprocessing import Pool
import numpy as np
R='/root/autodl-tmp/aic_race/M3F-DETR'; sys.path.insert(0,f'{R}/tools/v2')
from official_metric import evaluate
N=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
A=json.load(open(f'{R}/runs/detect/runs/s3/eval/trimodal_adapter_p23_frozen.json')); B=json.load(open(f'{R}/runs/detect/runs/s3/eval/trimodal_neckhead_ft.json')); S=sorted(A)
def gt(s):
 out=[]
 for z in open(f'{R}/data/train/labels/{s}.txt'):
  q=z.split(); out.append((int(float(q[0])),*[float(x) for x in q[1:5]]))
 return out
G={s:gt(s) for s in S}
def one(seed):
 rng=random.Random(seed); ss=[rng.choice(S) for _ in S]; aa={};bb={};gg={}
 for i,s in enumerate(ss):
  k=f'{i}:{s}'; aa[k]=[tuple(x) for x in A[s]];bb[k]=[tuple(x) for x in B[s]];gg[k]=G[s]
 return evaluate(bb,gg,names=N)[0]-evaluate(aa,gg,names=N)[0]
if __name__=='__main__':
 with Pool(min(16,os.cpu_count() or 1)) as p:x=np.array(p.map(one,range(2026091700,2026091900)))
 q=np.quantile(x,[.025,.975]);o={'n':len(x),'mean':float(x.mean()),'median':float(np.median(x)),'ci95':[float(v) for v in q],'p_gt0':float((x>0).mean()),'samples':x.tolist()};print({k:v for k,v in o.items() if k!='samples'});json.dump(o,open(f'{R}/runs/detect/runs/s3/eval/trimodal_neckhead_bootstrap.json','w'),indent=2)
