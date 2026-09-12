"""Parallel paired bootstrap: RGB vs IR vs trimodal official-metric caches."""
import json,os,random,sys
from multiprocessing import Pool
import numpy as np
ROOT='/root/autodl-tmp/aic_race/M3F-DETR'; sys.path.insert(0,f'{ROOT}/tools/v2')
from official_metric import evaluate
N=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
B=json.load(open(f'{ROOT}/runs/detect/runs/s2/eval/M1280_I1280.json'))
I=json.load(open(f'{ROOT}/runs/detect/runs/s3/eval/ir_adapter_p23_frozen.json'))
T=json.load(open(f'{ROOT}/runs/detect/runs/s3/eval/trimodal_adapter_p23_frozen.json'))
S=sorted(B)
def read_gt(s):
 out=[]
 for line in open(f'{ROOT}/data/train/labels/{s}.txt'):
  q=line.split(); out.append((int(float(q[0])),*[float(x) for x in q[1:5]]))
 return out
G={s:read_gt(s) for s in S}
def one(seed):
 rng=random.Random(seed); ss=[rng.choice(S) for _ in S]; pp=[{}, {}, {}]; gg={}
 for j,s in enumerate(ss):
  k=f'{j}:{s}'; gg[k]=G[s]
  for d,src in zip(pp,(B,I,T)): d[k]=[tuple(x) for x in src[s]]
 vals=[evaluate(d,gg,names=N)[0] for d in pp]
 return vals[2]-vals[0],vals[2]-vals[1]
if __name__=='__main__':
 seeds=range(2026091400,2026091600)
 with Pool(min(16,os.cpu_count() or 1)) as p: a=np.array(p.map(one,seeds))
 def rec(x):
  q=np.quantile(x,[.025,.975]); return {'mean':float(x.mean()),'median':float(np.median(x)),'ci95':[float(v) for v in q],'p_gt0':float((x>0).mean())}
 out={'n':len(a),'tri_minus_rgb':rec(a[:,0]),'tri_minus_ir':rec(a[:,1]),'samples':a.tolist()}
 print(json.dumps({k:v for k,v in out.items() if k!='samples'},indent=2)); json.dump(out,open(f'{ROOT}/runs/detect/runs/s3/eval/trimodal_bootstrap.json','w'),indent=2)
