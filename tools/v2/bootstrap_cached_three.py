"""Generic 3-way paired COCO bootstrap over cached predictions."""
import argparse,json,os,random,sys
from multiprocessing import Pool
import numpy as np
R='/root/autodl-tmp/aic_race/M3F-DETR';sys.path.insert(0,f'{R}/tools/v2')
from official_metric import evaluate
N=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--ir',required=True);p.add_argument('--tri',required=True);p.add_argument('--out',required=True);p.add_argument('--n',type=int,default=200);a=p.parse_args()
B=json.load(open(a.base));I=json.load(open(a.ir));T=json.load(open(a.tri));S=sorted(B);assert set(S)==set(I)==set(T)
def gt(s):
 out=[]
 for line in open(f'{R}/data/train/labels/{s}.txt'):
  q=line.split();out.append((int(float(q[0])),*[float(x) for x in q[1:5]]))
 return out
G={s:gt(s) for s in S}
def one(seed):
 rng=random.Random(seed);ss=[rng.choice(S) for _ in S];ds=[{}, {}, {}];gg={}
 for j,s in enumerate(ss):
  k=f'{j}:{s}';gg[k]=G[s]
  for d,src in zip(ds,(B,I,T)):d[k]=[tuple(x) for x in src[s]]
 v=[evaluate(d,gg,names=N)[0] for d in ds];return v[1]-v[0],v[2]-v[0],v[2]-v[1]
def rec(x):
 q=np.quantile(x,[.025,.975]);return {'mean':float(x.mean()),'median':float(np.median(x)),'ci95':[float(v) for v in q],'p_gt0':float((x>0).mean())}
if __name__=='__main__':
 with Pool(min(16,os.cpu_count() or 1)) as pool:z=np.array(pool.map(one,range(2026092000,2026092000+a.n)))
 o={'n':len(z),'ir_minus_rgb':rec(z[:,0]),'tri_minus_rgb':rec(z[:,1]),'tri_minus_ir':rec(z[:,2]),'samples':z.tolist()};print(json.dumps({k:v for k,v in o.items() if k!='samples'},indent=2));json.dump(o,open(a.out,'w'),indent=2)
