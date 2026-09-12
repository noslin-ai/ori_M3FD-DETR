"""OOF confidence sweep for cached trimodal predictions."""
import json,os,sys
from multiprocessing import Pool
R='/root/autodl-tmp/aic_race/M3F-DETR';sys.path.insert(0,f'{R}/tools/v2')
from official_metric import evaluate
N=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle'];P=json.load(open(f'{R}/runs/detect/runs/s4/eval/oof01/tri.json'));S=sorted(P)
def gt(s):
 o=[]
 for z in open(f'{R}/data/train/labels/{s}.txt'):
  q=z.split();o.append((int(float(q[0])),*[float(x) for x in q[1:5]]))
 return o
G={s:gt(s) for s in S}
def one(t):return t,evaluate({s:[tuple(x) for x in P[s] if x[5]>=t] for s in S},G,names=N)[0],sum(x[5]>=t for s in S for x in P[s])
if __name__=='__main__':
 ts=[.001,.05,.1,.2,.3,.4,.45,.47,.5,.55,.6]
 with Pool(len(ts)) as p:r=p.map(one,ts)
 for z in r:print(*z)
 json.dump(r,open(f'{R}/runs/detect/runs/s4/eval/oof01/conf_sweep.json','w'))
