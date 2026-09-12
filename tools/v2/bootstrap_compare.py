"""Parallel paired image bootstrap for cached official-metric predictions."""
import json, os, random, sys
from multiprocessing import Pool
import numpy as np
ROOT='/root/autodl-tmp/aic_race/M3F-DETR'
sys.path.insert(0,f'{ROOT}/tools/v2')
from official_metric import evaluate
NAMES=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
BASE=json.load(open(f'{ROOT}/runs/detect/runs/s2/eval/M1280_I1280.json'))
ADAPT=json.load(open(f'{ROOT}/runs/detect/runs/s3/eval/ir_adapter_p23_frozen.json'))
STEMS=sorted(BASE)
def read_gt(s):
    out=[]
    for line in open(f'{ROOT}/data/train/labels/{s}.txt'):
        q=line.split(); out.append((int(float(q[0])),*[float(x) for x in q[1:5]]))
    return out
GTS={s:read_gt(s) for s in STEMS}
def one(seed):
    rng=random.Random(seed); ss=[rng.choice(STEMS) for _ in STEMS]; bp={}; ap={}; gt={}
    for i,s in enumerate(ss):
        k=f'{i}:{s}'; bp[k]=[tuple(x) for x in BASE[s]]; ap[k]=[tuple(x) for x in ADAPT[s]]; gt[k]=GTS[s]
    return evaluate(ap,gt,names=NAMES)[0]-evaluate(bp,gt,names=NAMES)[0]
if __name__=='__main__':
    seeds=[2026091200+i for i in range(200)]
    with Pool(min(16,os.cpu_count() or 1)) as pool: a=np.array(pool.map(one,seeds))
    q=np.quantile(a,[.025,.975]); rec={'n':len(a),'seed_start':seeds[0],'mean':float(a.mean()),'median':float(np.median(a)),'ci95':[float(x) for x in q],'p_gt0':float((a>0).mean()),'samples':a.tolist()}
    print(json.dumps({k:v for k,v in rec.items() if k!='samples'},indent=2))
    json.dump(rec,open(f'{ROOT}/runs/detect/runs/s3/eval/ir_adapter_bootstrap.json','w'),indent=2)
