"""Exact pycocotools COCOeval for cached normalized YOLO predictions."""
import contextlib,glob,io,json,os
import cv2,numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
R='/root/autodl-tmp/aic_race/M3F-DETR'; N=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
def pmap(d):
 o={}
 for e in ('*.png','*.jpg','*.jpeg'):
  for p in glob.glob(f'{d}/{e}'):o[os.path.splitext(os.path.basename(p))[0]]=p
 return o
PM=pmap(f'{R}/data/train/visible'); S=sorted(open(f'{R}/data/folds5_v2/fold0.txt').read().split())
images=[];anns=[];aid=1
for iid,s in enumerate(S,1):
 im=cv2.imread(PM[s]);h,w=im.shape[:2];images.append({'id':iid,'file_name':s,'width':w,'height':h})
 for line in open(f'{R}/data/train/labels/{s}.txt'):
  q=line.split();c=int(float(q[0]));x,y,bw,bh=map(float,q[1:5]); box=[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h]
  anns.append({'id':aid,'image_id':iid,'category_id':c+1,'bbox':box,'area':box[2]*box[3],'iscrowd':0});aid+=1
gt={'images':images,'annotations':anns,'categories':[{'id':i+1,'name':n} for i,n in enumerate(N)],'info':{},'licenses':[]}
coco=COCO();coco.dataset=gt;coco.createIndex(); ids={s:i for i,s in enumerate(S,1)}
def run(path):
 p=json.load(open(path));d=[]
 for s,rows in p.items():
  im=images[ids[s]-1];w,h=im['width'],im['height']
  for z in rows:
   c,x,y,bw,bh,sc=z;d.append({'image_id':ids[s],'category_id':int(c)+1,'bbox':[(x-bw/2)*w,(y-bh/2)*h,bw*w,bh*h],'score':sc})
 with contextlib.redirect_stdout(io.StringIO()):
  dt=coco.loadRes(d);e=COCOeval(coco,dt,'bbox');e.params.imgIds=list(ids.values());e.params.maxDets=[1,10,100];e.evaluate();e.accumulate();e.summarize()
 # precision T,R,K,A,M; area all=0, maxDet100=-1
 pc=[]
 for k in range(len(N)):
  v=e.eval['precision'][:,:,k,0,-1];v=v[v>-1];pc.append(float(v.mean()*100) if len(v) else float('nan'))
 return float(e.stats[0]*100),pc,len(d)
paths={'rgb':f'{R}/runs/detect/runs/s2/eval/M1280_I1280.json','ir':f'{R}/runs/detect/runs/s3/eval/ir_adapter_p23_frozen.json','tri':f'{R}/runs/detect/runs/s3/eval/trimodal_adapter_p23_frozen.json','neck':f'{R}/runs/detect/runs/s3/eval/trimodal_neckhead_ft.json'}
out={}
for tag,p in paths.items():
 score,pc,nb=run(p);out[tag]={'score':score,'boxes':nb,'per_class':dict(zip(N,pc))};print(tag,round(score,4),nb,{n:round(v,2) for n,v in zip(N,pc)})
json.dump(out,open(f'{R}/runs/detect/runs/s3/eval/pycoco_comparison.json','w'),indent=2)
