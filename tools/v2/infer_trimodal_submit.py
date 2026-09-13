"""Six-channel trimodal test inference; cache top-100 then emit verified submission zips."""
import argparse,glob,json,math,os,shutil,sys,zipfile
import cv2,numpy as np,torch
R='/root/autodl-tmp/aic_race/M3F-DETR';sys.path.insert(0,f'{R}/tools/v2')
import adapter_model
import distribution_aligned_adapter  # noqa: F401 - checkpoint pickle dependency
import mage_exchange_adapter  # noqa: F401 - checkpoint pickle dependency
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils import ops
from ultralytics.utils.nms import non_max_suppression

def pmap(d):
 o={}
 for e in ('*.png','*.jpg','*.jpeg'):
  for p in glob.glob(f'{d}/{e}'):
   s=os.path.splitext(os.path.basename(p))[0];assert s not in o;o[s]=p
 return o
def aux(ip,dp):
 ir=cv2.imread(ip,cv2.IMREAD_UNCHANGED);d=cv2.imread(dp,cv2.IMREAD_UNCHANGED)
 if ir.ndim==3:ir=cv2.cvtColor(ir,cv2.COLOR_BGR2GRAY)
 if d.ndim==3:d=d[...,0]
 if d.dtype==np.uint16:
  v=d>0;z=np.zeros(d.shape,np.float32);z[v]=np.log1p(np.clip(d[v].astype(np.float32),0,20000))/np.log1p(20000.);z*=255.;v=v.astype(np.float32)*255.
 else:v=(d>0).astype(np.float32)*255.;z=d.astype(np.float32)
 return np.stack((ir.astype(np.float32),z,v),-1)
def pair(rgb,a,lb):
 p=lb.get_params({'img':rgb});rgb=lb.apply_image({'img':rgb},p)['img'];nu=p['new_unpad']
 if a.shape[:2][::-1]!=nu:a=np.stack((cv2.resize(a[...,0],nu,interpolation=cv2.INTER_LINEAR),cv2.resize(a[...,1],nu,interpolation=cv2.INTER_NEAREST),cv2.resize(a[...,2],nu,interpolation=cv2.INTER_NEAREST)),-1)
 a=cv2.copyMakeBorder(a,p['top'],p['bottom'],p['left'],p['right'],cv2.BORDER_CONSTANT,value=(0,0,0));return rgb,np.ascontiguousarray(a)
def infer(weights,cache):
 V=pmap(f'{R}/data/test/visible');I=pmap(f'{R}/data/test/infrared');D=pmap(f'{R}/data/test/depth');S=sorted(V);assert len(S)==1000 and set(S)==set(I)==set(D)
 net=YOLO(weights).model.cuda().half().eval();lb=LetterBox(new_shape=(1280,1280),auto=True,stride=int(net.stride.max()));out={}
 with torch.inference_mode():
  for n,s in enumerate(S,1):
   im0=cv2.imread(V[s]);im,a=pair(im0,aux(I[s],D[s]),lb);rgb=np.ascontiguousarray(im[:,:,::-1].transpose(2,0,1)).astype(np.float32);x=torch.from_numpy(np.concatenate((rgb,a.transpose(2,0,1)),0)).cuda().half().div_(255).unsqueeze(0)
   raw=net(x);pred=raw[0] if isinstance(raw,(tuple,list)) else raw;det=non_max_suppression(pred,.001,.7,max_det=100)[0];rows=[]
   if len(det):
    det[:,:4]=ops.scale_boxes(x.shape[2:],det[:,:4],im0.shape);z=ops.xyxy2xywh(det[:,:4]);z[:,[0,2]]/=im0.shape[1];z[:,[1,3]]/=im0.shape[0]
    rows=[(int(det[j,5]),*[float(v) for v in z[j]],float(det[j,4])) for j in range(len(det))]
   out[s]=rows
   if n%50==0:print(n,'/1000',flush=True)
 json.dump(out,open(cache,'w'));return out
def emit(raw,thr,outdir,zpath):
 if os.path.isdir(outdir):shutil.rmtree(outdir)
 os.makedirs(outdir);boxes=0
 for s in sorted(raw):
  rows=[]
  for r in raw[s]:
   c,x,y,w,h,sc=r;vals=(x,y,w,h,sc)
   if sc>=thr and 0<=c<12 and all(math.isfinite(v) for v in vals) and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1:
    rows.append(r)
   if len(rows)==100:break
  boxes+=len(rows)
  with open(f'{outdir}/{s}.txt','w') as f:
   for c,x,y,w,h,sc in rows:
    f.write(f'{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f} {sc:.6f}\n')
 files=glob.glob(f'{outdir}/*.txt');assert len(files)==1000 and max(sum(1 for _ in open(p)) for p in files)<=100
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
  for p in sorted(files):z.write(p,os.path.basename(p))
 with zipfile.ZipFile(zpath) as z:assert len(z.namelist())==1000 and all('/' not in n for n in z.namelist())
 print('PACKAGE',zpath,'files',len(files),'boxes',boxes,'bytes',os.path.getsize(zpath))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--weights',required=True);p.add_argument('--cache',required=True);p.add_argument('--thresholds',nargs='+',type=float,default=[.47]);a=p.parse_args();raw=json.load(open(a.cache)) if os.path.exists(a.cache) else infer(a.weights,a.cache)
 for t in a.thresholds:emit(raw,t,f'{R}/submissions/trimodal_fold0_conf{t:.2f}',f'{R}/submissions/trimodal_fold0_conf{t:.2f}.zip')
