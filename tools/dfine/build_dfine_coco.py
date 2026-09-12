import json,sys
from pathlib import Path
from PIL import Image
ROOT=Path('/root/autodl-tmp/aic_race/M3F-DETR')
OUT=Path('/root/autodl-tmp/aic_race/D-FINE/data/aic_soft')
NAMES=['person','boat','animal','seat','sign','bicycle','car','ball','light','garbage can','uav','tricycle']
(OUT/'annotations').mkdir(parents=True,exist_ok=True)
for split in ('train','val'):
    imgdir=ROOT/f'data/yolo_full2000_soft/{split}/images'
    paths=sorted(p for p in imgdir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.tif','.tiff'})
    images=[]; anns=[]; aid=1
    for iid,p in enumerate(paths,1):
        with Image.open(p) as im:w,h=im.size
        images.append({'id':iid,'file_name':p.name,'width':w,'height':h})
        lp=ROOT/'data/train/labels'/f'{p.stem}.txt'
        for line in lp.read_text().splitlines():
            q=line.split(); c=int(float(q[0]));cx,cy,bw,bh=map(float,q[1:5])
            x=max(0.0,(cx-bw/2)*w);y=max(0.0,(cy-bh/2)*h);x2=min(float(w),(cx+bw/2)*w);y2=min(float(h),(cy+bh/2)*h)
            ww=x2-x;hh=y2-y
            if not (0<=c<12 and ww>0 and hh>0):raise ValueError((p,line))
            anns.append({'id':aid,'image_id':iid,'category_id':c,'bbox':[x,y,ww,hh],'area':ww*hh,'iscrowd':0});aid+=1
    data={'images':images,'annotations':anns,'categories':[{'id':i,'name':n} for i,n in enumerate(NAMES)]}
    dst=OUT/'annotations'/f'instances_{split}.json';dst.write_text(json.dumps(data,separators=(',',':')))
    print(split,'images',len(images),'annotations',len(anns),'json',dst)
    expected=1800 if split=='train' else 200
    assert len(images)==expected and len({x['file_name'] for x in images})==expected
