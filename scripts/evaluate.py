"""Evaluate predicted binary PNG masks; missing predictions count as empty masks."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--root',required=True);p.add_argument('--predictions',required=True);a=p.parse_args()
ious=[]; intersection=union=missing=0
for line in Path(a.manifest).read_text().splitlines():
    if not line.strip(): continue
    row=json.loads(line); gt=np.asarray(Image.open(Path(a.root)/row['mask']).convert('L'))>0
    path=Path(a.predictions)/(row['id']+'.png.mask.png')
    if path.exists():
        pred=np.asarray(Image.open(path).convert('L'))>0
        if pred.shape!=gt.shape: raise ValueError('Mask size mismatch: '+row['id'])
    else: pred=np.zeros_like(gt);missing+=1
    i=int((pred&gt).sum());u=int((pred|gt).sum());intersection+=i;union+=u;ious.append(i/u if u else 1.)
if not ious: raise ValueError('Empty manifest')
print(json.dumps({'samples':len(ious),'missing_predictions':missing,'mean_IoU':float(np.mean(ious)), 'global_IoU':intersection/union if union else 1.},indent=2))
