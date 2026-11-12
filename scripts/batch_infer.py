"""Simple isolated per-sample baseline; reloads weights for each sample."""
import argparse,json,os,re,subprocess,sys
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
base=Path(__file__).resolve().parents[1];out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
weights=Path(os.environ.get('SEG_ROOT','/data/yukun/Seg'))/'weights';seen=set();rows=[]
for line in Path(a.manifest).read_text().splitlines():
    if not line.strip():continue
    r=json.loads(line)
    if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']) or r['id'] in seen:raise ValueError('Invalid/duplicate id')
    seen.add(r['id']);image=Path(a.root)/r['image']
    with Image.open(image) as im: size=im.size
    if 'mask' in r:
        with Image.open(Path(a.root)/r['mask']) as m:
            if m.size!=size:raise ValueError('Mask size mismatch')
    rows.append(r)
for r in rows:
    cmd=[sys.executable,str(base/'infer.py'),'--reasoning_model_path',str(weights/'Seg-Zero-7B'),'--segmentation_model_path',str(weights/'sam2-hiera-large/sam2_hiera_large.pt'),'--image_path',str((Path(a.root)/r['image']).resolve()),'--text',r['text'],'--output_path',str(out/(r['id']+'.png'))]
    with (out/(r['id']+'.log')).open('w') as f: result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
    with (out/'status.jsonl').open('a') as f:f.write(json.dumps({'id':r['id'],'returncode':result.returncode})+'\n')
