"""Convert a JSON list and optional class-ID masks into the benchmark format."""
import argparse,json,re
from pathlib import Path
import numpy as np
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root).resolve();dest=Path(a.output).resolve();dest.parent.mkdir(parents=True,exist_ok=True)
rows=json.loads(Path(a.input).read_text());converted=[];seen=set()
for row in rows:
    key=row['id']
    if not re.fullmatch('[A-Za-z0-9_-]+',key) or key in seen:raise ValueError('Invalid/duplicate id: '+key)
    seen.add(key)
    image=(root/row['image']).resolve()
    with Image.open(image) as im:size=im.size
    result={'id':key,'image':str(image),'text':row['text']}
    if not isinstance(row['text'],str) or not row['text'].strip():raise ValueError('Empty text: '+key)
    if 'mask' in row:
        with Image.open(root/row['mask']) as im:
            if im.size!=size:raise ValueError('Mask size mismatch: '+key)
            mask=np.asarray(im)
        if mask.ndim!=2:raise ValueError('Mask must be single-channel: '+key)
        binary=mask==row['target_id'] if 'target_id' in row else mask>0
        folder=dest.parent/'binary_masks';folder.mkdir(exist_ok=True)
        path=folder/(key+'.png');Image.fromarray(binary.astype(np.uint8)*255).save(path)
        result['mask']=str(path)
    converted.append(result)
dest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in converted))
print('Converted',len(converted),'samples to',dest)
