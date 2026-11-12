import os
from pathlib import Path
from huggingface_hub import snapshot_download
root = Path(os.environ.get('SEG_ROOT', '/data/yipu/Seg'))
for repo, name in [('Ricky06662/Seg-Zero-7B', 'Seg-Zero-7B'), ('facebook/sam2-hiera-large', 'sam2-hiera-large')]:
    snapshot_download(repo, local_dir=str(root / 'weights' / name), max_workers=4)
