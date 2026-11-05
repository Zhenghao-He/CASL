"""Remove explicitly approved raw datasets and disposable derived files.

This is destructive. Dataset acquisition instructions and labels are retained.
"""
import json
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT.parent
AUDIT = Path('/bigtemp/wff7ad/h_SAE/audit')
TARGETS = [
    'CelebA/images', 'CelebA-HQ/data1024', 'ffhq-dataset/images1024x1024',
    'stargan-v2/data/train', 'stargan-v2/data/test', 'lsun/church',
    'CelebA-HQ.tar', 'img_align_celeba.zip',
]

def main():
    log = (AUDIT / 'deletions.jsonl').open('a', buffering=1)
    def record(path, reason):
        log.write(json.dumps({'path': str(path), 'reason': reason}) + '\n')
    for relative in TARGETS:
        target = PARENT / relative
        if not target.exists() or target.is_symlink():
            continue
        record(target, 'User-requested removal of downloadable raw dataset; labels and loader source retained')
        if target.is_dir():
            with ThreadPoolExecutor(max_workers=32) as pool:
                for folder, dirs, files in os.walk(target, topdown=False, followlinks=False):
                    list(pool.map(lambda p: p.unlink(), (Path(folder) / name for name in files)))
                    for name in dirs:
                        p = Path(folder) / name
                        p.unlink() if p.is_symlink() else p.rmdir()
                target.rmdir()
        else:
            target.unlink()
        print(f'deleted dataset {target}', flush=True)
    for folder, dirs, files in os.walk(ROOT, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules'}]
        if '__pycache__' in dirs:
            target = Path(folder) / '__pycache__'
            if not target.is_symlink():
                record(target, 'Regenerated Python bytecode')
                shutil.rmtree(target)
            dirs.remove('__pycache__')
        for name in files:
            p = Path(folder) / name
            if p.is_symlink():
                continue
            if name.endswith(('.backup', '.broken')):
                dst = AUDIT / 'old_scripts' / p.relative_to(ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dst)
                record(p, 'Obsolete script variant retained in audit/old_scripts')
                p.unlink()
            elif p.suffix.lower() in {'.png', '.pdf', '.svg'} and any(x in {'charts_results', 'charts_results_pdf', 'plots'} for x in p.relative_to(ROOT).parts):
                record(p, 'Replottable chart; source metrics and plotting code retained')
                p.unlink()

if __name__ == '__main__':
    main()
