"""Pack remaining generated images, verify against sources, then remove copies.

Already migrated image symlinks remain valid through migration.jsonl.
Restore tar archives with: tar -xf ARCHIVE -C ORIGINAL_REPOSITORY
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import argparse

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/bigtemp/wff7ad/h_SAE')

def main():
    global ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', choices=['Asyrp_official', 'sliders', 'MasaCtrl', 'SwiftEdit'])
    args = parser.parse_args()
    if args.baseline:
        ROOT = ROOT.parent / args.baseline
    groups = ['runs', 'results', 'sdxl_rebuttal/results', 'alpha_rebuttal/results', 'novelty_ablation/artifacts', 'sd15_mvp_out', 'sd15_negcfg_out', 'sd15_asyrp_multistep_out', 'test_images']
    if args.baseline:
        groups = ['runs', 'results', 'alpha_sweep', 'test_images']
    output = STORE / 'image_archives'
    output.mkdir(exist_ok=True)
    log = (STORE / 'audit' / 'image_archives.jsonl').open('a', buffering=1)
    for group in groups:
        paths = []
        for folder, dirs, files in os.walk(ROOT / group, followlinks=False):
            with os.scandir(folder) as entries:
                for entry in entries:
                    p = Path(entry.path)
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'} and entry.is_file(follow_symlinks=False):
                        paths.append(p.relative_to(ROOT))
        if not paths:
            continue
        destination = output / ((args.baseline + '__' if args.baseline else '') + group.replace('/', '__') + '.tar')
        if destination.exists():
            raise RuntimeError(f'Archive already exists; inspect before retrying: {destination}')
        listing = STORE / 'audit' / (destination.stem + '_images.list')
        listing.write_bytes(b'\0'.join(os.fsencode(p) for p in paths) + b'\0')
        subprocess.run(['tar', '-cf', str(destination), '--null', '-T', str(listing)], cwd=ROOT, check=True)
        subprocess.run(['tar', '-df', str(destination)], cwd=ROOT, check=True)
        h = hashlib.sha256()
        with destination.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024**2), b''):
                h.update(chunk)
        log.write(json.dumps({'archive': str(destination), 'original_root': str(ROOT), 'images': len(paths), 'bytes': destination.stat().st_size, 'sha256': h.hexdigest()}) + '\n')
        with ThreadPoolExecutor(max_workers=32) as pool:
            list(pool.map(lambda p: (ROOT / p).unlink(), paths))
        print(f'Archived and verified {len(paths)} images: {destination}', flush=True)

if __name__ == '__main__':
    main()
