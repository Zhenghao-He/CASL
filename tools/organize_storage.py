"""Archive expensive artifacts, preserving original paths as symlinks.

Run from any directory; storage destination is deliberately explicit.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
import threading

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/bigtemp/wff7ad/h_SAE')
BINARY = {'.pt', '.pth', '.ckpt', '.safetensors', '.npy', '.npz', '.pkl', '.pickle', '.onnx', '.bin', '.h5', '.hdf5'}
OUTPUTS = {'runs', 'results', 'artifacts', 'test_images', 'sd15_asyrp_multistep_out', 'sd15_asyrp_style_out', 'sd15_mvp_out', 'sd15_negcfg_out'}

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    STORE.mkdir(parents=True, exist_ok=True)
    log = (STORE / 'audit' / 'migration.jsonl').open('a', buffering=1)
    log_lock = threading.Lock()
    roots = [(ROOT, Path('Asyrp_SAE'))]
    for name in ['Asyrp_official', 'DiT', 'sliders', 'MasaCtrl', 'SwiftEdit', 'CelebFaces_Attributes_Classification']:
        roots.append((ROOT.parent / name, Path(name)))
    for name in ['casl_cross_attribute_cache', 'casl_mapping_ablation']:
        roots.append((Path('/u/wff7ad') / name, Path('external') / name))
    roots.append((Path('/tmp/casl_sdxl'), Path('external/casl_sdxl')))
    def tasks():
        for root, prefix in roots:
            for folder, dirs, files in os.walk(root, followlinks=False):
                dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'node_modules', '.venv', 'reproduction'}]
                for name in files:
                    src = Path(folder) / name
                    if (name.startswith('importance_k') and src.suffix == '.npy') or src.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}:
                        continue
                    if src.is_symlink() or not src.is_file():
                        continue
                    if src.suffix.lower() in BINARY or (src.stat().st_size > 5 * 1024**2 and src.suffix.lower() not in {'.txt', '.csv', '.json', '.patch'}):
                        yield src, STORE / prefix / src.relative_to(root)

    def migrate(item):
        src, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        checksum = digest(src)
        if dst.exists():
            if dst.stat().st_size != size or digest(dst) != checksum:
                raise RuntimeError(f'Conflicting archive: {dst}')
        else:
            tmp = dst.with_name(dst.name + '.transfer')
            shutil.copy2(src, tmp)
            if tmp.stat().st_size != size or digest(tmp) != checksum:
                raise RuntimeError(f'Copy verification failed: {src}')
            tmp.rename(dst)
        src.unlink()
        src.symlink_to(dst)
        with log_lock:
            log.write(json.dumps({'source': str(src), 'destination': str(dst), 'bytes': size, 'sha256': checksum}) + '\n')
            print(f'archived {size} {src}', flush=True)

    # Per-file parallelism keeps one remaining checkpoint directory from
    # serializing the entire tail of the transfer. Paths are disjoint.
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(migrate, tasks()))

if __name__ == '__main__':
    main()
