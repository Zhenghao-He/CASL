"""Capture locally modified dependency source without nested Git repositories."""
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {'.py', '.sh', '.md', '.txt', '.yaml', '.yml', '.json', '.toml', '.cfg', '.ini', '.csv', '.ipynb', '.cu', '.cpp', '.h', '.c', '.html', '.css', '.js'}
SKIP = {'.git', '__pycache__', 'node_modules', 'data', 'images', 'images1024x1024', 'checkpoint', 'checkpoints', 'pretrained', 'precomputed', 'runs', 'logs', 'saved_info', 'weights', 'results', 'artifacts', 'test_images', '.venv'}

def git(path, *args):
    return subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True).stdout.strip()

def copy_source(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        if p.name in SKIP or p.is_symlink():
            continue
        if p.is_dir():
            copy_source(p, dst / p.name)
        elif (p.suffix.lower() in EXTENSIONS or p.name.lower().startswith(('license', 'copying'))) and p.stat().st_size <= 5 * 1024**2:
            shutil.copy2(p, dst / p.name)

def main():
    provenance = []
    sources = []
    for name in ['Asyrp_official', 'DiT', 'MasaCtrl', 'SwiftEdit', 'sliders', 'Diffusion-Pullback', 'sparse_autoencoder', 'stargan-v2', 'ffhq-dataset', 'CelebFaces_Attributes_Classification', 'augclip_eval', 'revelio']:
        sources.append((ROOT.parent / name, ROOT / 'reproduction' / 'dependencies' / name))
    for src in (ROOT / 'third_party').iterdir():
        if src.is_dir():
            sources.append((src, ROOT / 'reproduction' / 'third_party' / src.name))
    sources.append((ROOT / 'celeba_classifier', ROOT / 'reproduction' / 'celeba_classifier'))
    for src, dst in sources:
        copy_source(src, dst)
        patch = git(src, 'diff', '--binary', 'HEAD')
        (dst / 'LOCAL_CHANGES.patch').write_text(patch)
        provenance.append({'name': src.name, 'original_path': str(src), 'snapshot': str(dst.relative_to(ROOT)), 'commit': git(src, 'rev-parse', 'HEAD'), 'remotes': git(src, 'remote', '-v'), 'status': git(src, 'status', '--short')})
    for src in ROOT.parent.iterdir():
        if src.is_file() and src.suffix in {'.py', '.sh'}:
            shutil.copy2(src, ROOT / 'reproduction' / src.name)
    metadata = ROOT / 'reproduction' / 'data_metadata'
    for name in ['CelebA-HQ', 'CelebA']:
        src = ROOT.parent / name
        if src.exists():
            copy_source(src, metadata / name)
    (ROOT / 'reproduction' / 'dependencies.json').write_text(json.dumps(provenance, indent=2))

if __name__ == '__main__':
    main()
