"""Create the portable archive index and final cleanup audit after transfers finish."""
import json
from pathlib import Path
import shutil
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/bigtemp/wff7ad/h_SAE')

def main():
    raw = STORE / 'audit' / 'migration.jsonl'
    rows = [json.loads(line) for line in raw.read_text().splitlines()]
    unique = {row['source']: row for row in rows}
    portable = []
    for row in unique.values():
        source, target = Path(row['source']), Path(row['destination'])
        if not target.is_file() or target.stat().st_size != row['bytes']:
            raise RuntimeError(f'Archive size/existence check failed: {target}')
        if not source.is_symlink() or source.resolve() != target:
            raise RuntimeError(f'Original path is not connected to archive: {source}')
        portable.append({**row, 'archive_relative_path': str(target.relative_to(STORE))})
    for base in [ROOT, ROOT.parent / 'Asyrp_official', ROOT.parent / 'SwiftEdit', Path('/u/wff7ad/casl_mapping_ablation'), Path('/u/wff7ad/casl_cross_attribute_cache')]:
        for folder, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'reproduction'}]
            for name in files:
                p = Path(folder) / name
                if p.suffix.lower() in {'.jpg', '.png', '.jpeg', '.webp', '.txt', '.csv', '.json', '.patch'}:
                    continue
                if not p.is_symlink() and p.stat().st_size > 5 * 1024**2:
                    raise RuntimeError(f'Large artifact not yet migrated: {p}')
    image_log = STORE / 'audit' / 'image_archives.jsonl'
    images = [json.loads(line) for line in image_log.read_text().splitlines()] if image_log.exists() else []
    for rel in ['CelebA/images', 'CelebA-HQ/data1024', 'ffhq-dataset/images1024x1024', 'stargan-v2/data/train', 'stargan-v2/data/test', 'lsun/church', 'CelebA-HQ.tar', 'img_align_celeba.zip']:
        if (ROOT.parent / rel).exists():
            raise RuntimeError(f'Dataset deletion incomplete: {rel}')
    # These importance arrays are cheap to recompute from the retained DCI code.
    folder = ROOT / 'rebuttal_metrics/results/celeba_sae_omes_independent/dci_selection'
    removed_caches = []
    for p in folder.glob('importance_k*.npy'):
        if not p.is_symlink():
            p.unlink()
            removed_caches.append({'path': str(p), 'reason': 'Cheap DCI importance cache; evaluation code and metric CSVs retained'})
    if removed_caches:
        with (STORE / 'audit/deletions.jsonl').open('a') as log:
            log.write(''.join(json.dumps(row) + '\n' for row in removed_caches))
    out = ROOT / 'reproduction' / 'storage'
    out.mkdir(exist_ok=True)
    # Preserve pre-existing aliases too (e.g. shared SDXL reconstructions and
    # the 8-image inversion-cache alias), not just links created by migration.
    aliases = []
    for base in [ROOT, ROOT.parent / 'Asyrp_official']:
        for folder, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', 'reproduction'}]
            for name in dirs + files:
                p = Path(folder) / name
                if p.is_symlink() and str(p) not in unique:
                    aliases.append({'workspace_path': str(p.relative_to(ROOT.parent)), 'target': os.readlink(p)})
    (out / 'path_aliases.json').write_text(json.dumps(aliases, indent=2) + '\n')
    (out / 'artifact_index.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in portable))
    for name in ['deletions.jsonl', 'image_archives.jsonl', 'source_validation.json', 'dit_backbone.sha256']:
        if (STORE / 'audit' / name).exists():
            shutil.copy2(STORE / 'audit' / name, out / name)
    summary = {
        'archived_files': len(portable),
        'archived_file_bytes': sum(row['bytes'] for row in portable),
        'image_archives': len(images),
        'archived_images': sum(row['images'] for row in images),
        'image_archive_bytes': sum(row['bytes'] for row in images),
        'workspace_backbone_bytes': sum(p.stat().st_size for p in (STORE / 'workspace').glob('*') if p.is_file()),
        'preserved_path_aliases': len(aliases),
        'removed_dci_importance_caches': len(removed_caches),
        'datasets_removed': True,
        'storage_root': str(STORE),
        'validation': 'SHA-256 checked at migration; source symlinks and archive sizes rechecked during finalization; image tar files compared to sources before deletion',
    }
    # Interrupted transfers are disposable only after all source/archive checks.
    for temporary in STORE.rglob('*.transfer'):
        if temporary.is_file():
            temporary.unlink()
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    subprocess.run(['git', 'add', '-f', 'tools', 'reproduction/storage'], cwd=ROOT, check=True)
    subprocess.run(['git', 'add', '-u'], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '-m', 'Record verified artifact migration and dataset cleanup audit'], cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'origin', 'cleanup/reproducibility'], cwd=ROOT, check=True)

if __name__ == '__main__':
    main()
