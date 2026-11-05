"""Restore sibling source paths and optionally attach archived artifacts.

python tools/restore_layout.py --archive /bigtemp/wff7ad/h_SAE
Download datasets separately; this tool never replaces existing files.
"""
import argparse
import json
from pathlib import Path
import subprocess
import os

ROOT = Path(__file__).resolve().parents[1]

def link(source, destination):
    if destination.exists() or destination.is_symlink():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--images', action='store_true', help='Also extract verified image archives into this checkout')
    args = parser.parse_args()
    snapshots = ROOT / 'reproduction'
    for source in (snapshots / 'dependencies').iterdir():
        if source.is_dir():
            link(source, ROOT.parent / source.name)
    for source in (snapshots / 'third_party').iterdir():
        if source.is_dir():
            link(source, ROOT / 'third_party' / source.name)
    link(snapshots / 'celeba_classifier', ROOT / 'celeba_classifier')
    for source in snapshots.iterdir():
        if source.suffix in {'.py', '.sh'}:
            link(source, ROOT.parent / source.name)
    for domain in (snapshots / 'data_metadata').iterdir():
        for source in domain.iterdir():
            if source.is_file():
                link(source, ROOT.parent / domain.name / source.name)
    if args.archive:
        for source in (args.archive / 'workspace').glob('*'):
            link(source, ROOT.parent / source.name)
        manifest = args.archive / 'audit' / 'migration.jsonl'
        for line in manifest.read_text().splitlines():
            row = json.loads(line)
            archived = args.archive / Path(row['destination']).relative_to('/bigtemp/wff7ad/h_SAE')
            relative = archived.relative_to(args.archive)
            if relative.parts[0] == 'external':
                if relative.parts[1:3] == ('casl_mapping_ablation', 'checkpoints'):
                    link(archived, ROOT / 'checkpoint/delta_sae_ReLU_k4096_scale128_alpha32.0_normalizeFalse' / archived.name)
                else:
                    link(archived, ROOT / 'reproduction/external_artifacts' / Path(*relative.parts[1:]))
                continue
            if not archived.is_file():
                raise FileNotFoundError(archived)
            destination = (ROOT if relative.parts[0] == 'Asyrp_SAE' else ROOT.parent / relative.parts[0]) / Path(*relative.parts[1:])
            link(archived, destination)
        aliases = ROOT / 'reproduction/storage/path_aliases.json'
        if aliases.exists():
            for row in json.loads(aliases.read_text()):
                relative = Path(row['workspace_path'])
                destination = (ROOT if relative.parts[0] == 'Asyrp_SAE' else ROOT.parent / relative.parts[0]) / Path(*relative.parts[1:])
                target = Path(row['target'])
                if not target.is_absolute():
                    target = destination.parent / target
                elif str(target).startswith('/p/realai/zhenghao/LoRA_SAE/'):
                    rel = target.relative_to('/p/realai/zhenghao/LoRA_SAE')
                    target = (ROOT if rel.parts[0] == 'Asyrp_SAE' else ROOT.parent / rel.parts[0]) / Path(*rel.parts[1:])
                elif str(target).startswith('/bigtemp/wff7ad/h_SAE/'):
                    target = args.archive / target.relative_to('/bigtemp/wff7ad/h_SAE')
                elif str(target).startswith('/u/wff7ad/casl_'):
                    target = ROOT / 'reproduction/external_artifacts' / target.relative_to('/u/wff7ad')
                link(Path(os.path.normpath(target)), destination)
        if args.images:
            image_log = args.archive / 'audit' / 'image_archives.jsonl'
            for line in image_log.read_text().splitlines():
                row = json.loads(line)
                archive = args.archive / 'image_archives' / Path(row['archive']).name
                original = Path(row['original_root']).name
                destination = ROOT if original == 'Asyrp_SAE' else ROOT.parent / original
                destination.mkdir(exist_ok=True)
                subprocess.run(['tar', '--skip-old-files', '-xf', str(archive), '-C', str(destination)], check=True)
    print('Layout restored; existing files were preserved.')

if __name__ == '__main__':
    main()
