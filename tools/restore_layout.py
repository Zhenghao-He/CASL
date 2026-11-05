"""Restore sibling source paths and optionally attach archived artifacts.

python tools/restore_layout.py --archive /bigtemp/wff7ad/h_SAE
Download datasets separately; this tool never replaces existing files.
"""
import argparse
import json
from pathlib import Path
import subprocess

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
                continue
            if not archived.is_file():
                raise FileNotFoundError(archived)
            destination = (ROOT if relative.parts[0] == 'Asyrp_SAE' else ROOT.parent / relative.parts[0]) / Path(*relative.parts[1:])
            link(archived, destination)
        if args.images:
            for archive in (args.archive / 'image_archives').glob('*.tar'):
                subprocess.run(['tar', '--skip-old-files', '-xf', str(archive), '-C', str(ROOT)], check=True)
    print('Layout restored; existing files were preserved.')

if __name__ == '__main__':
    main()
