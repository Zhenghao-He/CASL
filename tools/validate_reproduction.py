"""CPU-only source validation; does not import GPU dependencies or create pyc."""
import ast
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    errors = []
    checked = {'python': 0, 'shell': 0}
    for relative in files:
        p = ROOT / relative
        if not p.is_file() or p.is_symlink():
            continue
        if p.suffix == '.py':
            try:
                ast.parse(p.read_bytes(), filename=relative)
                checked['python'] += 1
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append({'path': relative, 'error': str(exc)})
        elif p.suffix == '.sh':
            result = subprocess.run(['bash', '-n', str(p)], capture_output=True, text=True)
            checked['shell'] += 1
            if result.returncode:
                errors.append({'path': relative, 'error': result.stderr})
    report = {'checked': checked, 'errors': errors}
    print(json.dumps(report, indent=2))
    return bool(errors)

if __name__ == '__main__':
    raise SystemExit(main())
