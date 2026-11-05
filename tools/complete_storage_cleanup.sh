#!/usr/bin/env bash
# Wait for the explicitly supplied cleanup jobs, then validate and publish audit.
set -euo pipefail
cd "$(dirname "$0")/.."
for task_pid in "$@"; do
  while kill -0 "$task_pid" 2>/dev/null; do
    sleep 20
  done
done
/usr/bin/python3 tools/validate_reproduction.py > /bigtemp/wff7ad/h_SAE/audit/source_validation.json
/usr/bin/python3 tools/finalize_cleanup.py > /bigtemp/wff7ad/h_SAE/audit/finalization.log 2>&1
