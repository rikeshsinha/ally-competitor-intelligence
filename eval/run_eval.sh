#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v promptfoo >/dev/null 2>&1; then
  echo "Installing promptfoo..."
  pip install promptfoo >/dev/null
fi
promptfoo eval -c test_config.promptfoo.yml --output results/
echo "Done. View with: promptfoo view results/"
