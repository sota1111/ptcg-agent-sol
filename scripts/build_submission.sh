#!/usr/bin/env bash
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
archive="$repo/submission.tar.gz"
for required in main.py deck.csv cg; do
  [ -e "$repo/$required" ] || { echo "missing required submission path: $required" >&2; exit 1; }
done
tar -C "$repo" -czf "$archive" --exclude='__pycache__' --exclude='*.pyc' main.py deck.csv cg
gzip -t "$archive"
listing="$(mktemp)"
trap 'rm -f -- "$listing"' EXIT
tar -tzf "$archive" > "$listing"
grep -Fx main.py "$listing" >/dev/null
grep -Fx deck.csv "$listing" >/dev/null
if grep -E '(^|/)(\.env($|\.)|\.git/|tests/|eval/|\.venv/|access_token|kaggle\.json|__pycache__/|.*\.pyc$)' "$listing"; then
  echo "submission contains a forbidden path" >&2
  exit 1
fi
echo "submission archive: $archive"
