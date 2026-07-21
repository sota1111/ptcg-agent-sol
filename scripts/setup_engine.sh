#!/usr/bin/env bash
# Install competition-use-only runtime assets locally; they remain gitignored.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
source_root="${SRC:-/workspaces/kaggle-ptcg-matsu/data/simulation/extracted}"
sample="$source_root/sample_submission/sample_submission"
[ -d "$sample/cg" ] || { echo "engine not found at $sample/cg; set SRC" >&2; exit 1; }
mkdir -p "$repo/cg" "$repo/data"
cp -R "$sample/cg/." "$repo/cg/"
cp "$source_root"/*_Card_Data.csv "$repo/data/"
echo "competition runtime installed in cg/ and data/ (gitignored)"
