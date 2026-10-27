#!/usr/bin/env bash
set -euo pipefail

repo_dir="data/CoachAI-Projects"

mkdir -p data
if [ ! -d "$repo_dir/.git" ]; then
  git clone --depth 1 --filter=blob:none --sparse https://github.com/wywyWang/CoachAI-Projects.git "$repo_dir"
fi
git -C "$repo_dir" sparse-checkout set ShuttleSet
git -C "$repo_dir" pull --ff-only
