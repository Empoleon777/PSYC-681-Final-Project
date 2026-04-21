#!/usr/bin/env bash
set -euo pipefail

mkdir -p Data/sources

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    git -C "$dir" pull --ff-only
  else
    git clone --depth 1 "$url" "$dir"
  fi
}

clone_or_update "https://github.com/LST1836/MITweet.git" "Data/sources/MITweet"
clone_or_update "https://github.com/valentinhofmann/politosphere.git" "Data/sources/politosphere"
clone_or_update "https://github.com/arnestc/political-compass.git" "Data/sources/political-compass"

echo "Source repos synced under Data/sources/"
echo "POLITISKY24 lives on Zenodo: https://zenodo.org/records/15616911"
