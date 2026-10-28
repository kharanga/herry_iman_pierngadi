#!/usr/bin/env bash
set -euo pipefail

repo_dir="data/third_party/TrackNetV3"
repo_url="https://github.com/qaz812345/TrackNetV3.git"
repo_commit="6eda442ada1740573f200f836d93edc9a541ee86"
checkpoint_dir="data/models/tracknetv3"
checkpoint_zip_url="https://drive.usercontent.google.com/download?id=1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA&export=download&confirm=t"
tracknet_sha256="df867641a02712b021f04548ff4b1208ddfdb47f629ab2094ceb978667e83b1a"
inpaintnet_sha256="5749b66b8002f3ad9e0af841604004706fc796df30599e6bf01952696009688c"

mkdir -p data/third_party "$checkpoint_dir"
if [ ! -d "$repo_dir/.git" ]; then
  git clone "$repo_url" "$repo_dir"
fi
git -C "$repo_dir" fetch --quiet origin "$repo_commit" || true
git -C "$repo_dir" checkout --quiet "$repo_commit"

checkpoint_is_valid() {
  [ -f "$1" ] && [ "$(shasum -a 256 "$1" | cut -d ' ' -f 1)" = "$2" ]
}

fail_download() {
  echo "error: $1" >&2
  echo "The TrackNetV3 checkpoint download from Google Drive failed. Drive often returns an HTML page" >&2
  echo "instead of the file (daily download quota exceeded, or its virus-scan warning for large files)." >&2
  echo "Download the zip by hand and run this script again:" >&2
  echo "  1. Open https://drive.google.com/file/d/1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA/view in a browser and download it." >&2
  echo "  2. Save it as $manual_zip (replacing any file already there)" >&2
  echo "Nothing was changed in $checkpoint_dir." >&2
  exit 1
}

describe_non_zip() {
  if head -c 512 "$1" | grep -qi -e '<!doctype html' -e '<html'; then
    echo "$2 is an HTML page, not a zip"
  else
    echo "$2 is not a valid zip ($(wc -c < "$1" | tr -d ' ') bytes)"
  fi
}

manual_zip="$checkpoint_dir/TrackNetV3_ckpts.zip"

if ! checkpoint_is_valid "$checkpoint_dir/TrackNet_best.pt" "$tracknet_sha256" \
  || ! checkpoint_is_valid "$checkpoint_dir/InpaintNet_best.pt" "$inpaintnet_sha256"; then
  work_dir="$(mktemp -d "$checkpoint_dir/.download.XXXXXX")"
  trap 'rm -rf "$work_dir"' EXIT
  if [ -f "$manual_zip" ]; then
    zip_file="$manual_zip"
    zip_label="$manual_zip"
    echo "using $manual_zip"
  else
    zip_file="$work_dir/TrackNetV3_ckpts.zip"
    zip_label="the file downloaded from Google Drive"
    curl --fail --location --silent --show-error -o "$zip_file" "$checkpoint_zip_url" \
      || fail_download "curl could not download the zip"
  fi
  unzip -tq "$zip_file" > /dev/null 2>&1 || fail_download "$(describe_non_zip "$zip_file" "$zip_label")"
  unzip -o -j -q "$zip_file" 'ckpts/TrackNet_best.pt' 'ckpts/InpaintNet_best.pt' -d "$work_dir/ckpts" \
    || fail_download "the zip does not contain ckpts/TrackNet_best.pt and ckpts/InpaintNet_best.pt"
  checkpoint_is_valid "$work_dir/ckpts/TrackNet_best.pt" "$tracknet_sha256" \
    && checkpoint_is_valid "$work_dir/ckpts/InpaintNet_best.pt" "$inpaintnet_sha256" \
    || fail_download "the checkpoints in the zip do not match the expected SHA-256 checksums"
  mv "$work_dir/ckpts/TrackNet_best.pt" "$work_dir/ckpts/InpaintNet_best.pt" "$checkpoint_dir/"
  if [ "$zip_file" = "$manual_zip" ]; then
    rm "$manual_zip"
  fi
fi

checkpoint_is_valid "$checkpoint_dir/TrackNet_best.pt" "$tracknet_sha256" || { echo "TrackNet_best.pt checksum mismatch" >&2; exit 1; }
checkpoint_is_valid "$checkpoint_dir/InpaintNet_best.pt" "$inpaintnet_sha256" || { echo "InpaintNet_best.pt checksum mismatch" >&2; exit 1; }
echo "TrackNetV3 code in $repo_dir at $repo_commit, checkpoints in $checkpoint_dir"
