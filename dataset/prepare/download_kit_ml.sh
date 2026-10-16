#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/mnt/workspace/Wilson/miniconda3/envs/T2M-GPT/bin/python}"
DEST="${KIT_ML_DEST:-dataset/KIT-ML}"
SOURCE_URL="${KIT_ML_SOURCE_URL:-https://drive.google.com/drive/folders/1D3bf2G2o4Hv-Ale26YW18r1Wrh7oIAwK}"

if [[ -d "$DEST" ]] && find "$DEST" -mindepth 1 -print -quit | grep -q .; then
  echo "Refusing to overwrite non-empty KIT-ML directory: $DEST" >&2
  exit 1
fi

mkdir -p "$DEST"
"$PYTHON" -m gdown --folder --remaining-ok "$SOURCE_URL" -O "$DEST"

if [[ ! -f "$DEST/train.txt" ]]; then
  nested_root="$(find "$DEST" -type f -name train.txt -printf '%h\n' | head -n 1)"
  if [[ -n "$nested_root" ]]; then
    echo "KIT-ML was downloaded under nested directory: $nested_root"
    echo "Set KIT_DATA_ROOT=$nested_root when running the experiment."
  else
    echo "Download completed but no KIT-ML train.txt was found under $DEST" >&2
    exit 1
  fi
fi
