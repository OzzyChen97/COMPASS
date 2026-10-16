#!/bin/bash
set -e
cd /mnt/workspace/Wilson/T2M-GPT

echo "=== Step 1: Generating alter v2 ==="
conda run -n T2M-GPT python3 dataset/HumanML3D/build_train_alter_v2.py

echo ""
echo "=== Step 2: Merging alter into txt files ==="
conda run -n T2M-GPT python3 dataset/HumanML3D/merge_alter_to_txt.py

echo ""
echo "=== Done! ==="
