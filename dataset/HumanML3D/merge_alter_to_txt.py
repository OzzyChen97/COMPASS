"""
Merge alter texts into HumanML3D training txt files.
Appends #alter_text as 7th field to each caption line.
Only processes train split.
"""

import json
import os
import codecs as cs
import shutil
from collections import defaultdict
from os.path import join as pjoin

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = pjoin(ROOT_DIR, "texts")
TRAIN_FILE = pjoin(ROOT_DIR, "train.txt")

# Load existing alter from preserving_pairs_full.jsonl
EXISTING_ALTER = pjoin(ROOT_DIR, "preserving_pairs_full.jsonl")
# Load new alter supplement (v2 includes action swap for all captions)
SUPPLEMENT = pjoin(ROOT_DIR, "train_alter_v2.json")
SUPPLEMENT_V1 = pjoin(ROOT_DIR, "train_alter_supplement.json")
# Backup dir
BACKUP_DIR = pjoin(ROOT_DIR, "texts_backup")


def load_alter_index():
    """Build (motion_id, text_orig) -> text_pair index."""
    index = {}

    # From existing pairs
    with open(EXISTING_ALTER, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            if p.get("relation") == "alter" and p.get("text_pair"):
                key = (p["motion_id"], p["text_orig"])
                index[key] = {
                    "text_pair": p["text_pair"],
                    "edit_type": p.get("edit_type", ""),
                }

    # From supplement v1 (direction/bodypart only)
    if os.path.exists(SUPPLEMENT_V1):
        with open(SUPPLEMENT_V1, "r", encoding="utf-8") as f:
            supplement_v1 = json.load(f)
        for item in supplement_v1:
            if item.get("text_pair"):
                key = (item["motion_id"], item["text_orig"])
                index[key] = {
                    "text_pair": item["text_pair"],
                    "edit_type": item.get("edit_type", ""),
                }

    # From supplement v2 (includes action swap, overwrites v1)
    if os.path.exists(SUPPLEMENT):
        with open(SUPPLEMENT, "r", encoding="utf-8") as f:
            supplement = json.load(f)
        for item in supplement:
            if item.get("text_pair"):
                key = (item["motion_id"], item["text_orig"])
                index[key] = {
                    "text_pair": item["text_pair"],
                    "edit_type": item.get("edit_type", ""),
                }

    return index


def main():
    alter_index = load_alter_index()
    print(f"Alter index: {len(alter_index)} entries")

    with open(TRAIN_FILE, "r") as f:
        train_ids = [l.strip() for l in f if l.strip()]
    print(f"Train motion_ids: {len(train_ids)}")

    # Stats
    total_lines = 0
    altered_lines = 0
    altered_mids = set()
    fps = 20

    for mid in train_ids:
        txt_path = pjoin(TEXT_DIR, mid + ".txt")
        if not os.path.exists(txt_path):
            continue

        # Read original lines
        with cs.open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []
        mid_altered = False
        for line in lines:
            line_stripped = line.rstrip("\n").rstrip("\r")
            parts = line_stripped.split("#")
            cap = parts[0].strip() if parts else ""
            f_tag = float(parts[2]) if len(parts) > 2 else 0
            to_tag = float(parts[3]) if len(parts) > 3 else 0

            total_lines += 1

            # Only add alter for full-range captions (f=0, to=0)
            if f_tag == 0.0 and to_tag == 0.0 and cap:
                key = (mid, cap)
                if key in alter_index:
                    alter_text = alter_index[key]["text_pair"]
                    edit_type = alter_index[key]["edit_type"]
                    # Append alter as 7th field
                    new_line = f"{line_stripped}#{alter_text}"
                    new_lines.append(new_line + "\n")
                    altered_lines += 1
                    mid_altered = True
                else:
                    # No alter, keep original
                    new_lines.append(line_stripped + "\n")
            else:
                # Sub-fragment or invalid, keep original
                new_lines.append(line_stripped + "\n")

        if mid_altered:
            altered_mids.add(mid)

        # Backup original and write new
        os.makedirs(BACKUP_DIR, exist_ok=True)
        backup_path = pjoin(BACKUP_DIR, mid + ".txt")
        if not os.path.exists(backup_path):
            shutil.copy2(txt_path, backup_path)

        with cs.open(txt_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    print(f"\nDone!")
    print(f"Total caption lines: {total_lines}")
    print(f"Lines with alter added: {altered_lines} ({altered_lines/total_lines*100:.1f}%)")
    print(f"Motion_ids with >=1 alter: {len(altered_mids)}/{len(train_ids)}")
    print(f"Original backups saved to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
