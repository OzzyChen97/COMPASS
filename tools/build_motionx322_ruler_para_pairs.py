"""Phase 1 (Motion-X-322): 构建尺子重训/验收用的确定性 P1/P2/P3 改写对。

来源: dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl
（覆盖 train split，P1/P2/P3 各约 7 万条；不含 test）。val.txt 不在池覆盖内，
因此用确定性 held-out 切分: 排序后的 train ids 每 10 个取 1 个作为尺子验收集，
其余为尺子训练集。选择规则与 test 配方一致: sha256(f"{seed}:{motion}:{level}") % n。

用法:
    python tools/build_motionx322_ruler_para_pairs.py
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.build_humanml3d_ruler_para_pairs import (
    read_ids, select_pairs, write_json)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool",
                        default="dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl")
    parser.add_argument("--data-root", default="dataset/Motion-X-322")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--holdout-every", type=int, default=10)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_ids = sorted(read_ids(data_root / "train.txt"))
    test_ids = read_ids(data_root / "test.txt")
    pool_motions = set()
    with open(args.pool, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            pool_motions.add(str(json.loads(line).get("motion_id")))
    assert not (test_ids & pool_motions), "池覆盖了 test motion，停止"

    val_ruler_ids = set(train_ids[:: args.holdout_every])
    train_ruler_ids = set(train_ids) - val_ruler_ids
    print(f"train {len(train_ids)} -> train_ruler {len(train_ruler_ids)}, "
          f"val_ruler {len(val_ruler_ids)}")

    outputs = {
        "train": (train_ruler_ids, data_root / "train_ruler_perb_p123.json"),
        "val": (val_ruler_ids, data_root / "val_ruler_perb_p123.json"),
    }
    for name, (motion_ids, out_path) in outputs.items():
        rows, skipped, missing = select_pairs(args.pool, motion_ids, args.seed)
        write_json(out_path, rows)
        meta = {
            "name": name,
            "pool": str(Path(args.pool).resolve()),
            "output": str(out_path.resolve()),
            "seed": args.seed,
            "holdout_every": args.holdout_every,
            "motions": len(motion_ids),
            "pairs": len(rows),
            "level_counts": dict(Counter(r["level"] for r in rows)),
            "skipped": skipped,
            "missing_motion_levels": len(missing),
            "missing_examples": missing[:10],
        }
        meta_path = out_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()