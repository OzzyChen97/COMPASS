"""Phase 1: 从已有 paraphrase pool 构建尺子重训/验收用的确定性 P1/P2/P3 对。

来源: dataset/HumanML3D/preserving_pairs_full.jsonl（仅含 train/val 覆盖的 motion，
不含 test，无泄漏）。每个 (motion, level) 用一个确定性候选，选择规则与
tools/finalize_sato_test_levels.py 的 test 配方一致: sha256(f"{seed}:{motion}:{level}") % n。

split 关系: train.txt 包含官方 val（24844 = 23384 + 1460），test 不相交。
输出两个文件，保证训练与验收不重叠：
- train_ruler_perb_p123.json: motion ∈ train.txt - val.txt（23384 × 3 levels）
- val_ruler_perb_p123.json:   motion ∈ val.txt（1460 × 3 levels）

用法:
    python tools/build_humanml3d_ruler_para_pairs.py
"""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LEVELS = ("P1", "P2", "P3")


def normalize_caption(caption):
    return re.sub(r"\s+", " ", str(caption).strip())


def basic_filter(original, candidate):
    """与 tools/finalize_sato_test_levels.py 相同的质量过滤。"""
    if not isinstance(candidate, str):
        return False
    candidate = normalize_caption(candidate)
    if not candidate or candidate.lower().rstrip(".") == str(original).lower().rstrip("."):
        return False
    original_len = len(normalize_caption(original).split())
    candidate_len = len(candidate.split())
    return max(1, int(original_len * 0.3)) <= candidate_len <= max(
        3, int(original_len * 3.0)
    )


def deterministic_choice(values, motion_id, level, seed):
    digest = hashlib.sha256(f"{seed}:{motion_id}:{level}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def read_ids(path):
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()}


def select_pairs(pool_path, motion_ids, seed):
    by_motion = defaultdict(lambda: defaultdict(list))
    skipped = Counter()
    with open(pool_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            motion_id = str(row.get("motion_id"))
            level = row.get("level")
            if level not in LEVELS:
                continue
            if row.get("relation") != "preserve":
                continue
            if motion_id not in motion_ids:
                continue
            text_orig = normalize_caption(row.get("text_orig"))
            text_pair = normalize_caption(row.get("text_pair"))
            if basic_filter(text_orig, text_pair):
                by_motion[motion_id][level].append(
                    {"text_orig": text_orig, "text_pair": text_pair})
            else:
                skipped["basic_filter"] += 1

    rows = []
    missing = []
    for motion_id in sorted(motion_ids):
        for level in LEVELS:
            candidates = sorted(by_motion[motion_id][level],
                                key=lambda r: (r["text_orig"], r["text_pair"]))
            if not candidates:
                missing.append({"motion_id": motion_id, "level": level})
                continue
            chosen = deterministic_choice(candidates, motion_id, level, seed)
            rows.append({
                "motion_id": motion_id,
                "text_orig": chosen["text_orig"],
                "text_pair": chosen["text_pair"],
                "relation": "preserve",
                "edit_type": "",
                "level": level,
            })
    for pair_id, row in enumerate(rows, start=1):
        row["pair_id"] = pair_id
    return rows, dict(skipped), missing


def write_json(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(rows, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default="dataset/HumanML3D/preserving_pairs_full.jsonl")
    parser.add_argument("--data-root", default="dataset/HumanML3D")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    train_ids = read_ids(data_root / "train.txt")
    val_ids = read_ids(data_root / "val.txt")
    test_ids = read_ids(data_root / "test.txt")
    assert not (test_ids & train_ids), "test 与 train 有交集，停止"

    outputs = {
        "train": (train_ids - val_ids, data_root / "train_ruler_perb_p123.json"),
        "val": (val_ids, data_root / "val_ruler_perb_p123.json"),
    }
    for name, (motion_ids, out_path) in outputs.items():
        rows, skipped, missing = select_pairs(args.pool, motion_ids, args.seed)
        write_json(out_path, rows)
        meta = {
            "name": name,
            "pool": str(Path(args.pool).resolve()),
            "output": str(out_path.resolve()),
            "seed": args.seed,
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