import argparse
import json
import random
import re
from pathlib import Path


DIRECTION_PAIRS = [
    (r"\bforward\b", "backward", [r"\bbackward\b", r"\bbackwards\b"]),
    (r"\bfront\b", "back", [r"\bback\b", r"\bbackward\b", r"\bbackwards\b"]),
    (r"\bbackward\b", "forward", [r"\bforward\b", r"\bforwards\b"]),
    (r"\bbackwards\b", "forwards", [r"\bforward\b", r"\bforwards\b"]),
    (
        r"\bleft(?=\s+(?:turn|turns|turning|side|direction|diagonal|diagonally|circle|circles|curve|curving|sidestep|sidesteps|sideways))",
        "right",
        [r"\bright\b"],
    ),
    (
        r"\bright(?=\s+(?:turn|turns|turning|side|direction|diagonal|diagonally|circle|circles|curve|curving|sidestep|sidesteps|sideways))",
        "left",
        [r"\bleft\b"],
    ),
]

SPEED_PAIRS = [
    (r"\bslowly\b", "quickly", [r"\bquickly\b", r"\bquick\b", r"\bfast\b", r"\bfaster\b", r"\brapidly\b", r"\bbrisk\b"]),
    (r"\bslow\b", "fast", [r"\bquickly\b", r"\bquick\b", r"\bfast\b", r"\bfaster\b", r"\brapidly\b", r"\bbrisk\b"]),
    (r"\bslower\b", "faster", [r"\bquickly\b", r"\bquick\b", r"\bfast\b", r"\bfaster\b", r"\brapidly\b", r"\bbrisk\b"]),
    (r"\bquickly\b", "slowly", [r"\bslowly\b", r"\bslow\b", r"\bslower\b"]),
    (r"\bquick\b", "slow", [r"\bslowly\b", r"\bslow\b", r"\bslower\b"]),
    (r"\bfast\b", "slow", [r"\bslowly\b", r"\bslow\b", r"\bslower\b"]),
    (r"\bfaster\b", "slower", [r"\bslowly\b", r"\bslow\b", r"\bslower\b"]),
    (r"\brapidly\b", "slowly", [r"\bslowly\b", r"\bslow\b", r"\bslower\b"]),
    (r"\bbrisk\b", "slow", [r"\bslowly\b", r"\bslow\b", r"\bslower\b"]),
]

BODYPART_PAIRS = [
    (r"\bleft\s+(arm|hand|leg|foot|shoulder|elbow|knee)\b", r"right \1"),
    (r"\bright\s+(arm|hand|leg|foot|shoulder|elbow|knee)\b", r"left \1"),
]
BODYPART_ANY = re.compile(r"\b(?:left|right)\s+(?:arm|hand|leg|foot|shoulder|elbow|knee)\b", re.IGNORECASE)


def iter_json_records(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            for row in json.load(f):
                yield row
        else:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def text_lines_for_split(data_root, split):
    ids = [x.strip() for x in (data_root / f"{split}.txt").read_text().splitlines() if x.strip()]
    for motion_id in ids:
        text_path = data_root / "texts" / f"{motion_id}.txt"
        if not text_path.exists():
            continue
        for line in text_path.read_text(errors="ignore").splitlines():
            caption = line.split("#", 1)[0].strip()
            if caption:
                yield motion_id, caption


def has_conflict(caption, conflict_patterns):
    return any(re.search(pattern, caption, flags=re.IGNORECASE) for pattern in conflict_patterns)


def make_edit(caption, pairs):
    for pattern, repl, conflict_patterns in pairs:
        if has_conflict(caption, conflict_patterns):
            continue
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if not match:
            continue
        edited = re.sub(pattern, repl, caption, count=1, flags=re.IGNORECASE)
        if edited.lower() != caption.lower():
            return edited, pattern, repl, match.group(0)
    return None, None, None, None


def make_bodypart_edit(caption):
    all_mentions = BODYPART_ANY.findall(caption)
    if len(all_mentions) != 1:
        return None, None, None, None
    for pattern, repl in BODYPART_PAIRS:
        match = re.search(pattern, caption, flags=re.IGNORECASE)
        if not match:
            continue
        matched = match.group(0)
        side, part = matched.split(None, 1)
        other_side = "right" if side.lower() == "left" else "left"
        same_part_other_side = rf"\b{other_side}\s+{re.escape(part)}\b"
        if re.search(same_part_other_side, caption, flags=re.IGNORECASE):
            continue
        edited = re.sub(pattern, repl, caption, count=1, flags=re.IGNORECASE)
        if edited.lower() != caption.lower():
            return edited, pattern, repl, matched
    return None, None, None, None


def collect_constructed(data_root, split, edit_type):
    if edit_type not in {"direction", "speed", "bodypart"}:
        raise ValueError(edit_type)

    rows = []
    seen = set()
    for motion_id, caption in text_lines_for_split(data_root, split):
        if edit_type == "direction":
            edited, pattern, repl, matched = make_edit(caption, DIRECTION_PAIRS)
        elif edit_type == "speed":
            edited, pattern, repl, matched = make_edit(caption, SPEED_PAIRS)
        else:
            edited, pattern, repl, matched = make_bodypart_edit(caption)
        key = (caption.lower(), edited.lower() if edited else "")
        if edited and key not in seen:
            seen.add(key)
            rows.append(
                {
                    "relation": "alter",
                    "edit_type": edit_type,
                    "motion_id": motion_id,
                    "text_orig": caption,
                    "text_pair": edited,
                    "source": "constructed_from_test_caption",
                    "edit_rule": {"pattern": pattern, "replacement": repl, "matched": matched},
                }
            )
    return rows


def collect_preserve(data_root, path):
    test_ids = set(x.strip() for x in (data_root / "test.txt").read_text().splitlines() if x.strip())
    rows = []
    seen = set()
    for row in iter_json_records(path):
        motion_id = str(row.get("motion_id", ""))
        if motion_id not in test_ids:
            continue
        orig = row.get("text_orig")
        pair = row.get("text_pair")
        key = (orig, pair)
        if row.get("relation") == "preserve" and orig and pair and key not in seen:
            seen.add(key)
            rows.append(
                {
                    "relation": "preserve",
                    "edit_type": "preserve",
                    "motion_id": motion_id,
                    "text_orig": orig,
                    "text_pair": pair,
                    "source": str(path),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("dataset/HumanML3D"))
    parser.add_argument("--preserve-path", type=Path, default=Path("dataset/HumanML3D/test_perb_deterministic.json"))
    parser.add_argument("--out", type=Path, default=Path("ablation/vlm_judge/pairs_20_each.jsonl"))
    parser.add_argument("--per-type", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    buckets = {
        "preserve": collect_preserve(args.data_root, args.preserve_path),
        "direction": collect_constructed(args.data_root, "test", "direction"),
        "speed": collect_constructed(args.data_root, "test", "speed"),
        "bodypart": collect_constructed(args.data_root, "test", "bodypart"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for edit_type, candidates in buckets.items():
        rng.shuffle(candidates)
        chosen = candidates[: args.per_type]
        if len(chosen) < args.per_type:
            raise RuntimeError(f"{edit_type} has only {len(chosen)} candidates")
        for idx, row in enumerate(chosen):
            row = dict(row)
            row["pair_uid"] = f"{edit_type}_{idx:03d}"
            rows.append(row)
        print(f"{edit_type}: selected {len(chosen)} from {len(candidates)} candidates")

    with open(args.out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
