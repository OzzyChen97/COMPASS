#!/usr/bin/env python
"""Drop motions whose Motion-X caption is a ChatGPT meta-response (e.g.
"I apologize, but I cannot...", "The subject of the sentence is not specified")
rather than a motion description. Removes them from splits, manifest, the
train preserving-pairs file, the test level files, and the pair progress logs,
then updates dataset_meta and writes a report.
"""
import argparse
import json
import re
from pathlib import Path

META = re.compile(
    r"\b(i understand that|certainly[!,. ]|i.d be happy|i.m happy to|"
    r"please provide|please give|as an ai|i apologize|i.m sorry|i cannot|"
    r"the subject of the sentence|is a term used|assuming you meant|"
    r"is not a clear sentence|let me know if|feel free to|"
    r"here.s a (?:clearer|revised|complete|better|rephrased)|"
    r"to complete the sentence|provide me with|what is your input|valid input|"
    r"i can help you|i can add the subject|understood[.,]|acknowledged|"
    r"there is no subject|it is not possible|the subject is unclear|"
    r"it is unclear what|the given text is|i.m not sure what|"
    r"in the context of the scene|"
    # High-precision PURE-meta residual phrasings (no motion described).
    r"a subject is not (?:specified|clear|mentioned)|"
    r"the subject is not (?:specified|clear|mentioned)|"
    r"the subject is unclear|the individual.s actions are not clear|"
    r"the meaning of .* is not clear|not specified in the (?:input|given|sentence|phrase)|"
    r"it is not clear (?:what|who) subject|it depends on the context|"
    r"it is unclear who or what)", re.I)

TEST_LEVEL_SUFFIXES = ["_p1.json", "_p2.json", "_p3.json",
                       "_p123.json", "_mixed_human_ratio.json"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--test-prefix", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def caption_of(root, motion_id):
    line = (root / "texts" / f"{motion_id}.txt").read_text(
        encoding="utf-8", errors="ignore").splitlines()[0]
    return line.split("#")[0]


def rewrite_jsonl(path, drop, dry):
    kept, removed = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        if json.loads(line)["motion_id"] in drop:
            removed += 1
        else:
            kept.append(line)
    if not dry:
        path.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
    return removed


def rewrite_json_array(path, drop, dry):
    rows = json.loads(path.read_text())
    kept = [r for r in rows if r.get("motion_id") not in drop]
    removed = len(rows) - len(kept)
    if not dry:
        path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")
    return removed


def main():
    args = parse_args()
    root = args.dataset_root
    drop = {}
    for split in ("train", "test", "val"):
        for motion_id in [l.strip() for l in
                          (root / f"{split}.txt").read_text().splitlines() if l.strip()]:
            if META.search(caption_of(root, motion_id)):
                drop[motion_id] = {"split": split,
                                   "caption": caption_of(root, motion_id)}
    drop_ids = set(drop)

    new_counts = {}
    for split in ("train", "test", "val"):
        ids = [l.strip() for l in
               (root / f"{split}.txt").read_text().splitlines() if l.strip()]
        keep = [m for m in ids if m not in drop_ids]
        new_counts[split] = len(keep)
        if not args.dry_run:
            (root / f"{split}.txt").write_text("\n".join(keep) + "\n",
                                               encoding="utf-8")

    manifest_removed = rewrite_jsonl(root / "manifest.jsonl", drop_ids, args.dry_run)
    train_removed = rewrite_jsonl(args.train_pairs, drop_ids, args.dry_run)
    progress_removed = {}
    for name, prog in (("train", args.train_pairs),
                       ("test", Path(f"{args.test_prefix}.json"))):
        prog_path = prog.with_suffix(prog.suffix + ".progress.jsonl")
        # progress bundles hold multiple items; keep bundles but drop is applied
        # at finalize via valid keys, so we filter whole bundles conservatively.
        if prog_path.exists():
            progress_removed[name] = filter_progress(prog_path, drop_ids, args.dry_run)

    test_removed = {}
    for suffix in TEST_LEVEL_SUFFIXES:
        path = Path(f"{args.test_prefix}{suffix}")
        if path.exists():
            test_removed[suffix] = rewrite_json_array(path, drop_ids, args.dry_run)

    report = {
        "dropped": len(drop_ids),
        "dropped_by_split": {s: sum(v["split"] == s for v in drop.values())
                             for s in ("train", "test", "val")},
        "new_split_counts": new_counts,
        "manifest_removed": manifest_removed,
        "train_pair_rows_removed": train_removed,
        "test_rows_removed": test_removed,
        "progress_removed": progress_removed,
        "dropped_records": [{"motion_id": m, **v} for m, v in drop.items()],
    }
    if not args.dry_run:
        (root / "meta_caption_drop_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8")
        meta = json.loads((root / "dataset_meta.json").read_text())
        meta["split_counts"] = new_counts
        meta["meta_caption_dropped"] = len(drop_ids)
        (root / "dataset_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {k: v for k, v in report.items() if k != "dropped_records"}
    print(("DRY-RUN " if args.dry_run else "") + json.dumps(summary, indent=2))


def filter_progress(path, drop_ids, dry):
    """Rewrite a progress.jsonl dropping items that reference a dropped motion,
    re-indexing the surviving preserve/alter entries within each bundle."""
    kept_lines, removed = [], 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        bundle = json.loads(line)
        items = bundle.get("items", [])
        preserve = {int(r["index"]) - 1: r for r in bundle.get("preserve", [])
                    if "index" in r}
        alter = {int(r["index"]) - 1: r for r in bundle.get("alter", [])
                 if "index" in r}
        new_items, new_pre, new_alt = [], [], []
        for old_idx, item in enumerate(items):
            if item.get("motion_id") in drop_ids:
                removed += 1
                continue
            new_idx = len(new_items) + 1
            new_items.append(item)
            if old_idx in preserve:
                new_pre.append(dict(preserve[old_idx], index=new_idx))
            if old_idx in alter:
                new_alt.append(dict(alter[old_idx], index=new_idx))
        if new_items:
            kept_lines.append(json.dumps(
                {"items": new_items, "preserve": new_pre, "alter": new_alt},
                ensure_ascii=True))
    if not dry:
        path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")
    return removed


if __name__ == "__main__":
    main()
