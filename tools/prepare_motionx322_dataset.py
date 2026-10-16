#!/usr/bin/env python
"""Build a native 322D Motion-X dataset without changing motion arrays."""

import argparse
import hashlib
import io
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


MOTION_PREFIX = "motion_data/smplx_322/"
TEXT_PREFIX = "motionx_seq_text_v1.1/"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-zip", type=Path, required=True)
    parser.add_argument("--text-zip", type=Path, required=True)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/Motion-X-322"))
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--std-floor", type=float, default=1e-8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def canonical(value):
    value = str(value).replace("\\", "/").strip("/")
    value = re.sub(r"/+", "/", value)
    return str(PurePosixPath(value).with_suffix("")).lower()


def source_key(member, prefix):
    value = member.replace("\\", "/")
    if not value.startswith(prefix):
        return None
    return canonical(value[len(prefix):])


def read_official_splits(split_root):
    memberships = defaultdict(set)
    raw_counts = {}
    for split in ("train", "val", "test"):
        values = [
            canonical(line)
            for line in (split_root / f"{split}.txt").read_text().splitlines()
            if line.strip()
        ]
        raw_counts[split] = len(values)
        for value in values:
            memberships[value].add(split)

    # The released lists contain duplicates. Evaluation sets take precedence.
    priority = {"train": 0, "val": 1, "test": 2}
    owners = {
        value: max(splits, key=priority.__getitem__)
        for value, splits in memberships.items()
    }
    return owners, {
        "raw_counts": raw_counts,
        "unique_paths": len(memberships),
        "cross_split_duplicates": sum(len(splits) > 1 for splits in memberships.values()),
        "priority": ["test", "val", "train"],
    }


def stable_id(key):
    return "mx322_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]


def tokenize_caption(nlp, caption):
    tokens = []
    for token in nlp(caption):
        if token.is_space or token.is_punct:
            continue
        word = token.text.lower()
        if token.pos_ in {"NOUN", "VERB"} and word != "left" and token.lemma_:
            word = token.lemma_.lower()
        tokens.append(f"{word}/{token.pos_ or 'OTHER'}")
    return tokens or ["unk/OTHER"]


def load_caption(text_zip, member):
    raw = text_zip.read(member).decode("utf-8", errors="replace")
    for line in raw.splitlines():
        caption = re.sub(r"\s+", " ", line.strip())
        if caption:
            return caption
    return None


def main():
    args = parse_args()
    for path in (args.motion_zip, args.text_zip):
        if not path.is_file():
            raise FileNotFoundError(path)
    owners, split_audit = read_official_splits(args.split_root)
    owners = {key: split for key, split in owners.items() if not key.startswith("humanml/")}

    try:
        import spacy
        nlp = spacy.load(args.spacy_model, disable=["parser", "ner"])
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"spaCy model {args.spacy_model!r} is required") from exc

    output = args.output
    motion_dir = output / "new_joint_vecs"
    text_dir = output / "texts"
    motion_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.motion_zip) as motion_zip, zipfile.ZipFile(args.text_zip) as text_zip:
        motion_members = {
            key: info.filename
            for info in motion_zip.infolist()
            if info.filename.endswith(".npy")
            for key in [source_key(info.filename, MOTION_PREFIX)]
            if key is not None
        }
        text_members = {
            key: info.filename
            for info in text_zip.infolist()
            if info.filename.endswith(".txt")
            for key in [source_key(info.filename, TEXT_PREFIX)]
            if key is not None
        }
        selected = sorted(set(owners) & set(motion_members) & set(text_members))
        if args.limit:
            selected = selected[: args.limit]
        if not selected:
            raise RuntimeError("No exact motion/text/official-split intersections")

        counts = Counter()
        lengths = []
        feature_sum = np.zeros(322, dtype=np.float64)
        feature_sq_sum = np.zeros(322, dtype=np.float64)
        training_frames = 0
        manifest_path = output / "manifest.jsonl"
        manifest_rows = []
        for index, key in enumerate(selected, start=1):
            motion_bytes = motion_zip.read(motion_members[key])
            motion = np.load(io.BytesIO(motion_bytes), allow_pickle=False)
            if motion.ndim != 2 or motion.shape[1] != 322 or motion.shape[0] < 1:
                raise ValueError(f"Invalid native Motion-X array {key}: {motion.shape}")
            if motion.dtype != np.float32:
                raise ValueError(f"Expected float32 native Motion-X array {key}, got {motion.dtype}")
            if not np.isfinite(motion).all():
                raise ValueError(f"Non-finite native Motion-X array {key}")
            caption = load_caption(text_zip, text_members[key])
            if not caption:
                raise ValueError(f"Empty caption for {key}")

            motion_id = stable_id(key)
            motion_path = motion_dir / f"{motion_id}.npy"
            text_path = text_dir / f"{motion_id}.txt"
            if args.overwrite or not motion_path.exists():
                motion_path.write_bytes(motion_bytes)
            token_text = " ".join(tokenize_caption(nlp, caption))
            text_path.write_text(f"{caption}#{token_text}#0.0#0.0\n", encoding="utf-8")

            split = owners[key]
            counts[split] += 1
            lengths.append(int(motion.shape[0]))
            if split == "train":
                values = motion.astype(np.float64, copy=False)
                feature_sum += values.sum(axis=0)
                feature_sq_sum += np.square(values).sum(axis=0)
                training_frames += len(values)
            manifest_rows.append({
                "motion_id": motion_id,
                "source_path": key,
                "motion_member": motion_members[key],
                "text_member": text_members[key],
                "category": key.split("/", 1)[0],
                "split": split,
                "frames": int(motion.shape[0]),
                "dim": 322,
                "fps": 30,
                "caption": caption,
                "sha256": hashlib.sha256(motion_bytes).hexdigest(),
            })
            if index % 2000 == 0:
                print(f"Prepared {index}/{len(selected)}", flush=True)

    if training_frames == 0:
        raise RuntimeError("No training frames for normalization")
    mean = feature_sum / training_frames
    variance = np.maximum(feature_sq_sum / training_frames - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    floored = np.flatnonzero(raw_std < args.std_floor).tolist()
    std = np.maximum(raw_std, args.std_floor)
    np.save(output / "Mean.npy", mean.astype(np.float32))
    np.save(output / "Std.npy", std.astype(np.float32))
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    for split in ("train", "val", "test"):
        ids = [row["motion_id"] for row in manifest_rows if row["split"] == split]
        (output / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")

    all_lengths = np.asarray(lengths, dtype=np.int64)
    metadata = {
        "dataset": "motionx322",
        "representation": "smplx322",
        "dim": 322,
        "fps": 30,
        "motions_unchanged": True,
        "motion_archive": str(args.motion_zip.resolve()),
        "text_archive": str(args.text_zip.resolve()),
        "selected_exact_intersection": len(selected),
        "split_counts": dict(counts),
        "excluded_humanml_paths": sum(key.startswith("humanml/") for key in read_official_splits(args.split_root)[0]),
        "motion_members": len(motion_members),
        "text_members": len(text_members),
        "official_non_humanml_paths": len(owners),
        "missing_motion": len(set(owners) - set(motion_members)),
        "missing_text": len(set(owners) - set(text_members)),
        "min_frames": int(all_lengths.min()),
        "max_frames": int(all_lengths.max()),
        "median_frames": float(np.median(all_lengths)),
        "p95_frames": float(np.percentile(all_lengths, 95)),
        "training_frames": training_frames,
        "std_floor": args.std_floor,
        "std_floored_channels": floored,
        "official_split_audit": split_audit,
    }
    (output / "dataset_meta.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
