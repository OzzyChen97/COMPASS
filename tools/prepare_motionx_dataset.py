#!/usr/bin/env python
"""Convert Motion-X++ joint archives into a HumanML3D-compatible dataset."""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import sys

import numpy as np
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.humanml_preprocess import humanml_mean_std, process_positions, target_offsets


CATEGORY_ALIASES = {
    "animation": ("animation", "game_motion"),
    "perform": ("perform", "fitness"),
    "music": ("music", "aist"),
    "haa500": ("haa500",),
    "humman": ("humman",),
    "idea400": ("idea400",),
    "kungfu": ("kungfu",),
}
KNOWN_CATEGORIES = {item for values in CATEGORY_ALIASES.values() for item in values}
WORKER_OFFSETS = None
WORKER_OUTPUT = None
WORKER_SOURCE_FPS = 30
WORKER_TARGET_FPS = 20
WORKER_MAX_FRAMES = 196
WORKER_MIN_FRAMES = 40
WORKER_SOURCE_UP_AXIS = "z"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("dataset/Motion-X/raw/bypy"))
    parser.add_argument("--joint-root", type=Path, default=None)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("dataset/Motion-X"))
    parser.add_argument(
        "--reference-joints",
        type=Path,
        default=Path("dataset/HumanML3D/new_joints/000021.npy"),
    )
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--target-fps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=196)
    parser.add_argument("--min-frames", type=int, default=40)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--min-split-match", type=float, default=0.98)
    parser.add_argument("--min-text-match", type=float, default=0.98)
    parser.add_argument("--max-errors", type=int, default=0)
    parser.add_argument("--source-up-axis", choices=("y", "z"), default="z")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalized_path(value):
    value = str(value).replace("\\", "/").strip("/")
    value = re.sub(r"/+/", "/", value)
    return str(PurePosixPath(value).with_suffix("")).lower()


def member_suffix(member, archive_category):
    parts = list(PurePosixPath(normalized_path(member)).parts)
    while parts and parts[0] in {
        "motion", "motion_generation", "new_joint", "text", "semantic_label",
        archive_category,
    }:
        parts.pop(0)
    for index, part in enumerate(parts):
        if part in KNOWN_CATEGORIES:
            parts = parts[index + 1:]
            break
    return "/".join(parts)


def exact_member_suffix(member, archive_category):
    value = str(member).replace("\\", "/").strip("/")
    value = re.sub(r"/+/", "/", value)
    parts = list(PurePosixPath(value).with_suffix("").parts)
    while parts and parts[0].lower() in {
        "motion", "motion_generation", "new_joint", "text", "semantic_label",
        archive_category.lower(),
    }:
        parts.pop(0)
    for index, part in enumerate(parts):
        if part.lower() in KNOWN_CATEGORIES:
            parts = parts[index + 1:]
            break
    return "/".join(parts)


def key_candidates(member, archive_category):
    suffix = member_suffix(member, archive_category)
    aliases = CATEGORY_ALIASES.get(archive_category, (archive_category,))
    return [f"{alias}/{suffix}" for alias in aliases if suffix]


def text_key_candidates(member, archive_category):
    suffix = exact_member_suffix(member, archive_category)
    aliases = CATEGORY_ALIASES.get(archive_category, (archive_category,))
    return [f"{alias}/{suffix}" for alias in aliases if suffix]


def canonical_motion_name(value):
    return re.sub(r"[^a-z0-9]", "", PurePosixPath(normalized_path(value)).name)


def processed_source_candidates(member, archive_category):
    suffix = member_suffix(member, archive_category)
    stem = PurePosixPath(normalized_path(suffix)).name
    original = stem
    while re.search(r"_clip_?\d+$", stem, flags=re.IGNORECASE):
        stem = re.sub(r"_clip_?\d+$", "", stem, flags=re.IGNORECASE)

    candidates = []
    numbered = re.match(r"^(.*)_(\d+)$", stem)
    if numbered:
        candidates.append(canonical_motion_name(
            f"{numbered.group(1)}_clip_{numbered.group(2)}"
        ))
    candidates.append(canonical_motion_name(stem))
    if numbered:
        candidates.append(canonical_motion_name(numbered.group(1)))
    candidates.append(canonical_motion_name(original))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def build_split_name_index(split_map):
    index = defaultdict(list)
    for source_key, split in split_map.items():
        category = PurePosixPath(source_key).parts[0].lower()
        index[(category, canonical_motion_name(source_key))].append((source_key, split))
    return index


def choose_group_split(member, archive_category, split_name_index):
    priority = {"train": 0, "val": 1, "test": 2}
    candidates = processed_source_candidates(member, archive_category)
    for candidate in candidates:
        matches = split_name_index.get((archive_category, candidate), [])
        if matches:
            split = max((item[1] for item in matches), key=priority.get)
            return split, "official", candidate, len(matches)

    group_key = f"{archive_category}/{candidates[0]}"
    value = int.from_bytes(hashlib.sha1(group_key.encode("utf-8")).digest()[:8], "big")
    fraction = value / float(1 << 64)
    if fraction < 0.80:
        split = "train"
    elif fraction < 0.85:
        split = "val"
    else:
        split = "test"
    return split, "hash_fallback", candidates[0], 0


def read_splits(split_root):
    values_by_split = {}
    counts = {}
    for split in ("train", "val", "test"):
        path = split_root / f"{split}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        values = [normalized_path(line) for line in path.read_text().splitlines() if line.strip()]
        values_by_split[split] = values
        counts[split] = len(values)

    # A few official Motion-X paths occur in more than one split. Preserve the
    # evaluation sets and remove those paths from lower-priority splits.
    unique_by_split = {
        split: list(dict.fromkeys(values))
        for split, values in values_by_split.items()
    }
    mapping = {}
    dropped = Counter()
    memberships = defaultdict(set)
    for split, values in unique_by_split.items():
        for value in values:
            memberships[value].add(split)
    for split in ("test", "val", "train"):
        for value in unique_by_split[split]:
            if value in mapping:
                if mapping[value] != split:
                    dropped[split] += 1
                continue
            mapping[value] = split

    overlaps = {
        value: sorted(splits)
        for value, splits in memberships.items()
        if len(splits) > 1
    }
    split_deduplication = {
        "priority": ["test", "val", "train"],
        "cross_split_paths": len(overlaps),
        "unique_counts": {
            split: len(values) for split, values in unique_by_split.items()
        },
        "within_split_duplicate_entries": {
            split: counts[split] - len(unique_by_split[split])
            for split in ("train", "val", "test")
        },
        "dropped_unique_paths_from_lower_priority": dict(dropped),
        "effective_counts": {
            split: sum(owner == split for owner in mapping.values())
            for split in ("train", "val", "test")
        },
        "overlap_examples": dict(list(sorted(overlaps.items()))[:50]),
    }
    return mapping, counts, split_deduplication


def parse_captions(payload):
    text = payload.decode("utf-8", errors="ignore").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            values = [parsed]
        elif isinstance(parsed, list):
            values = [str(item) for item in parsed]
        elif isinstance(parsed, dict):
            values = []
            for key in ("caption", "text", "semantic_label", "description"):
                if key in parsed:
                    item = parsed[key]
                    values.extend(item if isinstance(item, list) else [item])
                    break
        else:
            values = []
    except json.JSONDecodeError:
        values = text.splitlines()
    captions = []
    for value in values:
        caption = re.sub(r"\s+", " ", str(value)).strip().replace("#", " ")
        if caption and caption not in captions:
            captions.append(caption)
    return captions


def build_text_index(text_archives):
    index = defaultdict(list)
    for archive in text_archives:
        category = archive.stem.lower()
        with zipfile.ZipFile(archive) as handle:
            for member in handle.namelist():
                if member.endswith("/"):
                    continue
                captions = parse_captions(handle.read(member))
                if not captions:
                    continue
                for candidate in text_key_candidates(member, category):
                    for caption in captions:
                        if caption not in index[candidate]:
                            index[candidate].append(caption)
    return index


def z_up_to_y_up(joints):
    converted = np.asarray(joints)[..., [0, 2, 1]].copy()
    converted[..., 2] *= -1
    return converted


def init_worker(
    reference_path, output_dir, source_fps, target_fps, max_frames, min_frames,
    source_up_axis,
):
    global WORKER_OFFSETS, WORKER_OUTPUT, WORKER_SOURCE_FPS, WORKER_TARGET_FPS
    global WORKER_MAX_FRAMES, WORKER_MIN_FRAMES, WORKER_SOURCE_UP_AXIS
    reference = np.load(reference_path)
    WORKER_OFFSETS = target_offsets(reference)
    WORKER_OUTPUT = Path(output_dir)
    WORKER_SOURCE_FPS = source_fps
    WORKER_TARGET_FPS = target_fps
    WORKER_MAX_FRAMES = max_frames
    WORKER_MIN_FRAMES = min_frames
    WORKER_SOURCE_UP_AXIS = source_up_axis


def stable_motion_id(source_key):
    return "mx_" + hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:16]


def process_task(task):
    joint_path, source_key, split, captions, overwrite = task
    base_id = stable_motion_id(source_key)
    try:
        joints = np.load(joint_path, allow_pickle=False)
        joints = np.asarray(joints)
        if joints.ndim != 3 or joints.shape[1] < 22 or joints.shape[2] != 3:
            raise ValueError(f"unexpected joint shape {joints.shape}")
        joints = joints[:, :22].astype(np.float32)
        if WORKER_SOURCE_UP_AXIS == "z":
            joints = z_up_to_y_up(joints)
        if WORKER_SOURCE_FPS != WORKER_TARGET_FPS:
            divisor = math.gcd(WORKER_SOURCE_FPS, WORKER_TARGET_FPS)
            joints = resample_poly(
                joints,
                WORKER_TARGET_FPS // divisor,
                WORKER_SOURCE_FPS // divisor,
                axis=0,
            ).astype(np.float32)
        features = process_positions(joints, WORKER_OFFSETS)
        if len(features) < WORKER_MIN_FRAMES:
            return {"status": "too_short", "source_key": source_key}

        chunk_count = max(1, math.ceil(len(features) / WORKER_MAX_FRAMES))
        chunks = np.array_split(features, chunk_count)
        if min(map(len, chunks)) < WORKER_MIN_FRAMES:
            chunk_count = max(1, chunk_count - 1)
            chunks = np.array_split(features, chunk_count)

        records = []
        for index, chunk in enumerate(chunks):
            if len(chunk) < WORKER_MIN_FRAMES or len(chunk) > WORKER_MAX_FRAMES:
                raise ValueError(f"invalid chunk length {len(chunk)}")
            motion_id = base_id if len(chunks) == 1 else f"{base_id}_c{index:03d}"
            path = WORKER_OUTPUT / "new_joint_vecs" / f"{motion_id}.npy"
            if overwrite or not path.exists():
                np.save(path, chunk.astype(np.float32))
            records.append({
                "motion_id": motion_id,
                "source_key": source_key,
                "split": split,
                "frames": len(chunk),
                "captions": captions,
            })
        return {"status": "ok", "records": records}
    except Exception as exc:
        return {"status": "error", "source_key": source_key, "error": repr(exc)}


def tokenize_caption(nlp, caption):
    tokens = []
    for token in nlp(caption):
        if not token.text.isalpha():
            continue
        word = token.text.lower()
        if token.pos_ in {"NOUN", "VERB"} and word != "left" and token.lemma_:
            word = token.lemma_.lower()
        tokens.append(f"{word}/{token.pos_ or 'OTHER'}")
    return tokens or ["unk/OTHER"]


def remove_stale_generated_files(output, valid_ids):
    removed = {"motions": 0, "texts": 0}
    for directory, suffix, key in (
        (output / "new_joint_vecs", ".npy", "motions"),
        (output / "texts", ".txt", "texts"),
    ):
        if not directory.exists():
            continue
        for path in directory.glob(f"*{suffix}"):
            if path.stem not in valid_ids:
                path.unlink()
                removed[key] += 1
    return removed


def main():
    args = parse_args()
    motion_dir = args.joint_root or (args.raw_root / "motion/motion_generation/new_joint")
    text_dir = args.raw_root / "text/semantic_label"
    text_archives = sorted(text_dir.glob("*.zip"))
    joint_files = sorted(motion_dir.rglob("*.npy"))
    if not joint_files or not text_archives:
        raise FileNotFoundError("Motion-X++ converted joints/semantic labels are missing")
    if not args.reference_joints.is_file():
        raise FileNotFoundError(args.reference_joints)

    split_map, official_split_counts, split_deduplication = read_splits(args.split_root)
    split_name_index = build_split_name_index(split_map)
    text_index = build_text_index(text_archives)
    text_categories = {archive.stem.lower() for archive in text_archives}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "new_joint_vecs").mkdir(exist_ok=True)
    (args.output / "texts").mkdir(exist_ok=True)

    tasks = []
    unmatched_split = []
    unmatched_text = []
    excluded_without_text = Counter()
    split_assignment = Counter()
    official_ambiguous = 0
    discovered = 0
    text_eligible = 0
    for joint_path in joint_files:
        relative = joint_path.relative_to(motion_dir)
        category = relative.parts[0].lower()
        member = PurePosixPath(*relative.parts[1:]).as_posix()
        discovered += 1
        if category not in text_categories:
            excluded_without_text[category] += 1
            continue
        text_eligible += 1
        candidates = text_key_candidates(member, category)
        captions = []
        for key in candidates:
            captions.extend(text_index.get(key, []))
        captions = list(dict.fromkeys(captions))
        if not captions:
            unmatched_text.append({"path": str(joint_path), "candidates": candidates})
            continue
        split, assignment, split_group, match_count = choose_group_split(
            member, category, split_name_index
        )
        split_assignment[assignment] += 1
        official_ambiguous += int(assignment == "official" and match_count > 1)
        source_key = candidates[0]
        if assignment == "hash_fallback" and len(unmatched_split) < 50:
            unmatched_split.append({
                "source_key": source_key,
                "split_group": split_group,
                "assigned_split": split,
            })
        tasks.append((str(joint_path), source_key, split, captions, args.overwrite))

    assigned_tasks = sum(split_assignment.values())
    text_match = assigned_tasks / text_eligible if text_eligible else 0.0
    split_match = assigned_tasks / assigned_tasks if assigned_tasks else 0.0
    if split_match < args.min_split_match:
        raise RuntimeError(f"Official split match {split_match:.4f} below {args.min_split_match}")
    if text_match < args.min_text_match:
        raise RuntimeError(f"Semantic text match {text_match:.4f} below {args.min_text_match}")
    if args.limit is not None:
        tasks = tasks[:args.limit]

    worker_args = (
        str(args.reference_joints), str(args.output), args.source_fps, args.target_fps,
        args.max_frames, args.min_frames, args.source_up_axis,
    )
    records = []
    status = Counter()
    errors = []
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=init_worker, initargs=worker_args) as pool:
        for result in pool.imap_unordered(process_task, tasks, chunksize=4):
            status[result["status"]] += 1
            records.extend(result.get("records", []))
            if result["status"] == "error":
                errors.append(result)
    if errors:
        error_path = args.output / "preprocess_errors.json"
        error_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")
        if len(errors) > args.max_errors:
            raise RuntimeError(
                f"Motion preprocessing failed for {len(errors)} files; see {error_path}"
            )
        print(
            f"Excluding {len(errors)} recorded degenerate source motions; "
            f"limit={args.max_errors}",
            flush=True,
        )
    if not records:
        raise RuntimeError("No Motion-X records were produced")

    try:
        import spacy
        nlp = spacy.load(args.spacy_model, disable=["parser", "ner"])
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            f"spaCy model {args.spacy_model!r} is required; install it before preprocessing"
        ) from exc

    caption_tokens = {}
    unique_captions = list(dict.fromkeys(
        caption for record in records for caption in record["captions"]
    ))
    for document, caption in zip(nlp.pipe(unique_captions, batch_size=256), unique_captions):
        caption_tokens[caption] = tokenize_caption(lambda _: document, caption)

    by_split = defaultdict(list)
    id_map = []
    for record in sorted(records, key=lambda item: item["motion_id"]):
        motion_id = record["motion_id"]
        lines = []
        for caption in record["captions"]:
            tokens = " ".join(caption_tokens[caption])
            lines.append(f"{caption}#{tokens}#0.0#0.0")
        (args.output / "texts" / f"{motion_id}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        by_split[record["split"]].append(motion_id)
        id_map.append({key: value for key, value in record.items() if key != "captions"})

    for split in ("train", "val", "test"):
        ids = sorted(by_split[split])
        (args.output / f"{split}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    if set(by_split["train"]) & set(by_split["test"]):
        raise RuntimeError("Generated split leakage")

    train_motions = [
        np.load(args.output / "new_joint_vecs" / f"{motion_id}.npy", mmap_mode="r")
        for motion_id in by_split["train"]
    ]
    mean, std = humanml_mean_std(train_motions)
    np.save(args.output / "Mean.npy", mean)
    np.save(args.output / "Std.npy", std)
    (args.output / "motion_id_map.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in id_map),
        encoding="utf-8",
    )
    stale_files = remove_stale_generated_files(
        args.output, {record["motion_id"] for record in records}
    )

    report = {
        "source_fps": args.source_fps,
        "target_fps": args.target_fps,
        "source_up_axis": args.source_up_axis,
        "official_split_counts": official_split_counts,
        "official_split_deduplication": split_deduplication,
        "downloaded_motion_files": discovered,
        "text_eligible_motion_files": text_eligible,
        "excluded_without_text_category": dict(excluded_without_text),
        "split_match": split_match,
        "split_assignment": dict(split_assignment),
        "official_split_match": (
            split_assignment["official"] / assigned_tasks if assigned_tasks else 0.0
        ),
        "official_ambiguous_name_matches": official_ambiguous,
        "hash_fallback_ratios": {"train": 0.80, "val": 0.05, "test": 0.15},
        "text_match": text_match,
        "source_tasks": len(tasks),
        "task_status": dict(status),
        "excluded_error_count": len(errors),
        "removed_stale_files": stale_files,
        "generated_clips": len(records),
        "generated_splits": {key: len(value) for key, value in by_split.items()},
        "motion_dim": int(mean.shape[0]),
        "mean_finite": bool(np.isfinite(mean).all()),
        "std_min": float(std.min()),
        "unmatched_split_examples": unmatched_split[:50],
        "unmatched_text_examples": unmatched_text[:50],
    }
    (args.output / "preprocess_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
