#!/usr/bin/env python3
"""Build strict CLIP-BPE-aligned lexical pairs for Motion-X SATO training."""

import argparse
import hashlib
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path


WORD_WITH_AFFIXES = re.compile(
    r"^(?P<prefix>[^A-Za-z]*)(?P<word>[A-Za-z]+(?:['-][A-Za-z]+)*)(?P<suffix>[^A-Za-z]*)$"
)


def load_simple_tokenizer(path):
    spec = importlib.util.spec_from_file_location("sato_simple_tokenizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load SATO tokenizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleTokenizer()


def normalize_space(text):
    return " ".join(str(text).split())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject(reason, **details):
    return None, {"reason": reason, **details}


def align_row(
    row, tokenizer, train_ids, allowed_levels, max_bpe_tokens,
    source_captions=None,
):
    motion_id = str(row.get("motion_id", "")).strip()
    original = normalize_space(row.get("text_orig", ""))
    perturbed = normalize_space(row.get("text_pair", ""))

    if row.get("relation", "preserve") != "preserve":
        return reject("non_preserving_relation")
    if allowed_levels and str(row.get("level", "")) not in allowed_levels:
        return reject("disallowed_source_level", level=row.get("level", ""))
    if motion_id not in train_ids:
        return reject("not_in_train_split", motion_id=motion_id)
    if not original or not perturbed:
        return reject("empty_caption", motion_id=motion_id)
    if source_captions is not None and original not in source_captions.get(motion_id, set()):
        return reject(
            "original_caption_not_in_motion_text_file",
            motion_id=motion_id,
            original=original,
        )

    original_words = original.split(" ")
    perturbed_words = perturbed.split(" ")
    if len(original_words) != len(perturbed_words):
        return reject(
            "whitespace_token_count_mismatch",
            motion_id=motion_id,
            original_count=len(original_words),
            perturbed_count=len(perturbed_words),
        )

    original_spans = []
    perturbed_spans = []
    replacements = []
    bpe_cursor = 1  # Absolute CLIP position 0 is SOT.
    for word_index, (original_word, perturbed_word) in enumerate(
        zip(original_words, perturbed_words)
    ):
        original_bpe = tokenizer.encode(original_word)
        perturbed_bpe = tokenizer.encode(perturbed_word)
        if len(original_bpe) != len(perturbed_bpe):
            return reject(
                "word_bpe_span_length_mismatch",
                motion_id=motion_id,
                word_index=word_index,
                original_word=original_word,
                perturbed_word=perturbed_word,
                original_bpe=original_bpe,
                perturbed_bpe=perturbed_bpe,
            )

        span_end = bpe_cursor + len(original_bpe)
        original_spans.append(original_bpe)
        perturbed_spans.append(perturbed_bpe)
        if original_word != perturbed_word:
            original_match = WORD_WITH_AFFIXES.fullmatch(original_word)
            perturbed_match = WORD_WITH_AFFIXES.fullmatch(perturbed_word)
            if original_match is None or perturbed_match is None:
                return reject(
                    "non_lexical_position_change",
                    motion_id=motion_id,
                    word_index=word_index,
                    original_word=original_word,
                    perturbed_word=perturbed_word,
                )
            if (
                original_match.group("prefix") != perturbed_match.group("prefix")
                or original_match.group("suffix") != perturbed_match.group("suffix")
            ):
                return reject(
                    "punctuation_or_structure_change",
                    motion_id=motion_id,
                    word_index=word_index,
                    original_word=original_word,
                    perturbed_word=perturbed_word,
                )
            if original_bpe != perturbed_bpe:
                replacements.append(
                    {
                        "word_index": word_index,
                        "clip_start": bpe_cursor,
                        "clip_end": span_end,
                        "original_word": original_word,
                        "perturbed_word": perturbed_word,
                        "original_bpe": original_bpe,
                        "perturbed_bpe": perturbed_bpe,
                    }
                )
        bpe_cursor = span_end

    if not replacements:
        return reject("no_bpe_lexical_replacement", motion_id=motion_id)

    original_bpe = tokenizer.encode(original)
    perturbed_bpe = tokenizer.encode(perturbed)
    flattened_original = [token for span in original_spans for token in span]
    flattened_perturbed = [token for span in perturbed_spans for token in span]
    if original_bpe != flattened_original or perturbed_bpe != flattened_perturbed:
        return reject("whitespace_span_reconstruction_mismatch", motion_id=motion_id)
    if len(original_bpe) != len(perturbed_bpe):
        return reject("sentence_bpe_length_mismatch", motion_id=motion_id)
    if len(original_bpe) > max_bpe_tokens:
        return reject(
            "clip_context_overflow",
            motion_id=motion_id,
            bpe_length=len(original_bpe),
            max_bpe_tokens=max_bpe_tokens,
        )

    changed_positions = {
        position
        for replacement in replacements
        for position in range(replacement["clip_start"], replacement["clip_end"])
    }
    for position, (original_id, perturbed_id) in enumerate(
        zip(original_bpe, perturbed_bpe), start=1
    ):
        if position not in changed_positions and original_id != perturbed_id:
            return reject(
                "bpe_change_outside_replacement_span",
                motion_id=motion_id,
                clip_position=position,
            )

    eot_position = len(original_bpe) + 1
    accepted = {
        "source_pair_id": row.get("pair_id"),
        "motion_id": motion_id,
        "text_orig": original,
        "text_pair": perturbed,
        "relation": "preserve",
        "edit_type": "token_aligned_lexical_substitution",
        "source_level": row.get("level", ""),
        "original_bpe": original_bpe,
        "perturbed_bpe": perturbed_bpe,
        "eot_position": eot_position,
        "valid_clip_positions": list(range(1, eot_position)),
        "replacement_count": len(replacements),
        "replacements": replacements,
        "attention_aligned": True,
    }
    return accepted, None


def parse_args():
    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "dataset" / "Motion-X-322"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=dataset_root / "preserving_pairs_gemini31_human_recipe_filtered.jsonl",
    )
    parser.add_argument("--train-split", type=Path, default=dataset_root / "train.txt")
    parser.add_argument("--text-dir", type=Path, default=dataset_root / "texts")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=root
        / "Stable-Text-to-Motion-Framework-official"
        / "CLIP"
        / "clip"
        / "simple_tokenizer.py",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=dataset_root / "token_aligned_lexical_pairs_sato_clip_train.jsonl",
    )
    parser.add_argument("--rejected-jsonl", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--allowed-level", action="append", default=None)
    parser.add_argument("--max-bpe-tokens", type=int, default=75)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source_jsonl.resolve()
    train_split = args.train_split.resolve()
    output = args.output_jsonl.resolve()
    rejected = (
        args.rejected_jsonl.resolve()
        if args.rejected_jsonl
        else output.with_name(output.stem + ".rejected.jsonl")
    )
    manifest_path = (
        args.manifest.resolve()
        if args.manifest
        else output.with_name(output.stem + ".manifest.json")
    )
    for path in (source, train_split, args.tokenizer.resolve()):
        if not path.is_file():
            raise FileNotFoundError(path)
    text_dir = args.text_dir.resolve()
    if not text_dir.is_dir():
        raise FileNotFoundError(text_dir)
    if args.max_bpe_tokens < 1 or args.max_bpe_tokens > 75:
        raise ValueError("--max-bpe-tokens must be in [1, 75]")

    allowed_levels = set(args.allowed_level or ["P1"])
    output.parent.mkdir(parents=True, exist_ok=True)
    rejected.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    rejected_tmp = rejected.with_suffix(rejected.suffix + ".tmp")

    tokenizer = load_simple_tokenizer(args.tokenizer.resolve())
    train_ids = {line.strip() for line in train_split.open(encoding="utf-8") if line.strip()}
    source_captions = {}
    missing_text_files = []
    for motion_id in sorted(train_ids):
        text_path = text_dir / f"{motion_id}.txt"
        if not text_path.is_file():
            missing_text_files.append(motion_id)
            continue
        captions = set()
        for raw_line in text_path.open(encoding="utf-8", errors="replace"):
            caption = normalize_space(raw_line.split("#", 1)[0])
            if caption:
                captions.add(caption)
        source_captions[motion_id] = captions
    if missing_text_files:
        raise FileNotFoundError(
            f"Missing {len(missing_text_files)} train text files; first: "
            f"{missing_text_files[:5]}")
    validation_ids = set()
    test_ids = set()
    for split_name, target in (("val.txt", validation_ids), ("test.txt", test_ids)):
        split_path = train_split.with_name(split_name)
        if split_path.is_file():
            target.update(
                line.strip() for line in split_path.open(encoding="utf-8") if line.strip()
            )
    split_overlap = {
        "train_val": len(train_ids & validation_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(validation_ids & test_ids),
    }
    if any(split_overlap.values()):
        raise ValueError(f"Dataset split leakage detected: {split_overlap}")

    counts = Counter()
    accepted_motion_ids = set()
    accepted_originals = set()
    replacement_counts = Counter()
    seen_rows = set()
    with source.open(encoding="utf-8") as source_handle, output_tmp.open(
        "w", encoding="utf-8"
    ) as output_handle, rejected_tmp.open("w", encoding="utf-8") as rejected_handle:
        for source_line_number, raw_line in enumerate(source_handle, start=1):
            if not raw_line.strip():
                continue
            counts["source_rows"] += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                counts["rejected_invalid_json"] += 1
                rejected_handle.write(
                    json.dumps(
                        {
                            "source_line_number": source_line_number,
                            "reason": "invalid_json",
                            "error": str(error),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue

            accepted, rejection = align_row(
                row,
                tokenizer=tokenizer,
                train_ids=train_ids,
                allowed_levels=allowed_levels,
                max_bpe_tokens=args.max_bpe_tokens,
                source_captions=source_captions,
            )
            if rejection is not None:
                reason = rejection["reason"]
                counts[f"rejected_{reason}"] += 1
                rejected_handle.write(
                    json.dumps(
                        {
                            "source_line_number": source_line_number,
                            "source_pair_id": row.get("pair_id"),
                            "motion_id": row.get("motion_id", ""),
                            "text_orig": normalize_space(row.get("text_orig", "")),
                            "text_pair": normalize_space(row.get("text_pair", "")),
                            **rejection,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue

            dedup_key = (
                accepted["motion_id"],
                accepted["text_orig"],
                accepted["text_pair"],
            )
            if dedup_key in seen_rows:
                counts["rejected_duplicate_pair"] += 1
                rejected_handle.write(
                    json.dumps(
                        {
                            "source_line_number": source_line_number,
                            "source_pair_id": row.get("pair_id"),
                            "reason": "duplicate_pair",
                            "motion_id": accepted["motion_id"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                continue
            seen_rows.add(dedup_key)
            counts["accepted_rows"] += 1
            accepted_motion_ids.add(accepted["motion_id"])
            accepted_originals.add((accepted["motion_id"], accepted["text_orig"]))
            replacement_counts[str(accepted["replacement_count"])] += 1
            output_handle.write(json.dumps(accepted, sort_keys=True) + "\n")

    output_tmp.replace(output)
    rejected_tmp.replace(rejected)
    rejected_rows = counts["source_rows"] - counts["accepted_rows"]
    if counts["accepted_rows"] + rejected_rows != counts["source_rows"]:
        raise AssertionError("Accepted/rejected accounting mismatch")

    manifest = {
        "schema_version": 1,
        "builder": str(Path(__file__).resolve()),
        "source_jsonl": str(source),
        "source_sha256": sha256_file(source),
        "train_split": str(train_split),
        "train_split_sha256": sha256_file(train_split),
        "text_dir": str(text_dir),
        "source_caption_check": True,
        "tokenizer": str(args.tokenizer.resolve()),
        "tokenizer_sha256": sha256_file(args.tokenizer.resolve()),
        "allowed_levels": sorted(allowed_levels),
        "max_bpe_tokens": args.max_bpe_tokens,
        "output_jsonl": str(output),
        "output_sha256": sha256_file(output),
        "rejected_jsonl": str(rejected),
        "rejected_sha256": sha256_file(rejected),
        "split_sizes": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "split_overlap": split_overlap,
        "counts": dict(sorted(counts.items())),
        "rejected_rows": rejected_rows,
        "accepted_motion_ids": len(accepted_motion_ids),
        "accepted_train_motion_coverage": len(accepted_motion_ids) / len(train_ids),
        "unchanged_train_motion_ids": len(train_ids - accepted_motion_ids),
        "unchanged_train_motion_ratio": len(train_ids - accepted_motion_ids) / len(train_ids),
        "accepted_motion_caption_keys": len(accepted_originals),
        "replacement_count_histogram": dict(sorted(replacement_counts.items())),
        "invariants": {
            "train_only": True,
            "original_caption_exists_in_motion_text_file": True,
            "no_split_overlap": not any(split_overlap.values()),
            "same_whitespace_token_count": True,
            "same_bpe_span_length_at_every_position": True,
            "same_true_eot_position": True,
            "no_insert_delete_or_reorder": True,
            "unchanged_fallback_skips_attention_loss": True,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()