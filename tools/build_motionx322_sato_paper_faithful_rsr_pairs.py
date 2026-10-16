#!/usr/bin/env python3
"""Build Motion-X pairs for the paper-faithful SATO RSR recipe.

The SATO paper uses Random Synonym Replacement (RSR), keeps the original and
perturbed word sequence lengths equal in its formulation, and applies the
perturbation branch to every training sample.  Motion-X has no author-released
RSR annotations, so this builder adapts the paper data requirement by:

1. selecting only filtered P1 lexical-preserving pairs with equal whitespace
   token counts;
2. filling uncovered captions with one deterministic replacement from a
   manually curated synonym table; and
3. retaining an identity fallback when no safe replacement is available.

It intentionally does not impose strict CLIP-BPE alignment.  Training therefore
uses the released SATO absolute-position/EOT heuristic rather than the later
strict token-aligned repair.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


WORD = re.compile(r"^(?P<prefix>[^A-Za-z]*)(?P<word>[A-Za-z]+)(?P<suffix>[^A-Za-z]*)$")

# Single-token, morphology-matched replacements.  Direction, body side, count,
# speed category, and body-part words are deliberately absent.
MANUAL_SYNONYMS = {
    "person": ("human", "individual"),
    "human": ("person", "individual"),
    "individual": ("person", "human"),
    "man": ("gentleman", "male"),
    "woman": ("lady", "female"),
    "boy": ("youth",),
    "girl": ("youngster",),
    "guy": ("fellow",),
    "figure": ("individual",),
    "performs": ("executes",),
    "perform": ("execute",),
    "performing": ("executing",),
    "walk": ("stroll", "stride"),
    "walks": ("strolls", "strides"),
    "walking": ("strolling", "striding"),
    "move": ("proceed",),
    "moves": ("proceeds",),
    "moving": ("proceeding",),
    "sit": ("rest",),
    "sits": ("rests",),
    "sitting": ("seated",),
    "standing": ("upright",),
    "stand": ("remain",),
    "stands": ("remains",),
    "raise": ("lift",),
    "raises": ("lifts",),
    "raising": ("lifting",),
    "lift": ("raise",),
    "lifts": ("raises",),
    "lifting": ("raising",),
    "turn": ("rotate", "pivot"),
    "turns": ("rotates", "pivots"),
    "turning": ("rotating", "pivoting"),
    "bend": ("flex",),
    "bends": ("flexes",),
    "bending": ("flexing",),
    "jump": ("leap",),
    "jumps": ("leaps",),
    "jumping": ("leaping",),
    "kick": ("punt",),
    "kicks": ("punts",),
    "kicking": ("punting",),
    "punch": ("strike",),
    "punches": ("strikes",),
    "punching": ("striking",),
    "wave": ("gesture",),
    "waves": ("gestures",),
    "waving": ("gesturing",),
    "look": ("gaze",),
    "looks": ("gazes",),
    "looking": ("gazing",),
    "holds": ("grasps",),
    "hold": ("grasp",),
    "holding": ("grasping",),
    "uses": ("utilizes",),
    "use": ("utilize",),
    "using": ("utilizing",),
    "with": ("using",),
    "while": ("whilst",),
    "simultaneously": ("concurrently",),
    "quickly": ("rapidly", "swiftly"),
    "slowly": ("gradually",),
    "object": ("item",),
    "motion": ("movement",),
    "movement": ("motion",),
    "exercise": ("workout",),
    "repeatedly": ("continually",),
}


def normalize(text):
    return " ".join(str(text).split())


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(path):
    spec = importlib.util.spec_from_file_location("sato_simple_tokenizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load tokenizer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SimpleTokenizer()


def preserve_case(source, replacement):
    if source.isupper():
        return replacement.upper()
    if source[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def manual_replacement(caption, rng):
    words = caption.split(" ")
    candidates = []
    for index, token in enumerate(words):
        match = WORD.fullmatch(token)
        if match is None:
            continue
        source = match.group("word")
        synonyms = MANUAL_SYNONYMS.get(source.lower(), ())
        for synonym in synonyms:
            replacement = preserve_case(source, synonym)
            updated = list(words)
            updated[index] = (
                match.group("prefix") + replacement + match.group("suffix")
            )
            pair = " ".join(updated)
            if pair != caption and len(pair.split(" ")) == len(words):
                candidates.append((index, source, replacement, pair))
    if not candidates:
        return None
    return rng.choice(candidates)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    dataset = root / "dataset" / "Motion-X-322"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl", type=Path,
        default=dataset / "preserving_pairs_gemini31_human_recipe_filtered.jsonl")
    parser.add_argument("--train-split", type=Path, default=dataset / "train.txt")
    parser.add_argument("--val-split", type=Path, default=dataset / "val.txt")
    parser.add_argument("--test-split", type=Path, default=dataset / "test.txt")
    parser.add_argument("--text-dir", type=Path, default=dataset / "texts")
    parser.add_argument(
        "--tokenizer", type=Path,
        default=root / "Stable-Text-to-Motion-Framework-official" / "CLIP" /
        "clip" / "simple_tokenizer.py")
    parser.add_argument(
        "--output-jsonl", type=Path,
        default=dataset / "paper_faithful_rsr_pairs_sato_train.jsonl")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-bpe-tokens", type=int, default=75)
    return parser.parse_args()


def load_ids(path):
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()}


def main():
    args = parse_args()
    paths = [
        args.source_jsonl, args.train_split, args.val_split, args.test_split,
        args.tokenizer,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.text_dir.is_dir():
        raise FileNotFoundError(args.text_dir)
    if not 1 <= args.max_bpe_tokens <= 75:
        raise ValueError("--max-bpe-tokens must be in [1, 75]")

    train_ids = load_ids(args.train_split)
    val_ids = load_ids(args.val_split)
    test_ids = load_ids(args.test_split)
    split_overlap = {
        "train_val": len(train_ids & val_ids),
        "train_test": len(train_ids & test_ids),
        "val_test": len(val_ids & test_ids),
    }
    if any(split_overlap.values()):
        raise ValueError(f"Dataset split leakage detected: {split_overlap}")

    tokenizer = load_tokenizer(args.tokenizer)
    captions = []
    caption_keys = set()
    missing_text_files = []
    for motion_id in sorted(train_ids):
        text_path = args.text_dir / f"{motion_id}.txt"
        if not text_path.is_file():
            missing_text_files.append(motion_id)
            continue
        for raw_line in text_path.read_text(
                encoding="utf-8", errors="replace").splitlines():
            caption = normalize(raw_line.split("#", 1)[0])
            if not caption:
                continue
            key = (motion_id, caption)
            if key in caption_keys:
                continue
            caption_keys.add(key)
            captions.append(key)
    if missing_text_files:
        raise FileNotFoundError(
            f"Missing {len(missing_text_files)} train text files; first: "
            f"{missing_text_files[:5]}")

    source_candidates = defaultdict(list)
    source_counts = Counter()
    with args.source_jsonl.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            source_counts["rows"] += 1
            row = json.loads(raw_line)
            if row.get("relation") != "preserve" or row.get("level") != "P1":
                continue
            source_counts["p1_preserve"] += 1
            motion_id = str(row.get("motion_id", ""))
            if motion_id not in train_ids:
                source_counts["not_train"] += 1
                continue
            original = normalize(row.get("text_orig", ""))
            perturbed = normalize(row.get("text_pair", ""))
            key = (motion_id, original)
            if key not in caption_keys:
                source_counts["caption_not_in_train_text"] += 1
                continue
            if not perturbed or perturbed == original:
                source_counts["empty_or_identity"] += 1
                continue
            if len(original.split(" ")) != len(perturbed.split(" ")):
                source_counts["whitespace_length_mismatch"] += 1
                continue
            original_bpe = tokenizer.encode(original)
            perturbed_bpe = tokenizer.encode(perturbed)
            if (len(original_bpe) > args.max_bpe_tokens or
                    len(perturbed_bpe) > args.max_bpe_tokens):
                source_counts["clip_context_overflow"] += 1
                continue
            candidate = {
                "source_pair_id": row.get("pair_id"),
                "text_pair": perturbed,
                "original_bpe_length": len(original_bpe),
                "perturbed_bpe_length": len(perturbed_bpe),
            }
            signature = (candidate["source_pair_id"], perturbed)
            if any((existing["source_pair_id"], existing["text_pair"]) == signature
                   for existing in source_candidates[key]):
                source_counts["duplicate"] += 1
                continue
            source_candidates[key].append(candidate)
            source_counts["accepted_candidates"] += 1

    output = args.output_jsonl.resolve()
    manifest = (args.manifest.resolve() if args.manifest else
                output.with_name(output.stem + ".manifest.json"))
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")

    rng = random.Random(args.seed)
    counts = Counter()
    replacement_motion_ids = set()
    selected_source_ids = set()
    with output_tmp.open("w", encoding="utf-8") as out:
        for pair_id, (motion_id, original) in enumerate(sorted(captions), start=1):
            candidates = source_candidates.get((motion_id, original), [])
            if candidates:
                selected = rng.choice(sorted(
                    candidates,
                    key=lambda row: (str(row["source_pair_id"]), row["text_pair"])))
                perturbed = selected["text_pair"]
                source_type = "filtered_p1_lexical_pair"
                source_pair_id = selected["source_pair_id"]
                replacement = None
                selected_source_ids.add(source_pair_id)
            else:
                manual = manual_replacement(original, rng)
                if manual is not None:
                    word_index, source_word, replacement_word, perturbed = manual
                    source_type = "manual_single_token_synonym"
                    source_pair_id = None
                    replacement = {
                        "word_index": word_index,
                        "source_word": source_word,
                        "replacement_word": replacement_word,
                    }
                else:
                    perturbed = original
                    source_type = "identity_fallback"
                    source_pair_id = None
                    replacement = None

            if len(original.split(" ")) != len(perturbed.split(" ")):
                raise AssertionError("Builder emitted a whitespace-length mismatch")
            original_bpe = tokenizer.encode(original)
            perturbed_bpe = tokenizer.encode(perturbed)
            if len(original_bpe) > args.max_bpe_tokens:
                # Preserve training-set coverage.  CLIP would truncate this caption
                # in the original implementation, so the safe adaptation is an
                # explicit identity pair rather than a truncated synonym pair.
                perturbed = original
                perturbed_bpe = original_bpe
                source_type = "identity_fallback_clip_overflow"
                source_pair_id = None
                replacement = None

            has_replacement = perturbed != original
            counts[source_type] += 1
            counts["rows"] += 1
            counts["with_replacement"] += int(has_replacement)
            counts["without_replacement"] += int(not has_replacement)
            if has_replacement:
                replacement_motion_ids.add(motion_id)

            row = {
                "pair_id": pair_id,
                "motion_id": motion_id,
                "text_orig": original,
                "text_pair": perturbed,
                "relation": "preserve",
                "edit_type": "random_synonym_replacement" if has_replacement else "identity_fallback",
                "level": "P1" if has_replacement else "fallback",
                "source_type": source_type,
                "source_pair_id": source_pair_id,
                "manual_replacement": replacement,
                "same_whitespace_length": True,
                "original_whitespace_tokens": len(original.split(" ")),
                "perturbed_whitespace_tokens": len(perturbed.split(" ")),
                "original_bpe_length": len(original_bpe),
                "perturbed_bpe_length": len(perturbed_bpe),
                "paper_faithful_rsr": True,
                "attention_aligned": False,
                "has_synonym_replacement": has_replacement,
            }
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    output_tmp.replace(output)

    if counts["rows"] != len(caption_keys):
        raise AssertionError(
            f"Expected one row per train caption ({len(caption_keys)}), "
            f"wrote {counts['rows']}")

    manifest_data = {
        "schema_version": 1,
        "experiment_role": "SATO paper-faithful Motion-X RSR training pairs",
        "paper": {
            "title": "SATO: Stable Text-to-Motion Framework",
            "arxiv": "2405.01461v3",
            "selected_perturbation": "Random Synonym Replacement (RSR)",
            "application_probability": 1.0,
            "reported_generation_method": "manually human-designed synonym perturbations",
        },
        "official_source": {
            "repository": "https://github.com/sato-team/Stable-Text-to-Motion-Framework.git",
            "commit": "aa46405ad1af3c985e4c4e7c18d60d20993ef2a8",
            "position_logic": "absolute CLIP positions with whitespace-count EOT estimate",
        },
        "motionx_adaptation": {
            "source_p1_pairs": str(args.source_jsonl.resolve()),
            "source_p1_pairs_sha256": sha256_file(args.source_jsonl.resolve()),
            "source_generation": "Gemini 3.1 Flash Lite P1 lexical prompt plus semantic judge",
            "fallback_generation": "manually curated deterministic single-token synonym map",
            "identity_fallback": True,
            "strict_bpe_alignment": False,
            "same_whitespace_length_required": True,
            "max_clip_bpe_tokens": args.max_bpe_tokens,
            "selection_seed": args.seed,
        },
        "splits": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "overlap": split_overlap,
        },
        "coverage": {
            "train_caption_rows": len(caption_keys),
            "rows_written": counts["rows"],
            "rows_with_synonym_replacement": counts["with_replacement"],
            "identity_fallback_rows": counts["without_replacement"],
            "replacement_caption_coverage": counts["with_replacement"] / len(caption_keys),
            "replacement_motion_ids": len(replacement_motion_ids),
            "replacement_motion_coverage": len(replacement_motion_ids) / len(train_ids),
            "selected_source_pair_ids": len(selected_source_ids),
        },
        "output_counts": dict(sorted(counts.items())),
        "source_filter_counts": dict(sorted(source_counts.items())),
        "inputs": {
            "train_split": str(args.train_split.resolve()),
            "train_split_sha256": sha256_file(args.train_split.resolve()),
            "val_split": str(args.val_split.resolve()),
            "val_split_sha256": sha256_file(args.val_split.resolve()),
            "test_split": str(args.test_split.resolve()),
            "test_split_sha256": sha256_file(args.test_split.resolve()),
            "tokenizer": str(args.tokenizer.resolve()),
            "tokenizer_sha256": sha256_file(args.tokenizer.resolve()),
        },
        "output": {
            "jsonl": str(output),
            "sha256": sha256_file(output),
        },
    }
    manifest.write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")

    print(json.dumps({
        "output": str(output),
        "manifest": str(manifest),
        "rows": counts["rows"],
        "with_replacement": counts["with_replacement"],
        "identity_fallback": counts["without_replacement"],
        "replacement_coverage": counts["with_replacement"] / len(caption_keys),
        "sha256": manifest_data["output"]["sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()