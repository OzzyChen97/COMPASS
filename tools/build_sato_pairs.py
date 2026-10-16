#!/usr/bin/env python
import argparse
import codecs as cs
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI


PRESERVE_PROMPT = """You create semantic-preserving paraphrases of text-to-motion captions.
For every item, return exactly two paraphrases for each level P1, P2, and P3.
P1 changes words, P2 changes syntax, and P3 changes narrative style.
Preserve action, order, direction, speed, repetitions, body parts, and spatial relations.
Return JSON only: {"results":[{"index":1,"P1":["...","..."],"P2":["...","..."],"P3":["...","..."]}]}.
"""

HUMANML3D_TEST_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate EXACTLY 2 paraphrases at EACH of 3 levels (6 total).

P1 (Lexical): Replace words with synonyms. Keep sentence structure mostly identical.
P2 (Syntactic): Change sentence structure, word order, or clause order.
P3 (Scene/Narrative): Rewrite in a different narrative style.

Rules:
- Preserve ALL semantic content: action, direction, speed, body parts, count, order
- Do NOT change directions such as left/right/forward/backward/clockwise/counterclockwise
- Do NOT change body part references
- Do NOT change action type, step count, or speed
- Each paraphrase must be natural, grammatical English
- If the input is ungrammatical, preserve its intended motion meaning while improving grammar

Return JSON only:
{"results":[{"index":1,"P1":["p1","p2"],"P2":["p1","p2"],"P3":["p1","p2"]}]}.
"""

ALTER_PROMPT = """You create semantic-changing edits of text-to-motion captions.
For every item, return up to one natural edit for each applicable type: action, direction, bodypart.
Each edit must change only the named motion attribute and retain the other details.
Do not emit a type when the caption cannot support a clean edit of that type.
Return JSON only: {"results":[{"index":1,"edits":[{"edit_type":"action","text":"..."}]}]}.
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SATO++ preserving/altering pairs for HumanML3D or KIT-ML."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", choices=["train", "test"], required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("SATO_PAIR_MODEL", "gpt-5.4-mini"),
    )
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument(
        "--preserve-prompt-style",
        choices=("sato", "humanml3d"),
        default="sato",
        help="Prompt recipe used to generate P1/P2/P3 preserve candidates.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=8000,
        help="Maximum completion tokens for each batched JSON response.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--completion-rounds",
        type=int,
        default=3,
        help="Retry schema-incomplete captions within the same run.",
    )
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def normalize_caption(caption):
    caption = re.sub(r"\s+", " ", caption.strip())
    return caption


def read_motion_captions(dataset_root, split, one_per_motion):
    split_path = dataset_root / f"{split}.txt"
    text_dir = dataset_root / "texts"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    if not text_dir.is_dir():
        raise FileNotFoundError(text_dir)

    rows = []
    seen = set()
    motion_ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    mean_path = dataset_root / "Mean.npy"
    motion_dim = int(np.load(mean_path, mmap_mode="r").shape[-1]) if mean_path.exists() else None
    min_motion_len = 1 if motion_dim == 322 else (24 if motion_dim == 251 else 40)
    for motion_id in motion_ids:
        motion_path = dataset_root / "new_joint_vecs" / f"{motion_id}.npy"
        text_path = text_dir / f"{motion_id}.txt"
        if not motion_path.exists() or not text_path.exists():
            continue
        motion_length = int(np.load(motion_path, mmap_mode="r").shape[0])
        variable_length = motion_dim == 322
        if motion_length < min_motion_len or (not variable_length and motion_length >= 200):
            continue
        candidates = []
        with cs.open(text_path, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                parts = line.strip().split("#")
                if not parts:
                    continue
                caption = normalize_caption(parts[0])
                if not caption:
                    continue
                full_motion = len(parts) < 4 or (
                    parts[2] in {"0", "0.0", "nan", "NaN"}
                    and parts[3] in {"0", "0.0", "nan", "NaN"}
                )
                candidates.append((caption, full_motion))

        if one_per_motion and candidates:
            full = [item for item in candidates if item[1]]
            candidates = [full[0] if full else candidates[0]]

        for caption, _ in candidates:
            key = (motion_id, caption)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"motion_id": motion_id, "text_orig": caption})
    return rows


def call_json(client, model, prompt, batch, max_retries, max_output_tokens):
    items = "\n".join(
        f'{index + 1}. "{row["text_orig"]}"' for index, row in enumerate(batch)
    )
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": items},
                ],
                temperature=0.7,
                max_tokens=max_output_tokens,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if "```" in content:
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
                if match:
                    content = match.group(1)
            return json.loads(content).get("results", [])
        except Exception as exc:
            if attempt + 1 == max_retries:
                raise
            delay = min(60.0, 3.0 * (2 ** attempt)) + random.uniform(0.0, 2.0)
            response = getattr(exc, "response", None)
            headers = getattr(response, "headers", None)
            if headers is not None:
                retry_after = headers.get("retry-after")
                try:
                    delay = max(delay, float(retry_after))
                except (TypeError, ValueError):
                    pass
            print(
                f"API request failed ({type(exc).__name__}); "
                f"retry {attempt + 2}/{max_retries} in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)


def generate_batch(
    client_args, batch, protocol, preserve_prompt, max_retries,
    max_output_tokens,
):
    api_key, base_url, model = client_args
    client = OpenAI(api_key=api_key, base_url=base_url)
    preserve = call_json(
        client, model, preserve_prompt, batch, max_retries,
        max_output_tokens,
    )
    alter = []
    if protocol == "train":
        alter = call_json(
            client, model, ALTER_PROMPT, batch, max_retries,
            max_output_tokens,
        )
    return {"items": batch, "preserve": preserve, "alter": alter}


def basic_filter(original, candidate):
    if not isinstance(candidate, str):
        return False
    candidate = normalize_caption(candidate)
    if not candidate or candidate.lower().rstrip(".") == original.lower().rstrip("."):
        return False
    original_len = len(original.split())
    candidate_len = len(candidate.split())
    return max(1, int(original_len * 0.3)) <= candidate_len <= max(3, int(original_len * 3.0))


def candidate_values(row, level):
    values = row.get(level, []) if isinstance(row, dict) else []
    if isinstance(values, str):
        values = [values]
    return values if isinstance(values, list) else []


def result_by_index(results):
    indexed = {}
    for row in results:
        try:
            indexed[int(row.get("index")) - 1] = row
        except (TypeError, ValueError):
            continue
    return indexed


def deterministic_choice(values, motion_id, seed):
    digest = hashlib.sha256(f"{seed}:{motion_id}".encode("utf-8")).digest()
    return values[int.from_bytes(digest[:8], "big") % len(values)]


def progress_records(progress_path):
    """Return the latest API result for every motion/caption key."""
    latest = {}
    if not progress_path.is_file():
        return latest
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            bundle = json.loads(line)
            preserve = result_by_index(bundle.get("preserve", []))
            alter = result_by_index(bundle.get("alter", []))
            for local_index, item in enumerate(bundle.get("items", [])):
                key = (item["motion_id"], item["text_orig"])
                latest[key] = {
                    "item": item,
                    "preserve": preserve.get(local_index, {}),
                    "alter": alter.get(local_index, {}),
                }
    return latest


def valid_preserve_by_level(original, preserve):
    result = {}
    for level in ("P1", "P2", "P3"):
        values = [
            normalize_caption(value)
            for value in candidate_values(preserve, level)
            if basic_filter(original, value)
        ]
        result[level] = list(dict.fromkeys(values))
    return result


def complete_preserve_record(record):
    if not record:
        return False
    original = record["item"]["text_orig"]
    return all(valid_preserve_by_level(original, record["preserve"]).values())


def require_complete_records(records, progress_path, output_path):
    latest = progress_records(progress_path)
    incomplete = []
    for row in records:
        record = latest.get((row["motion_id"], row["text_orig"]))
        if complete_preserve_record(record):
            continue
        valid = (
            valid_preserve_by_level(row["text_orig"], record["preserve"])
            if record else {level: [] for level in ("P1", "P2", "P3")}
        )
        incomplete.append({
            **row,
            "missing_levels": [level for level, values in valid.items() if not values],
        })
    if incomplete:
        failure_path = output_path.with_suffix(
            output_path.suffix + ".incomplete.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(incomplete, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Missing usable P1/P2/P3 candidates for {len(incomplete)} "
            f"source captions; see {failure_path}"
        )


def finalize(progress_path, output_path, protocol, seed, metadata):
    pairs = []
    seen = set()
    incomplete = []
    for record in progress_records(progress_path).values():
        item = record["item"]
        original = item["text_orig"]
        preserve_by_level = valid_preserve_by_level(
            original, record["preserve"])
        missing = [
            level for level, values in preserve_by_level.items()
            if not values
        ]
        if missing:
            incomplete.append({
                "motion_id": item["motion_id"],
                "text_orig": original,
                "missing_levels": missing,
            })
            continue

        if protocol == "test":
            preserve_candidates = [
                (
                    level,
                    deterministic_choice(
                        preserve_by_level[level],
                        f'{item["motion_id"]}:{level}',
                        seed,
                    ),
                )
                for level in ("P1", "P2", "P3")
            ]
        else:
            preserve_candidates = [
                (level, value)
                for level in ("P1", "P2", "P3")
                for value in preserve_by_level[level]
            ]

        for level, text_pair in preserve_candidates:
            key = (item["motion_id"], original, text_pair, "preserve", level)
            if key in seen:
                continue
            seen.add(key)
            pairs.append({
                "motion_id": item["motion_id"],
                "text_orig": original,
                "text_pair": text_pair,
                "relation": "preserve",
                "edit_type": "",
                "level": level,
            })

        if protocol == "train":
            for edit in record["alter"].get("edits", []):
                edit_type = edit.get("edit_type", "")
                text_pair = normalize_caption(edit.get("text", ""))
                if edit_type not in {"action", "direction", "bodypart"}:
                    continue
                if not basic_filter(original, text_pair):
                    continue
                key = (item["motion_id"], original, text_pair, "alter", edit_type)
                if key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "motion_id": item["motion_id"],
                    "text_orig": original,
                    "text_pair": text_pair,
                    "relation": "alter",
                    "edit_type": edit_type,
                    "level": "",
                })

    if incomplete:
        failure_path = output_path.with_suffix(output_path.suffix + ".incomplete.json")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        failure_path.write_text(
            json.dumps(incomplete, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Missing usable P1/P2/P3 candidates for {len(incomplete)} captions; "
            f"see {failure_path}"
        )

    for pair_id, pair in enumerate(pairs, start=1):
        pair["pair_id"] = pair_id

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if protocol == "test":
        output_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    else:
        with output_path.open("w", encoding="utf-8") as handle:
            for pair in pairs:
                handle.write(json.dumps(pair, ensure_ascii=True) + "\n")

    relation_counts = {
        relation: sum(pair["relation"] == relation for pair in pairs)
        for relation in ("preserve", "alter")
    }
    metadata.update({
        "output": str(output_path.resolve()),
        "pairs": len(pairs),
        "relation_counts": relation_counts,
        "unique_motions": len({pair["motion_id"] for pair in pairs}),
        "level_counts": {
            level: sum(pair["level"] == level for pair in pairs)
            for level in ("P1", "P2", "P3")
        },
    })
    output_path.with_suffix(output_path.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


def main():
    args = parse_args()
    random.seed(args.seed)
    progress_path = args.output.with_suffix(args.output.suffix + ".progress.jsonl")
    metadata = {
        "dataset_root": str(args.dataset_root.resolve()),
        "split": args.split,
        "protocol": args.protocol,
        "model": args.model,
        "base_url": args.base_url,
        "preserve_prompt_style": args.preserve_prompt_style,
        "seed": args.seed,
    }

    records = read_motion_captions(
        args.dataset_root,
        args.split,
        one_per_motion=args.protocol == "test",
    )
    if args.limit is not None:
        records = records[:args.limit]
    metadata["source_records"] = len(records)

    if args.finalize_only:
        require_complete_records(records, progress_path, args.output)
        finalize(progress_path, args.output, args.protocol, args.seed, metadata)
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")
    if not args.model:
        raise RuntimeError("Set --model or SATO_PAIR_MODEL")

    progress_path.parent.mkdir(parents=True, exist_ok=True)
    client_args = (api_key, args.base_url, args.model)
    preserve_prompt = (
        HUMANML3D_TEST_PROMPT
        if args.preserve_prompt_style == "humanml3d"
        else PRESERVE_PROMPT
    )

    for completion_round in range(1, args.completion_rounds + 1):
        latest = progress_records(progress_path)
        pending = [
            row for row in records
            if not complete_preserve_record(
                latest.get((row["motion_id"], row["text_orig"])))
        ]
        if not pending:
            break
        batches = [
            pending[i:i + args.batch_size]
            for i in range(0, len(pending), args.batch_size)
        ]
        print(
            f"Completion round {completion_round}/{args.completion_rounds}: "
            f"{len(pending)} captions in {len(batches)} batches",
            flush=True,
        )
        with progress_path.open("a", encoding="utf-8") as output_handle:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [
                    executor.submit(
                        generate_batch, client_args, batch, args.protocol,
                        preserve_prompt, args.max_retries,
                        args.max_output_tokens,
                    )
                    for batch in batches
                ]
                for completed, future in enumerate(as_completed(futures), start=1):
                    bundle = future.result()
                    output_handle.write(json.dumps(bundle, ensure_ascii=True) + "\n")
                    output_handle.flush()
                    print(
                        f"Completed round {completion_round}: "
                        f"{completed}/{len(futures)} batches",
                        flush=True,
                    )

    require_complete_records(records, progress_path, args.output)
    finalize(progress_path, args.output, args.protocol, args.seed, metadata)


if __name__ == "__main__":
    main()
