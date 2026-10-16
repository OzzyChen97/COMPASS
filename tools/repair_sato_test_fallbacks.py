#!/usr/bin/env python
"""Regenerate only missing P1/P2/P3 outputs in SATO test progress."""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from build_sato_pairs import (
    PRESERVE_PROMPT,
    basic_filter,
    call_json,
    normalize_caption,
    result_by_index,
)


LEVELS = ("P1", "P2", "P3")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SATO_PAIR_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--max-fallback-rate", type=float, default=0.02)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_bundles(path):
    bundles = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                bundles.append(json.loads(line))
    return bundles


def missing_records(bundles):
    records = []
    total_items = 0
    for bundle_index, bundle in enumerate(bundles):
        preserve = result_by_index(bundle.get("preserve", []))
        for local_index, item in enumerate(bundle.get("items", [])):
            total_items += 1
            original = normalize_caption(item["text_orig"])
            result = preserve.get(local_index, {})
            missing = []
            for level in LEVELS:
                valid = [
                    value for value in result.get(level, [])
                    if basic_filter(original, value)
                ]
                if not valid:
                    missing.append(level)
            if missing:
                records.append({
                    "bundle_index": bundle_index,
                    "local_index": local_index,
                    "motion_id": item["motion_id"],
                    "text_orig": original,
                    "missing_levels": missing,
                })
    return records, total_items


def generate_repair_batch(client_args, batch, max_retries):
    api_key, base_url, model = client_args
    client = OpenAI(api_key=api_key, base_url=base_url)
    prompt = PRESERVE_PROMPT + (
        " This is a repair pass. Always provide non-identical valid outputs "
        "for every P1, P2, and P3 field for every input item."
    )
    results = call_json(client, model, prompt, batch, max_retries)
    return batch, result_by_index(results)


def merge_results(bundles, batch, repaired):
    repaired_levels = 0
    for batch_index, record in enumerate(batch):
        candidate = repaired.get(batch_index, {})
        bundle = bundles[record["bundle_index"]]
        preserve = result_by_index(bundle.get("preserve", []))
        target = preserve.get(record["local_index"])
        if target is None:
            target = {"index": record["local_index"] + 1}
            bundle.setdefault("preserve", []).append(target)

        for level in record["missing_levels"]:
            valid = [
                normalize_caption(value)
                for value in candidate.get(level, [])
                if basic_filter(record["text_orig"], value)
            ]
            if valid:
                target[level] = valid
                repaired_levels += 1
    return repaired_levels


def write_bundles_atomic(path, bundles):
    temp_path = path.with_suffix(path.suffix + ".repair.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for bundle in bundles:
            handle.write(json.dumps(bundle, ensure_ascii=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_path, path)


def main():
    args = parse_args()
    bundles = load_bundles(args.progress)
    missing, total_items = missing_records(bundles)
    initial_missing = sum(len(record["missing_levels"]) for record in missing)
    total_levels = total_items * len(LEVELS)
    print(json.dumps({
        "items": total_items,
        "total_levels": total_levels,
        "missing_levels": initial_missing,
        "fallback_rate": initial_missing / total_levels if total_levels else 1.0,
    }, indent=2), flush=True)
    if args.dry_run:
        return

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in {args.api_key_env}")
    client_args = (api_key, args.base_url, args.model)

    for round_index in range(1, args.max_rounds + 1):
        missing, total_items = missing_records(bundles)
        missing_count = sum(len(record["missing_levels"]) for record in missing)
        fallback_rate = missing_count / (total_items * len(LEVELS))
        if fallback_rate <= args.max_fallback_rate:
            break

        batches = [
            missing[index:index + args.batch_size]
            for index in range(0, len(missing), args.batch_size)
        ]
        repaired_levels = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    generate_repair_batch, client_args, batch, args.max_retries
                )
                for batch in batches
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                batch, repaired = future.result()
                repaired_levels += merge_results(bundles, batch, repaired)
                print(
                    f"Round {round_index}: completed {completed}/{len(futures)} "
                    f"batches, repaired_levels={repaired_levels}",
                    flush=True,
                )
        write_bundles_atomic(args.progress, bundles)

    missing, total_items = missing_records(bundles)
    remaining = sum(len(record["missing_levels"]) for record in missing)
    final_rate = remaining / (total_items * len(LEVELS))
    summary = {
        "initial_missing_levels": initial_missing,
        "remaining_missing_levels": remaining,
        "fallback_rate": final_rate,
        "threshold": args.max_fallback_rate,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if final_rate > args.max_fallback_rate:
        raise RuntimeError(
            f"Fallback rate remains {final_rate:.4%}, above {args.max_fallback_rate:.4%}"
        )


if __name__ == "__main__":
    main()
