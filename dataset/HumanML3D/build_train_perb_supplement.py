"""
Supplement missing motion_ids in preserving_pairs_filtered.jsonl.
Each motion_id gets P1/P2/P3 paraphrases (one each).
Output merges into a new file: preserving_pairs_supplement.jsonl
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

random.seed(42)

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = "gemini-3.1-flash-lite-preview"
BATCH_SIZE = 10
MAX_WORKERS = 16

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MISSING_FILE = os.path.join(ROOT_DIR, "train_perb_missing.json")
EXISTING_FILE = os.path.join(ROOT_DIR, "preserving_pairs_filtered.jsonl")
OUTPUT_FILE = os.path.join(ROOT_DIR, "preserving_pairs_supplement.jsonl")
MERGED_FILE = os.path.join(ROOT_DIR, "preserving_pairs_full.jsonl")
SUPPLEMENT_TMP = os.path.join(ROOT_DIR, "train_perb_supplement.jsonl.tmp")

PRESERVING_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate EXACTLY 1 paraphrase at EACH of 3 levels (3 total).

P1 (Lexical): Replace words with synonyms. Keep sentence structure mostly identical.
P2 (Syntactic): Change sentence structure, word order, or clause order.
P3 (Scene/Narrative): Rewrite in a different narrative style.

Rules:
- Preserve ALL semantic content: action, direction, speed, body parts, count, order
- Do NOT change directions such as left/right/forward/backward/clockwise/counterclockwise
- Do NOT change body part references
- Do NOT change action type, step count, or speed
- Each paraphrase must be natural, grammatical English

Output JSON:
{"results": [{"index": 0, "P1": "paraphrase1", "P2": "paraphrase2", "P3": "paraphrase3"}, ...]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(batch_captions, max_retries=3):
    items = "\n".join(f'{i + 1}. "{cap}"' for i, cap in enumerate(batch_captions))
    content = f"Paraphrase these {len(batch_captions)} motion descriptions:\n\n{items}"

    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": PRESERVING_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.7,
                max_tokens=6000,
                response_format={"type": "json_object"},
            )
            text = resp.choices[0].message.content
            if "```" in text:
                match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
                if match:
                    text = match.group(1)
            result = json.loads(text)
            return result.get("results", [])
        except Exception as exc:
            if attempt == max_retries - 1:
                print(f"    LLM batch failed: {exc}")
                return []
            time.sleep(3 * (attempt + 1))


def basic_filter(orig, para):
    if not para or not para.strip():
        return False
    o = orig.lower().rstrip(".")
    p = para.lower().rstrip(".")
    if o == p:
        return False
    ol, pl = len(orig.split()), len(para.split())
    return ol * 0.3 <= pl <= ol * 3.0


def fallback_paraphrase(caption, level):
    cap = caption.strip()
    if cap.endswith("."):
        cap = cap[:-1]
    if cap[:1].isupper():
        cap = cap[:1].lower() + cap[1:]
    templates = {
        "P1": f"The motion shows {cap}.",
        "P2": f"What is shown is {cap}.",
        "P3": f"This animation depicts a scene where {cap}.",
    }
    return templates.get(level, f"The motion shows {cap}.")


def main():
    # Load missing
    with open(MISSING_FILE, "r", encoding="utf-8") as f:
        missing = json.load(f)
    print(f"Missing motion_ids: {len(missing)}")

    # Load existing to get pair_ids and dedup
    existing_mids = set()
    max_pair_id = 0
    with open(EXISTING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            existing_mids.add(p["motion_id"])
            pid = p.get("pair_id", 0)
            if isinstance(pid, int) and pid > max_pair_id:
                max_pair_id = pid

    # Filter already-done
    to_generate = [d for d in missing if d["motion_id"] not in existing_mids]
    print(f"After filtering existing: {len(to_generate)} to generate")

    if not to_generate:
        print("Nothing to generate. Merging only.")
    else:
        # Resume support
        done_mids = set()
        new_pairs = []
        if os.path.exists(SUPPLEMENT_TMP):
            with open(SUPPLEMENT_TMP, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    new_pairs.append(item)
                    done_mids.add(item["motion_id"])

        to_generate = [d for d in to_generate if d["motion_id"] not in done_mids]
        print(f"Already supplemented: {len(done_mids)}, remaining: {len(to_generate)}")

        if to_generate:
            batches = [
                (i // BATCH_SIZE + 1, to_generate[i:i + BATCH_SIZE])
                for i in range(0, len(to_generate), BATCH_SIZE)
            ]
            print(f"Batches: {len(batches)}, batch_size={BATCH_SIZE}, workers={MAX_WORKERS}")

            total_new = 0
            start_time = time.time()

            with open(SUPPLEMENT_TMP, "a", encoding="utf-8") as out_f:
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = {}
                    for batch in batches:
                        batch_idx, batch_data = batch
                        captions = [d["text_orig"] for d in batch_data]
                        future = executor.submit(call_llm, captions)
                        futures[future] = batch

                    for future in as_completed(futures):
                        batch_idx, batch_data = futures[future]
                        results = future.result()
                        batch_written = 0

                        for i, d in enumerate(batch_data):
                            mid = d["motion_id"]
                            cap = d["text_orig"]
                            r = results[i] if i < len(results) and isinstance(results[i], dict) else {}

                            for level in ["P1", "P2", "P3"]:
                                para = r.get(level, "").strip()
                                if not basic_filter(cap, para):
                                    para = fallback_paraphrase(cap, level)
                                    eff_level = "fallback"
                                else:
                                    eff_level = level

                                max_pair_id += 1
                                batch_written += 1
                                total_new += 1
                                item = {
                                    "pair_id": max_pair_id,
                                    "motion_id": mid,
                                    "text_orig": cap,
                                    "text_pair": para,
                                    "relation": "preserve",
                                    "edit_type": "",
                                    "level": eff_level,
                                }
                                new_pairs.append(item)
                                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

                        out_f.flush()
                        elapsed = time.time() - start_time
                        rate = total_new / max(elapsed, 1)
                        print(
                            f"Batch {batch_idx}/{len(batches)}: +{batch_written}, "
                            f"total_new={total_new}, rate={rate:.1f}/s"
                        )

    # ---- Merge: existing + supplement ----
    print(f"\n=== Merging ===")
    all_lines = []

    # Copy existing
    with open(EXISTING_FILE, "r", encoding="utf-8") as f:
        for line in f:
            all_lines.append(line.rstrip())

    # Append new
    with open(SUPPLEMENT_TMP, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_lines.append(line)

    with open(MERGED_FILE, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    # Stats
    level_dist = Counter()
    mid_set = set()
    total = 0
    for line in all_lines:
        p = json.loads(line)
        level_dist[p.get("level", "?")] += 1
        mid_set.add(p["motion_id"])
        total += 1

    print(f"Merged: {total} pairs, {len(mid_set)} unique motion_ids")
    print(f"Level distribution: {dict(sorted(level_dist.items()))}")
    print(f"Output: {MERGED_FILE}")


if __name__ == "__main__":
    main()
