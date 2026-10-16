"""
Supplement missing motion_ids in test_perb_deterministic.json.
Each motion_id gets exactly ONE P1-level perturbation.
"""

import json
import os
import random
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed

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
MISSING_FILE = os.path.join(ROOT_DIR, "test_perb_missing.json")
DETERMINISTIC_FILE = os.path.join(ROOT_DIR, "test_perb_deterministic.json")
SUPPLEMENT_TMP = os.path.join(ROOT_DIR, "test_perb_supplement.jsonl.tmp")

PRESERVING_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate EXACTLY ONE paraphrase that preserves ALL semantic meaning.

Rules:
- Replace words with synonyms (lexical level paraphrase)
- Keep sentence structure mostly identical
- Preserve ALL semantic content: action, direction, speed, body parts, count, order
- Do NOT change directions such as left/right/forward/backward/clockwise/counterclockwise
- Do NOT change body part references
- Do NOT change action type, step count, or speed
- The paraphrase must be natural, grammatical English

Output JSON:
{"results": [{"index": 0, "paraphrase": "the paraphrased text"}, ...]}"""


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
                max_tokens=4000,
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


def fallback_paraphrase(caption):
    cap = caption.strip()
    if cap.endswith("."):
        cap = cap[:-1]
    if cap[:1].isupper():
        cap = cap[:1].lower() + cap[1:]
    return f"The motion shows {cap}."


def main():
    # Load missing
    with open(MISSING_FILE, "r", encoding="utf-8") as f:
        missing = json.load(f)
    print(f"Missing motion_ids to generate: {len(missing)}")

    # Load existing deterministic to get max pair_id
    with open(DETERMINISTIC_FILE, "r", encoding="utf-8") as f:
        existing = json.load(f)
    next_pair_id = max(d["pair_id"] for d in existing) + 1
    existing_mids = set(d["motion_id"] for d in existing)

    # Filter out any that already exist
    to_generate = [d for d in missing if d["motion_id"] not in existing_mids]
    print(f"After filtering existing: {len(to_generate)} to generate")

    if not to_generate:
        print("Nothing to generate.")
        return

    # Load resume state
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

    if not to_generate:
        print("All already done. Merging...")
    else:
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
                        para = None

                        if i < len(results) and isinstance(results[i], dict):
                            para = results[i].get("paraphrase", "").strip()

                        if not basic_filter(cap, para):
                            para = fallback_paraphrase(cap)
                            level = "fallback"
                        else:
                            level = "P1"

                        next_pair_id += 1
                        batch_written += 1
                        total_new += 1
                        item = {
                            "pair_id": next_pair_id,
                            "motion_id": mid,
                            "text_orig": cap,
                            "text_pair": para,
                            "relation": "preserve",
                            "edit_type": "",
                            "level": level,
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

    # Merge with existing and save
    all_pairs = existing + new_pairs
    # Deduplicate by motion_id (keep first occurrence from existing)
    seen = set()
    deduped = []
    for p in all_pairs:
        if p["motion_id"] not in seen:
            seen.add(p["motion_id"])
            deduped.append(p)

    deduped.sort(key=lambda x: x["pair_id"])
    # Re-number pair_ids
    for i, p in enumerate(deduped):
        p["pair_id"] = i + 1

    from collections import Counter
    level_dist = Counter(p["level"] for p in deduped)
    print(f"\nMerged total: {len(deduped)} pairs, {len(seen)} unique motion_ids")
    print(f"Level distribution: {dict(sorted(level_dist.items()))}")

    with open(DETERMINISTIC_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)
    print(f"Saved to {DETERMINISTIC_FILE}")


if __name__ == "__main__":
    main()
