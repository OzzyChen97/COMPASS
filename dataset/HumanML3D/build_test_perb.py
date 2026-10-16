"""
Build semantic-preserving perturbation data for the HumanML3D test split.

Output:
  test_perb.json       JSON array, requested final artifact
  test_perb.jsonl.tmp  append-only resume file
"""

import argparse
import codecs as cs
import json
import os
import random
import re
import sys
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from os.path import join as pjoin

from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

random.seed(42)

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = "gemini-3.1-flash-lite-preview"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEXT_DIR = pjoin(ROOT_DIR, "texts")
TEST_FILE = pjoin(ROOT_DIR, "test.txt")
OUTPUT_JSON = pjoin(ROOT_DIR, "test_perb.json")
OUTPUT_TMP = OUTPUT_JSON + "l.tmp"

PRESERVING_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate EXACTLY 2 paraphrases at EACH of 3 levels (6 total).

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

Output JSON:
{"results": [{"index": 0, "P1": ["p1","p2"], "P2": ["p1","p2"], "P3": ["p1","p2"]}, ...]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def collect_test_captions():
    with open(TEST_FILE, "r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]

    captions = OrderedDict()
    for mid in ids:
        path = pjoin(TEXT_DIR, mid + ".txt")
        if not os.path.exists(path):
            continue
        with cs.open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("#")
                if not parts:
                    continue
                cap = parts[0].strip()
                if cap and len(cap) > 0 and cap not in captions:
                    captions[cap] = mid
    return captions


def load_done(path):
    done = set()
    pairs = []
    if not os.path.exists(path):
        return done, pairs

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            pairs.append(item)
            done.add(item.get("text_orig", ""))
    return done, pairs


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
                max_tokens=8000,
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


def generate_batch(batch_data):
    batch_idx, batch_captions = batch_data
    results = call_llm(batch_captions)
    return batch_idx, batch_captions, results


def write_final_json(tmp_path, output_path):
    pairs = []
    seen = set()
    with open(tmp_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            key = (
                item.get("motion_id"),
                item.get("text_orig"),
                item.get("text_pair"),
                item.get("level"),
            )
            if key in seen:
                continue
            seen.add(key)
            item["pair_id"] = len(pairs) + 1
            pairs.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=OUTPUT_JSON)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()

    output_json = os.path.abspath(args.output)
    tmp_path = output_json + "l.tmp"

    if args.finalize_only:
        pairs = write_final_json(tmp_path, output_json)
        print(f"Finalized {len(pairs)} pairs -> {output_json}")
        return

    captions_by_text = collect_test_captions()
    all_captions = list(captions_by_text.keys())
    if args.limit is not None:
        all_captions = all_captions[:args.limit]

    done, existing_pairs = load_done(tmp_path)
    todo = [cap for cap in all_captions if cap not in done]

    print(f"Test unique captions: {len(all_captions)}")
    print(f"Already done captions: {len(done)}")
    print(f"Remaining captions: {len(todo)}")
    print(f"Output tmp: {tmp_path}")
    print(f"Output json: {output_json}")

    batches = [
        (i // args.batch_size + 1, todo[i:i + args.batch_size])
        for i in range(0, len(todo), args.batch_size)
    ]
    print(f"Batches: {len(batches)}, batch_size={args.batch_size}, workers={args.workers}")

    next_pair_id = len(existing_pairs)
    total_new = 0
    start_time = time.time()

    with open(tmp_path, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(generate_batch, batch): batch[0]
                for batch in batches
            }

            for future in as_completed(futures):
                batch_idx, batch_caps, results = future.result()
                batch_written = 0
                for i, cap in enumerate(batch_caps):
                    # The prompt includes an index field, but different models may
                    # return it as 0-based or 1-based. The list order has been more
                    # reliable and matches the existing dataset builders.
                    r = results[i] if i < len(results) and isinstance(results[i], dict) else {}
                    wrote_for_caption = False
                    for level in ["P1", "P2", "P3"]:
                        for para in r.get(level, []):
                            para = str(para).strip()
                            if not basic_filter(cap, para):
                                continue
                            next_pair_id += 1
                            batch_written += 1
                            total_new += 1
                            wrote_for_caption = True
                            out_f.write(json.dumps({
                                "pair_id": next_pair_id,
                                "motion_id": captions_by_text[cap],
                                "text_orig": cap,
                                "text_pair": para,
                                "relation": "preserve",
                                "edit_type": "",
                                "level": level,
                            }, ensure_ascii=False) + "\n")

                    if not wrote_for_caption:
                        next_pair_id += 1
                        batch_written += 1
                        total_new += 1
                        out_f.write(json.dumps({
                            "pair_id": next_pair_id,
                            "motion_id": captions_by_text[cap],
                            "text_orig": cap,
                            "text_pair": fallback_paraphrase(cap),
                            "relation": "preserve",
                            "edit_type": "fallback",
                            "level": "fallback",
                        }, ensure_ascii=False) + "\n")

                out_f.flush()
                elapsed = time.time() - start_time
                done_now = len(done) + sum(1 for cap in todo[:0])
                rate = total_new / max(elapsed, 1)
                print(
                    f"Batch {batch_idx}/{len(batches)}: +{batch_written} pairs, "
                    f"new_total={total_new}, rate={rate:.1f} pairs/s"
                )

    pairs = write_final_json(tmp_path, output_json)
    unique_orig = len({p["text_orig"] for p in pairs})
    fallback = sum(1 for p in pairs if p.get("level") == "fallback")
    print(f"Done. pairs={len(pairs)}, unique_text_orig={unique_orig}, fallback_pairs={fallback}")
    print(f"Output: {output_json}")


if __name__ == "__main__":
    main()
