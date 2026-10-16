"""
Generate alter pairs (direction/bodypart swap) for all missing training captions.
Output: train_alter_supplement.jsonl (alter data for merging into txt files)
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

random.seed(42)

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = "gemini-3.1-flash-lite-preview"
BATCH_SIZE = 20
MAX_WORKERS = 16

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MISSING_FILE = os.path.join(ROOT_DIR, "train_alter_missing.json")
SUPPLEMENT_TMP = os.path.join(ROOT_DIR, "train_alter_supplement.jsonl.tmp")
OUTPUT_JSON = os.path.join(ROOT_DIR, "train_alter_supplement.json")

ALTERING_PROMPT = """You are a motion description editing expert. For each input motion description, create EXACTLY ONE edit that changes the motion semantics.

Choose ONE of these edit types:
- direction: swap left↔right, forward↔backward, clockwise↔counterclockwise, up↔down, inward↔outward
- bodypart: swap left hand↔right hand, left arm↔right arm, left leg↔right leg, left foot↔right foot, left knee↔right knee, left elbow↔right elbow

Rules:
- Change ONLY the target attribute (e.g., only direction, nothing else)
- Make minimal edits — change as few words as possible
- The rest of the sentence must remain exactly the same
- Pick the edit type that produces the most natural result
- If the caption has NO directional or bodypart references that can be swapped, output {"index": N, "edit_type": "none", "edited": ""}

Output JSON:
{"results": [{"index": 0, "edit_type": "direction" or "bodypart" or "none", "edited": "the modified caption"}, ...]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(batch_captions, max_retries=3):
    items = "\n".join(f'{i + 1}. "{cap}"' for i, cap in enumerate(batch_captions))
    content = f"Edit these {len(batch_captions)} motion descriptions (change direction or bodypart):\n\n{items}"

    for attempt in range(max_retries):
        try:
            resp = get_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": ALTERING_PROMPT},
                    {"role": "user", "content": content},
                ],
                temperature=0.3,
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
    return ol * 0.5 <= pl <= ol * 2.0


def main():
    with open(MISSING_FILE, "r", encoding="utf-8") as f:
        missing = json.load(f)
    print(f"Missing captions needing alter: {len(missing)}")

    # Resume support
    done_keys = set()
    new_pairs = []
    if os.path.exists(SUPPLEMENT_TMP):
        with open(SUPPLEMENT_TMP, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                new_pairs.append(item)
                done_keys.add((item["motion_id"], item["text_orig"]))

    to_generate = [d for d in missing if (d["motion_id"], d["text_orig"]) not in done_keys]
    print(f"Already done: {len(done_keys)}, remaining: {len(to_generate)}")

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

                        edited = r.get("edited", "").strip()
                        edit_type = r.get("edit_type", "none")

                        if edit_type == "none" or not basic_filter(cap, edited):
                            # No valid alter possible for this caption
                            batch_written += 1
                            total_new += 1
                            item = {
                                "motion_id": mid,
                                "text_orig": cap,
                                "text_pair": "",
                                "relation": "alter",
                                "edit_type": "none",
                            }
                        else:
                            batch_written += 1
                            total_new += 1
                            item = {
                                "motion_id": mid,
                                "text_orig": cap,
                                "text_pair": edited,
                                "relation": "alter",
                                "edit_type": edit_type,
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

    # Stats
    has_alter = sum(1 for p in new_pairs if p["text_pair"])
    no_alter = sum(1 for p in new_pairs if not p["text_pair"])
    edit_types = Counter(p["edit_type"] for p in new_pairs)
    print(f"\nGenerated: {len(new_pairs)} total")
    print(f"  Has alter: {has_alter}")
    print(f"  No alter (none): {no_alter}")
    print(f"  Edit types: {dict(sorted(edit_types.items()))}")

    # Save as JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
