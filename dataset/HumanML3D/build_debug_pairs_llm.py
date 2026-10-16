"""
SATO++ Debug Pair Generator (LLM-based, with altering)

Generates preserving + altering pairs via LLM for pipeline debugging.
~350 preserving + ~150 altering = 500 total.

Usage:
    python build_debug_pairs_llm.py
"""

import json
import os
import codecs as cs
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from os.path import join as pjoin

random.seed(42)

API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = "gpt-4o-mini"
BATCH_SIZE = 25
MAX_WORKERS = 5

TEXT_DIR = './dataset/HumanML3D/texts'
TRAIN_FILE = './dataset/HumanML3D/train.txt'
OUTPUT = './dataset/HumanML3D/preserving_pairs_debug.jsonl'

PRESERVING_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate EXACTLY 2 paraphrases that preserve ALL semantic meaning (action, direction, speed, body parts, count, order).

Rules:
- Keep ALL directions unchanged (left/right/forward/backward)
- Keep ALL body parts unchanged
- Keep action, count, order unchanged
- Only change: synonyms for verbs, descriptive words, sentence structure
- Each paraphrase must be natural English

Output JSON:
{"results": [{"index": 0, "paraphrases": ["p1", "p2"]}, ...]}"""

ALTERING_PROMPT = """You are a motion description editing expert. For each input motion description, create EXACTLY ONE edit that changes the motion semantics.

Choose ONE of these edit types:
- direction: swap left↔right, forward↔backward, clockwise↔counterclockwise
- bodypart: swap left hand↔right hand, left arm↔right arm, left leg↔right leg, left foot↔right foot

Rules:
- Change ONLY the target attribute (e.g., only direction, nothing else)
- Make minimal edits — change as few words as possible
- The rest of the sentence must remain exactly the same
- Pick the edit type that produces the most natural result

Output JSON:
{"results": [{"index": 0, "edit_type": "direction" or "bodypart", "edited": "the modified caption"}], ...]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def collect_captions():
    """Collect unique captions from train split, pick diverse subset."""
    with open(TRAIN_FILE) as f:
        ids = [l.strip() for l in f.readlines()]

    captions = {}  # caption -> motion_id
    for mid in ids:
        path = pjoin(TEXT_DIR, mid + '.txt')
        if not os.path.exists(path):
            continue
        with cs.open(path) as f:
            for line in f:
                parts = line.strip().split('#')
                cap = parts[0].strip()
                if cap and len(cap) > 20 and cap not in captions:
                    captions[cap] = mid

    all_caps = list(captions.items())
    random.shuffle(all_caps)
    return all_caps


def call_llm(system_prompt, user_content, max_retries=3):
    """Call LLM with retries."""
    for attempt in range(max_retries):
        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"  LLM call FAILED: {e}")
                return None
            time.sleep(2)


def generate_preserving_batch(batch_captions, batch_idx):
    """Generate preserving pairs for a batch."""
    items = "\n".join(f'{i+1}. "{cap}"' for i, (cap, _) in enumerate(batch_captions))
    content = f"Paraphrase these {len(batch_captions)} motion descriptions:\n\n{items}"
    result = call_llm(PRESERVING_PROMPT, content)
    return batch_idx, result


def generate_altering_batch(batch_captions, batch_idx):
    """Generate altering pairs for a batch."""
    items = "\n".join(f'{i+1}. "{cap}"' for i, (cap, _) in enumerate(batch_captions))
    content = f"Edit these {len(batch_captions)} motion descriptions (change direction or bodypart):\n\n{items}"
    result = call_llm(ALTERING_PROMPT, content)
    return batch_idx, result


def basic_filter(orig, para):
    """Quick sanity check."""
    if not para or not para.strip():
        return False
    o = orig.lower().rstrip('.')
    p = para.lower().rstrip('.')
    if o == p:
        return False
    ol, pl = len(orig.split()), len(para.split())
    return ol * 0.3 <= pl <= ol * 3.0


def main():
    all_caps = collect_captions()
    print(f"Total unique captions: {len(all_caps)}")

    # Split: ~200 for preserving (2 pairs each → ~400), ~150 for altering
    preserving_caps = all_caps[:200]
    altering_caps = all_caps[200:350]

    all_pairs = []
    pair_id = 0

    # ======== Preserving ========
    print(f"\n=== Generating preserving pairs ({len(preserving_caps)} captions) ===")
    batches = [(i // BATCH_SIZE + 1, preserving_caps[i:i + BATCH_SIZE])
               for i in range(0, len(preserving_caps), BATCH_SIZE)]
    print(f"  Batches: {len(batches)}, Workers: {MAX_WORKERS}")

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_preserving_batch, caps, idx): idx
                   for idx, caps in batches}

        for future in as_completed(futures):
            batch_idx, result = future.result()
            if result is None:
                continue
            batch = batches[batch_idx - 1][1]

            for i, (cap, mid) in enumerate(batch):
                if i >= len(result.get('results', [])):
                    continue
                r = result['results'][i]
                for para in r.get('paraphrases', []):
                    if basic_filter(cap, para):
                        pair_id += 1
                        all_pairs.append({
                            'pair_id': pair_id,
                            'motion_id': mid,
                            'text_orig': cap,
                            'text_pair': para.strip(),
                            'relation': 'preserve',
                            'edit_type': None,
                            'level': 'P1',
                        })

            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx}/{len(batches)} done, "
                  f"total preserving: {sum(1 for p in all_pairs if p['relation']=='preserve')}, "
                  f"elapsed: {elapsed:.0f}s")

    n_preserve = sum(1 for p in all_pairs if p['relation'] == 'preserve')
    print(f"  Preserving total: {n_preserve}")

    # ======== Altering ========
    print(f"\n=== Generating altering pairs ({len(altering_caps)} captions) ===")
    batches = [(i // BATCH_SIZE + 1, altering_caps[i:i + BATCH_SIZE])
               for i in range(0, len(altering_caps), BATCH_SIZE)]

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(generate_altering_batch, caps, idx): idx
                   for idx, caps in batches}

        for future in as_completed(futures):
            batch_idx, result = future.result()
            if result is None:
                continue
            batch = batches[batch_idx - 1][1]

            for i, (cap, mid) in enumerate(batch):
                if i >= len(result.get('results', [])):
                    continue
                r = result['results'][i]
                edited = r.get('edited', '').strip()
                edit_type = r.get('edit_type', 'direction')
                if basic_filter(cap, edited):
                    pair_id += 1
                    all_pairs.append({
                        'pair_id': pair_id,
                        'motion_id': mid,
                        'text_orig': cap,
                        'text_pair': edited,
                        'relation': 'alter',
                        'edit_type': edit_type,
                    })

            elapsed = time.time() - start_time
            print(f"  Batch {batch_idx}/{len(batches)} done, "
                  f"total altering: {sum(1 for p in all_pairs if p['relation']=='alter')}, "
                  f"elapsed: {elapsed:.0f}s")

    n_alter = sum(1 for p in all_pairs if p['relation'] == 'alter')
    print(f"  Altering total: {n_alter}")

    # ======== Write ========
    print(f"\n=== Total: {len(all_pairs)} pairs ({n_preserve} preserve + {n_alter} alter) ===")

    with open(OUTPUT, 'w', encoding='utf-8') as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f"Written to: {OUTPUT}")

    # Examples
    print("\n=== Preserving examples ===")
    for p in [x for x in all_pairs if x['relation'] == 'preserve'][:3]:
        print(f"  ORIG: {p['text_orig'][:80]}")
        print(f"  PAIR: {p['text_pair'][:80]}")
        print()

    print("=== Altering examples ===")
    for p in [x for x in all_pairs if x['relation'] == 'alter'][:3]:
        print(f"  ORIG: {p['text_orig'][:80]}")
        print(f"  PAIR: {p['text_pair'][:80]}")
        print(f"  TYPE: {p['edit_type']}")
        print()


if __name__ == '__main__':
    main()
