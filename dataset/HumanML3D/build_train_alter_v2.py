"""
Generate alter pairs for all remaining captions.
Three alter types: direction swap, bodypart swap, action swap.
At least one alter type MUST be applied to every caption.
"""

import json
import os
import random
import re
import sys
import time
from collections import Counter
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
MISSING_FILE = os.path.join(ROOT_DIR, "train_alter_missing_v2.json")
SUPPLEMENT_TMP = os.path.join(ROOT_DIR, "train_alter_v2.jsonl.tmp")
OUTPUT_JSON = os.path.join(ROOT_DIR, "train_alter_v2.json")

ALTERING_PROMPT = """You are a motion description editing expert. For each input motion description, create EXACTLY ONE edit that changes the motion semantics.

You MUST choose one of these edit types (pick the MOST appropriate one):

1. **direction**: swap spatial directions
   - left ↔ right, forward ↔ backward, clockwise ↔ counterclockwise, up ↔ down, inward ↔ outward, sideways left ↔ sideways right, ascending ↔ descending

2. **bodypart**: swap lateral body parts
   - left hand ↔ right hand, left arm ↔ right arm, left leg ↔ right leg, left foot ↔ right foot, left knee ↔ right knee, left elbow ↔ right elbow

3. **action**: replace the main action/verb with a DIFFERENT action that produces a visibly different motion
   - walk → run/skip/hop, kick → throw/punch, jump → crouch/dodge, wave → point/reach, sit → squat/kneel, turn → spin/twist, bend → stretch/arch, push → pull/lift, clap → snap/stomp, etc.
   - The replacement should be a clearly different body movement, not just a synonym.

Rules:
- Change ONLY the target attribute — everything else must stay the same
- Make minimal edits — change as few words as possible
- Pick the edit type that produces the most natural, meaningful change
- Prefer direction or bodypart when they apply; use action as the fallback for captions without directional/bodypart references
- EVERY caption MUST get an edit — there are no exceptions

Output JSON:
{"results": [{"index": 0, "edit_type": "direction" or "bodypart" or "action", "edited": "the modified caption"}, ...]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def call_llm(batch_captions, max_retries=3):
    items = "\n".join(f'{i + 1}. "{cap}"' for i, cap in enumerate(batch_captions))
    content = f"Edit these {len(batch_captions)} motion descriptions — each one MUST be changed:\n\n{items}"

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
    return ol * 0.3 <= pl <= ol * 3.0


def fallback_action_swap(caption):
    """Rule-based fallback: replace common verbs."""
    swaps = [
        ("walk", "run"), ("walks", "runs"), ("walking", "running"), ("walked", "ran"),
        ("jump", "crouch"), ("jumps", "crouches"), ("jumping", "crouching"), ("jumped", "crouched"),
        ("run", "walk"), ("runs", "walks"), ("running", "walking"), ("ran", "walked"),
        ("kick", "throw"), ("kicks", "throws"), ("kicking", "throwing"), ("kicked", "threw"),
        ("sit", "squat"), ("sits", "squats"), ("sitting", "squatting"), ("sat", "squatted"),
        ("wave", "point"), ("waves", "points"), ("waving", "pointing"), ("waved", "pointed"),
        ("turn", "spin"), ("turns", "spins"), ("turning", "spinning"), ("turned", "spun"),
        ("bend", "stretch"), ("bends", "stretches"), ("bending", "stretching"), ("bent", "stretched"),
        ("push", "pull"), ("pushes", "pulls"), ("pushing", "pulling"), ("pushed", "pulled"),
        ("clap", "stomp"), ("claps", "stomps"), ("clapping", "stomping"), ("clapped", "stomped"),
        ("raise", "lower"), ("raises", "lowers"), ("raising", "lowering"), ("raised", "lowered"),
        ("stand", "kneel"), ("stands", "kneels"), ("standing", "kneeling"), ("stood", "knelt"),
        ("dance", "exercise"), ("dances", "exercises"), ("dancing", "exercising"), ("danced", "exercised"),
        ("punch", "slap"), ("punches", "slaps"), ("punching", "slapping"), ("punched", "slapped"),
        ("throw", "catch"), ("throws", "catches"), ("throwing", "catching"), ("threw", "caught"),
        ("lift", "drop"), ("lifts", "drops"), ("lifting", "dropping"), ("lifted", "dropped"),
        ("swim", "crawl"), ("swims", "crawls"), ("swimming", "crawling"), ("swam", "crawled"),
    ]
    cap_lower = caption.lower()
    for orig_verb, new_verb in swaps:
        # Case-sensitive replacement
        if orig_verb in cap_lower:
            # Find the actual case
            idx = cap_lower.index(orig_verb)
            actual = caption[idx:idx + len(orig_verb)]
            if actual[0].isupper():
                new_verb = new_verb[0].upper() + new_verb[1:]
            return caption[:idx] + new_verb + caption[idx + len(orig_verb):], "action"
    return None, None


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
                        edit_type = r.get("edit_type", "")

                        if not basic_filter(cap, edited):
                            # Try rule-based fallback
                            fallback, fb_type = fallback_action_swap(cap)
                            if fallback:
                                edited = fallback
                                edit_type = fb_type
                            else:
                                # Last resort: prefix with "does not"
                                edited = f"does not {cap.lower().rstrip('.')}"
                                edit_type = "action"

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
    edit_types = Counter(p["edit_type"] for p in new_pairs)
    print(f"\nGenerated: {len(new_pairs)} total")
    print(f"Edit types: {dict(sorted(edit_types.items()))}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(new_pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
