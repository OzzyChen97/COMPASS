"""
SATO++ Full Pair Dataset Builder

Generates complete preserving (P1/P2/P3) + altering pairs for HumanML3D.

Preserving levels:
  P1 (Lexical):   Synonym replacement, same sentence structure
  P2 (Syntactic): Word order change, active/passive, clause reordering
  P3 (Scene):     Narrative rewrite, different phrasing style

Altering types:
  direction: left↔right, forward↔backward, clockwise↔counterclockwise
  bodypart:  left hand↔right hand, left leg↔right leg, etc.

Usage:
    # Step 1: Generate all pairs (preserving + altering)
    python build_full_pairs.py

    # Step 2: Filter with LLM judge (optional but recommended)
    python build_full_pairs.py --filter-only

Output:
    dataset/HumanML3D/preserving_pairs.jsonl          (raw)
    dataset/HumanML3D/preserving_pairs_filtered.jsonl  (filtered)
"""  # noqa: E501

import sys
import json
import os
import codecs as cs
import time
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from os.path import join as pjoin

# Force stdout/stderr to be unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
from collections import OrderedDict

# ======== Config ========
API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = "gemini-3.1-flash-lite-preview"
FILTER_MODEL = "gemini-3.1-flash-lite-preview"
BATCH_SIZE = 1
MAX_WORKERS = 16

TEXT_DIR = './dataset/HumanML3D/texts'
TRAIN_FILE = './dataset/HumanML3D/train.txt'
OUTPUT_RAW = './dataset/HumanML3D/preserving_pairs.jsonl'
OUTPUT_FILTERED = './dataset/HumanML3D/preserving_pairs_filtered.jsonl'

# Target counts (None = use all available captions)
N_PRESERVE_CAPS = None     # All captions for preserving (2 paraphrases × 3 levels = 6 pairs each)
N_ALTER_CAPS = None        # All remaining captions for altering (1 edit each)

# ======== Prompts ========

PRESERVING_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate EXACTLY 2 paraphrases at EACH of 3 levels (6 total).

**P1 (Lexical)**: Replace words with synonyms. Keep sentence structure identical.
Example: "A person walks forward slowly." → "A human strolls ahead at a slow pace."

**P2 (Syntactic)**: Change sentence structure (word order, clause reordering, active/passive). Must still be grammatical.
Example: "A person walks forward and then turns left." → "After walking forward, a person turns to the left."
Example: "The man raises his left hand slowly." → "Slowly, the left hand is raised by the man."

**P3 (Scene/Narrative)**: Rewrite in a different narrative style or perspective.
Example: "A person walks forward slowly." → "An individual can be seen moving in a forward direction at a leisurely pace."

Rules:
- Preserve ALL semantic content: action, direction, speed, body parts, count, order
- Do NOT change any direction (left/right/forward/backward/clockwise)
- Do NOT change any body part reference
- Do NOT change action type, step count, or speed
- Each paraphrase must be natural, grammatical English
- "in reverse" = backward, "ahead" = forward — these are acceptable

Output JSON:
{"results": [{"index": 0, "P1": ["p1","p2"], "P2": ["p1","p2"], "P3": ["p1","p2"]}, ...]}"""

ALTERING_PROMPT = """You are a motion description editing expert. For each input motion description, create EXACTLY ONE minimal edit that changes the motion semantics.

Choose ONE edit type:
- direction: swap left↔right, forward↔backward, clockwise↔counterclockwise, ahead↔back
- bodypart: swap left hand↔right hand, left arm↔right arm, left leg↔right leg, left foot↔right foot

Rules:
- Change ONLY the target attribute — everything else stays identical
- Make the MINIMUM possible edit (change as few words as possible)
- The unedited parts must remain EXACTLY the same
- If the caption has no direction or bodypart to swap, write "SKIPPED" for the edited field
- Pick the edit type that is most applicable to the caption

Output JSON:
{"results": [{"index": 0, "edit_type": "direction" or "bodypart", "edited": "the modified caption or SKIPPED"}], ...]}"""

FILTER_SYSTEM_PROMPT = """Motion description semantic judge.

You will receive pairs of motion descriptions. For each pair, determine:

For PRESERVING pairs: Does PAIR describe the EXACT same motion as ORIG?
- Check: action, direction, speed, body parts, count, order must ALL match
- "in reverse" = backward, "ahead" = forward — acceptable
- Minor wording differences are OK as long as the described motion is identical

For ALTERING pairs: Does PAIR change EXACTLY ONE attribute (direction or bodypart) while keeping everything else identical?
- The change must be minimal and clean
- Only the target attribute should differ

Output JSON:
{"results": [true/false, ...]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def collect_captions():
    """Collect all unique captions from train split."""
    with open(TRAIN_FILE) as f:
        ids = [l.strip() for l in f.readlines()]

    captions = OrderedDict()
    for mid in ids:
        path = pjoin(TEXT_DIR, mid + '.txt')
        if not os.path.exists(path):
            continue
        with cs.open(path) as f:
            for line in f:
                parts = line.strip().split('#')
                cap = parts[0].strip()
                if cap and len(cap) > 15 and cap not in captions:
                    captions[cap] = mid
    return captions


def call_llm(system_prompt, user_content, max_retries=3):
    for attempt in range(max_retries):
        try:
            client = get_client()
            kwargs = dict(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=0.7,
                max_tokens=4000,
            )
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except:
                pass
            resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content
            # Extract JSON from markdown code blocks if present
            if '```' in text:
                import re
                m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
                if m:
                    text = m.group(1)
            return json.loads(text)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"    LLM FAILED: {e}")
                return None
            time.sleep(5 * (attempt + 1))


def basic_filter(orig, para):
    if not para or not para.strip():
        return False
    o = orig.lower().rstrip('.')
    p = para.lower().rstrip('.')
    if o == p:
        return False
    ol, pl = len(orig.split()), len(para.split())
    return ol * 0.3 <= pl <= ol * 3.0


def generate_preserving_batch(batch_captions, batch_idx):
    items = "\n".join(f'{i+1}. "{cap}"' for i, (cap, _) in enumerate(batch_captions))
    content = f"Paraphrase these {len(batch_captions)} motion descriptions at 3 levels:\n\n{items}"
    result = call_llm(PRESERVING_PROMPT, content)
    return batch_idx, result


def generate_altering_batch(batch_captions, batch_idx):
    items = "\n".join(f'{i+1}. "{cap}"' for i, (cap, _) in enumerate(batch_captions))
    content = f"Edit these {len(batch_captions)} motion descriptions:\n\n{items}"
    result = call_llm(ALTERING_PROMPT, content)
    return batch_idx, result


def load_done(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    p = json.loads(line.strip())
                    done.add(p.get('text_orig', ''))
                except:
                    pass
    return done


def run_generate(args):
    all_caps = collect_captions()
    all_list = list(all_caps.items())
    random.shuffle(all_list)
    print(f"Total unique captions: {len(all_list)}")

    # Split captions: preserving uses all, altering reuses subset with direction/bodypart
    n_preserve = len(all_list) if N_PRESERVE_CAPS is None else min(N_PRESERVE_CAPS, len(all_list))
    preserve_caps = all_list[:n_preserve]

    # For altering, find captions that contain direction/bodypart keywords
    direction_kw = {'left', 'right', 'forward', 'backward', 'clockwise', 'counterclockwise'}
    bodypart_kw = {'left hand', 'right hand', 'left arm', 'right arm',
                   'left leg', 'right leg', 'left foot', 'right foot',
                   'left knee', 'right knee', 'left elbow', 'right elbow'}
    alter_candidates = []
    for cap, mid in all_list:
        cap_lower = cap.lower()
        has_dir = any(kw in cap_lower.split() for kw in direction_kw)
        has_bp = any(kw in cap_lower for kw in bodypart_kw)
        if has_dir or has_bp:
            alter_candidates.append((cap, mid))

    n_alter_target = len(alter_candidates) if N_ALTER_CAPS is None else min(N_ALTER_CAPS, len(alter_candidates))
    alter_caps = alter_candidates[:n_alter_target]
    print(f"Preserving captions: {len(preserve_caps)}, Altering captions: {len(alter_caps)}")

    # ======== Load already-done captions (resume support) ========
    done_caps = set()
    if os.path.exists(OUTPUT_RAW):
        with open(OUTPUT_RAW, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    p = json.loads(line.strip())
                    key = p.get('text_orig', '') + '|' + p.get('relation', '')
                    done_caps.add(key)
                except:
                    pass
        print(f"Already done: {len(done_caps)} caption-relation combos, resuming...")

    # Filter out already-done captions
    preserve_caps = [(c, m) for c, m in preserve_caps
                     if c + '|preserve' not in done_caps]
    alter_caps = [(c, m) for c, m in alter_caps
                  if c + '|alter' not in done_caps]
    print(f"After resume: preserve={len(preserve_caps)}, alter={len(alter_caps)}")

    # ======== Open output file in APPEND mode ========
    out_f = open(OUTPUT_RAW, 'a', encoding='utf-8')

    # Get current max pair_id
    pair_id = 0
    if os.path.exists(OUTPUT_RAW):
        with open(OUTPUT_RAW, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    p = json.loads(line.strip())
                    pair_id = max(pair_id, p.get('pair_id', 0))
                except:
                    pass

    total_written = 0

    # ======== Preserving (P1/P2/P3) ========
    print(f"\n{'='*60}")
    print(f"  Generating PRESERVING pairs (P1/P2/P3)")
    print(f"  Target: {len(preserve_caps)} captions × 3 levels × 2 paraphrases")
    print(f"{'='*60}")

    batches = [(i // BATCH_SIZE + 1, preserve_caps[i:i + BATCH_SIZE])
               for i in range(0, len(preserve_caps), BATCH_SIZE)]
    print(f"  Batches: {len(batches)}, Workers: {MAX_WORKERS}")

    start_time = time.time()
    n_preserve = 0
    level_counts = {'P1': 0, 'P2': 0, 'P3': 0}

    if batches:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_preserving_batch, caps, idx): idx
                       for idx, caps in batches}

            for future in as_completed(futures):
                batch_idx, result = future.result()
                if result is None:
                    continue
                batch = batches[batch_idx - 1][1]
                batch_written = 0

                results_list = result.get('results', []) if isinstance(result, dict) else result if isinstance(result, list) else []
                for i, (cap, mid) in enumerate(batch):
                    if i >= len(results_list):
                        continue
                    r = results_list[i]
                    if not isinstance(r, dict):
                        continue
                    for level in ['P1', 'P2', 'P3']:
                        for para in r.get(level, []):
                            if basic_filter(cap, para):
                                pair_id += 1
                                n_preserve += 1
                                level_counts[level] += 1
                                total_written += 1
                                batch_written += 1
                                out_f.write(json.dumps({
                                    'pair_id': pair_id,
                                    'motion_id': mid,
                                    'text_orig': cap,
                                    'text_pair': para.strip(),
                                    'relation': 'preserve',
                                    'edit_type': '',
                                    'level': level,
                                }, ensure_ascii=False) + '\n')

                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}/{len(batches)}: "
                      f"P1={level_counts['P1']} P2={level_counts['P2']} P3={level_counts['P3']} "
                      f"total={n_preserve} (+{batch_written}) ({elapsed:.0f}s)")
                out_f.flush()

    print(f"\n  Preserving total: {n_preserve}")
    print(f"    P1: {level_counts['P1']}, P2: {level_counts['P2']}, P3: {level_counts['P3']}")

    # ======== Altering ========
    print(f"\n{'='*60}")
    print(f"  Generating ALTERING pairs")
    print(f"  Target: {len(alter_caps)} captions")
    print(f"{'='*60}")

    batches = [(i // BATCH_SIZE + 1, alter_caps[i:i + BATCH_SIZE])
               for i in range(0, len(alter_caps), BATCH_SIZE)]

    start_time = time.time()
    n_alter = 0
    alter_counts = {}

    if batches:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(generate_altering_batch, caps, idx): idx
                       for idx, caps in batches}

            for future in as_completed(futures):
                batch_idx, result = future.result()
                if result is None:
                    continue
                batch = batches[batch_idx - 1][1]
                batch_written = 0

                results_list = result.get('results', []) if isinstance(result, dict) else result if isinstance(result, list) else []
                for i, (cap, mid) in enumerate(batch):
                    if i >= len(results_list):
                        continue
                    r = results_list[i]
                    if not isinstance(r, dict):
                        continue
                    edited = r.get('edited', '').strip()
                    edit_type = r.get('edit_type', 'direction')

                    if edited == 'SKIPPED' or not edited:
                        continue

                    if basic_filter(cap, edited):
                        pair_id += 1
                        n_alter += 1
                        alter_counts[edit_type] = alter_counts.get(edit_type, 0) + 1
                        total_written += 1
                        batch_written += 1
                        out_f.write(json.dumps({
                            'pair_id': pair_id,
                            'motion_id': mid,
                            'text_orig': cap,
                            'text_pair': edited,
                            'relation': 'alter',
                            'edit_type': edit_type,
                            'level': '',
                        }, ensure_ascii=False) + '\n')

                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}/{len(batches)}: "
                      f"alter total={n_alter} (+{batch_written}) ({elapsed:.0f}s)")
                out_f.flush()

    out_f.close()

    print(f"\n  Altering total: {n_alter}")
    print(f"    Types: {alter_counts}")

    # ======== Summary ========
    # Count total in file
    total_in_file = 0
    with open(OUTPUT_RAW, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                total_in_file += 1

    print(f"\n{'='*60}")
    print(f"  This run: {total_written} new pairs")
    print(f"  Total in file: {total_in_file} pairs")
    print(f"  Output: {OUTPUT_RAW}")
    print(f"{'='*60}")


def run_filter(args):
    pairs = [json.loads(l) for l in open(OUTPUT_RAW, 'r', encoding='utf-8')]
    print(f"Total pairs to filter: {len(pairs)}")

    FILTER_BATCH = getattr(args, 'filter_batch_size', 50)
    FILTER_WORKERS = getattr(args, 'filter_workers', 16)
    MAX_RETRIES = getattr(args, 'filter_max_retries', 3)
    decision_path = OUTPUT_FILTERED + '.decisions.jsonl'

    # Persist both accepted and rejected decisions so interrupted runs resume exactly.
    decisions = {}
    if os.path.exists(decision_path):
        with open(decision_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    pid = row['pair_id']
                    passed = row['passed']
                    if isinstance(passed, bool):
                        decisions[pid] = passed
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    pass

    # Backward compatibility for an already-completed filtered file.
    if os.path.exists(OUTPUT_FILTERED):
        with open(OUTPUT_FILTERED, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    decisions.setdefault(row.get('pair_id', -1), True)
                except json.JSONDecodeError:
                    pass

    if decisions:
        print(f"Already judged {len(decisions)} pairs, resuming...")

    remaining = [p for p in pairs if p.get('pair_id', -1) not in decisions]
    print(f"Remaining to filter: {len(remaining)}")
    batches = [(i, remaining[i:i + FILTER_BATCH])
               for i in range(0, len(remaining), FILTER_BATCH)]
    total = len(batches)

    import threading
    progress_lock = threading.Lock()
    decision_lock = threading.Lock()
    progress = {'done': 0, 'passed': 0, 'errors': 0}

    def filter_batch(batch_data, decision_file):
        batch_idx, batch_pairs = batch_data
        items = "\n".join(
            f'{i+1}. ORIG: "{p["text_orig"]}"\n   PAIR: "{p["text_pair"]}"\n   TYPE: {p["relation"]}'
            for i, p in enumerate(batch_pairs)
        )

        for attempt in range(MAX_RETRIES):
            try:
                client = get_client()
                kwargs = dict(
                    model=FILTER_MODEL,
                    messages=[
                        {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                        {"role": "user", "content": f"Judge {len(batch_pairs)} pairs:\n\n{items}"}
                    ],
                    temperature=0,
                    max_tokens=3000,
                )
                try:
                    kwargs["response_format"] = {"type": "json_object"}
                except:
                    pass
                resp = client.chat.completions.create(**kwargs)
                if not resp.choices:
                    raise ValueError("Judge response has no choices")
                text = resp.choices[0].message.content
                if not text:
                    raise ValueError("Judge response content is empty")
                if '```' in text:
                    import re as _re
                    m = _re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, _re.DOTALL)
                    if m:
                        text = m.group(1)
                results = json.loads(text).get('results', [])
                if len(results) != len(batch_pairs):
                    raise ValueError(
                        f"Expected {len(batch_pairs)} decisions, got {len(results)}"
                    )
                if any(not isinstance(result, bool) for result in results):
                    raise ValueError("Judge results must contain only booleans")
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"  Batch {batch_idx+1} FAILED after {MAX_RETRIES} retries: {e}")
                    raise RuntimeError(
                        f"Judge batch {batch_idx+1} failed after {MAX_RETRIES} retries"
                    ) from e
                else:
                    time.sleep(2 * (attempt + 1))
                    continue

        batch_decisions = [
            (p.get('pair_id', -1), result)
            for p, result in zip(batch_pairs, results)
        ]
        passed = [p for p, result in zip(batch_pairs, results) if result]

        with decision_lock:
            for pid, result in batch_decisions:
                decisions[pid] = result
                decision_file.write(json.dumps({
                    'pair_id': pid,
                    'passed': result,
                }) + '\n')
            decision_file.flush()

        with progress_lock:
            progress['done'] += 1
            progress['passed'] += len(passed)
            if batch_idx % 100 == 0 or batch_idx == total - 1:
                elapsed = time.time() - start_time
                rate = progress['done'] / max(elapsed, 1)
                eta = (total - progress['done']) / max(rate, 0.001) / 60
                print(f"  Filter {batch_idx+1}/{total}: "
                      f"passed={progress['passed']}, errors={progress['errors']} "
                      f"({progress['done']/total*100:.1f}%) "
                      f"rate={rate:.1f}/s ETA={eta:.1f}min")

        return passed

    start_time = time.time()
    if remaining:
        print(f"Filtering {len(remaining)} pairs in {total} batches, {FILTER_WORKERS} workers...")
        with open(decision_path, 'a', encoding='utf-8') as decision_file:
            with ThreadPoolExecutor(max_workers=FILTER_WORKERS) as executor:
                futures = {
                    executor.submit(filter_batch, batch_data, decision_file): idx
                    for idx, batch_data in enumerate(batches)
                }
                for future in as_completed(futures):
                    future.result()
    else:
        print("Nothing left to judge; rebuilding filtered output from decisions.")

    missing = [p.get('pair_id', -1) for p in pairs
               if p.get('pair_id', -1) not in decisions]
    if missing:
        raise RuntimeError(f"Missing judge decisions for {len(missing)} pairs")

    merged = [p for p in pairs if decisions[p.get('pair_id', -1)]]
    output_tmp = OUTPUT_FILTERED + '.tmp'
    with open(output_tmp, 'w', encoding='utf-8') as f:
        for p in merged:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    os.replace(output_tmp, OUTPUT_FILTERED)

    # Stats
    print(f"\n{'='*60}")
    elapsed = time.time() - start_time
    print(f"  Filter done in {elapsed/60:.1f} min")
    print(f"  Total judged: {len(decisions)}")
    print(f"  Total in file: {len(merged)}")
    print(f"  Output: {OUTPUT_FILTERED}")

    n_preserve = sum(1 for p in merged if p['relation'] == 'preserve')
    n_alter = sum(1 for p in merged if p['relation'] == 'alter')
    print(f"  Preserve: {n_preserve}, Alter: {n_alter}")

    for lv in ['P1', 'P2', 'P3', '']:
        c = sum(1 for p in merged if p.get('level') == lv)
        if c > 0:
            label = lv if lv else 'alter'
            print(f"    {label}: {c}")


def main():
    global API_KEY, BASE_URL, MODEL, FILTER_MODEL
    global BATCH_SIZE, MAX_WORKERS, TEXT_DIR, TRAIN_FILE
    global OUTPUT_RAW, OUTPUT_FILTERED

    parser = argparse.ArgumentParser()
    parser.add_argument('--filter-only', action='store_true',
                        help='Only run filtering on existing raw pairs')
    parser.add_argument('--dataset-root', default=None,
                        help='Override the HumanML3D dataset root')
    parser.add_argument('--output-raw', default=None)
    parser.add_argument('--output-filtered', default=None)
    parser.add_argument('--api-key-env', default=None,
                        help='Read the endpoint API key from this environment variable')
    parser.add_argument('--base-url', default=None)
    parser.add_argument('--model', default=None)
    parser.add_argument('--filter-model', default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--max-workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    if args.dataset_root:
        TEXT_DIR = pjoin(args.dataset_root, 'texts')
        TRAIN_FILE = pjoin(args.dataset_root, 'train.txt')
    if args.output_raw:
        OUTPUT_RAW = args.output_raw
    if args.output_filtered:
        OUTPUT_FILTERED = args.output_filtered
    if args.api_key_env:
        API_KEY = os.environ.get(args.api_key_env, '')
        if not API_KEY:
            raise RuntimeError(
                f'Missing API key in environment variable {args.api_key_env}')
    if args.base_url:
        BASE_URL = args.base_url
    if args.model:
        MODEL = args.model
    if args.filter_model:
        FILTER_MODEL = args.filter_model
    elif args.model:
        FILTER_MODEL = args.model
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.max_workers is not None:
        MAX_WORKERS = args.max_workers
    if args.seed is not None:
        random.seed(args.seed)

    if args.filter_only:
        run_filter(args)
    else:
        run_generate(args)


if __name__ == '__main__':
    main()
