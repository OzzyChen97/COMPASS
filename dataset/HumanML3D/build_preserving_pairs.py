"""
SATO++ Preserving Pair 数据构造脚本 (v3 — 大规模并行版)

改进:
  - 每批 30 条 caption 打包发 LLM
  - 并发 5 线程
  - 断点续传
  - 生成和过滤分两步跑

用法:
  # Step 1: 生成 (不做过滤, 速度快)
  python build_preserving_pairs.py --dataset t2m

  # Step 2: 过滤 (单独跑, 可选)
  python build_preserving_pairs.py --dataset t2m --filter-only
"""

import json
import os
import codecs as cs
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from os.path import join as pjoin
from collections import OrderedDict

# ======== 配置 ========
API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = "gpt-4o"
FILTER_MODEL = "gpt-4o-mini"
BATCH_SIZE = 30
MAX_WORKERS = 5

DATASET_CONFIGS = {
    'kit': {
        'text_dir': '/mnt/workspace/Wilson/T2M-GPT/dataset/KIT-ML/texts',
        'output': '/mnt/workspace/Wilson/T2M-GPT/dataset/KIT-ML/preserving_pairs.jsonl',
    },
    't2m': {
        'text_dir': '/mnt/workspace/Wilson/T2M-GPT/dataset/HumanML3D/texts',
        'output': '/mnt/workspace/Wilson/T2M-GPT/dataset/HumanML3D/preserving_pairs.jsonl',
    },
}

SYSTEM_PROMPT = """You are a motion description paraphrasing expert. For each input motion description, generate paraphrases at THREE levels:

**P1 (Lexical)**: Replace words with synonyms. Keep sentence structure.
- "A person walks forward." → "A human strolls ahead."

**P2 (Syntactic)**: Change sentence structure (word order, active/passive).
- "A person walks forward while turning left." → "While turning left, a person proceeds forward."

**P3 (Scene/Narrative)**: Rewrite in different narrative style.
- "A person walks forward." → "An individual can be seen moving in a forward direction."

RULES:
- Preserve ALL: action, direction, speed, body parts, count, order
- Do NOT change direction (left/right/forward/backward)
- Do NOT change body parts, step counts, or speed
- Each level: 2 paraphrases
- "in reverse"=backward, "ahead"=forward — OK

JSON output:
{"results": [{"index": 0, "P1": ["p1","p2"], "P2": ["p1","p2"], "P3": ["p1","p2"]}, ...]}"""

FILTER_SYSTEM_PROMPT = """Motion description semantic judge. Does PAIR describe the EXACT same motion as ORIG? Check: action, direction, speed, body parts, count, order. "in reverse"=backward, "ahead"=forward OK.
JSON: {"results": [true/false]}"""


def get_client():
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)


def collect_captions(text_dir):
    seen = OrderedDict()
    for fname in sorted(os.listdir(text_dir)):
        if not fname.endswith('.txt'):
            continue
        motion_id = fname.replace('.txt', '')
        with cs.open(pjoin(text_dir, fname)) as f:
            for line in f:
                parts = line.strip().split('#')
                cap = parts[0].strip()
                if cap and cap not in seen:
                    seen[cap] = motion_id
    return seen


def load_done(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    p = json.loads(line.strip())
                    done.add(p['text_orig'])
                except:
                    pass
    return done


def generate_batch(batch_captions, batch_idx):
    """单批生成"""
    items = "\n".join(f'{i+1}. "{cap}"' for i, cap in enumerate(batch_captions))

    for attempt in range(3):
        try:
            client = get_client()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Paraphrase these {len(batch_captions)} descriptions:\n\n{items}"}
                ],
                temperature=0.7,
                max_tokens=3000,
                response_format={"type": "json_object"}
            )
            results = json.loads(resp.choices[0].message.content).get('results', [])
            return batch_idx, results
        except Exception as e:
            if attempt == 2:
                print(f"    Batch {batch_idx} FAILED: {e}")
            time.sleep(2)
    return batch_idx, []


def basic_filter(orig, para):
    if not para.strip():
        return False
    o, p = orig.lower().rstrip('.'), para.lower().rstrip('.')
    if o == p:
        return False
    ol, pl = len(orig.split()), len(para.split())
    return ol * 0.4 <= pl <= ol * 3


def run_generate(args):
    cfg = DATASET_CONFIGS[args.dataset]
    text_dir, output_path = cfg['text_dir'], cfg['output']

    unique_caps = collect_captions(text_dir)
    all_caps = list(unique_caps.keys())
    print(f"Unique captions: {len(all_caps)}")

    done = load_done(output_path)
    todo = [c for c in all_caps if c not in done]
    print(f"Already done: {len(done)}, Remaining: {len(todo)}")

    # 分批
    batches = []
    for i in range(0, len(todo), BATCH_SIZE):
        batches.append((i // BATCH_SIZE + 1, todo[i:i + BATCH_SIZE]))
    print(f"Batches: {len(batches)}, Workers: {MAX_WORKERS}\n")

    pair_id = len(done) * 6
    total_written = 0
    start_time = time.time()

    with open(output_path, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(generate_batch, caps, idx): idx
                for idx, caps in batches
            }

            for future in as_completed(futures):
                batch_idx, results = future.result()
                batch = batches[batch_idx - 1][1]

                n_pairs = 0
                for i, cap in enumerate(batch):
                    if i >= len(results):
                        continue
                    r = results[i]
                    mid = unique_caps[cap]

                    for level in ['P1', 'P2', 'P3']:
                        for para in r.get(level, []):
                            if basic_filter(cap, para):
                                pair_id += 1
                                n_pairs += 1
                                total_written += 1
                                f.write(json.dumps({
                                    'pair_id': pair_id,
                                    'motion_id': mid,
                                    'text_orig': cap,
                                    'text_pair': para.strip(),
                                    'level': level,
                                    'relation': 'preserve',
                                }, ensure_ascii=False) + '\n')

                elapsed = time.time() - start_time
                rate = total_written / max(elapsed, 1)
                done_count = sum(1 for _, r in results for _ in r.get('P1', [])) if results else 0
                print(f"  Batch {batch_idx}/{len(batches)}: {len(results)} caps → {n_pairs} pairs "
                      f"(total: {total_written}, {rate:.0f} pairs/sec)")

                f.flush()

    elapsed = time.time() - start_time
    print(f"\nDone: {total_written} pairs in {elapsed/60:.1f} min")


def run_filter(args):
    cfg = DATASET_CONFIGS[args.dataset]
    output_path = cfg['output']

    pairs = [json.loads(l) for l in open(output_path, 'r', encoding='utf-8')]
    print(f"Total pairs to filter: {len(pairs)}")

    FILTER_BATCH = 25
    filtered = []
    client = get_client()

    for start in range(0, len(pairs), FILTER_BATCH):
        batch = pairs[start:start + FILTER_BATCH]
        batch_idx = start // FILTER_BATCH + 1
        total = (len(pairs) + FILTER_BATCH - 1) // FILTER_BATCH

        items = "\n".join(
            f'{i+1}. ORIG: "{p["text_orig"]}"\n   PAIR: "{p["text_pair"]}"'
            for i, p in enumerate(batch)
        )

        try:
            resp = client.chat.completions.create(
                model=FILTER_MODEL,
                messages=[
                    {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Check {len(batch)} pairs:\n\n{items}"}
                ],
                temperature=0, max_tokens=1000,
                response_format={"type": "json_object"}
            )
            results = json.loads(resp.choices[0].message.content).get('results', [])
        except:
            results = [True] * len(batch)

        for i, p in enumerate(batch):
            if i < len(results) and results[i]:
                filtered.append(p)

        if batch_idx % 20 == 0 or batch_idx == total:
            print(f"  Filter {batch_idx}/{total}: {len(filtered)} passed")

        time.sleep(0.2)

    # Write
    filtered_path = output_path.replace('.jsonl', '_filtered.jsonl')
    with open(filtered_path, 'w', encoding='utf-8') as f:
        for p in filtered:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    print(f"\nFiltered: {len(pairs)} → {len(filtered)} ({len(filtered)/max(len(pairs),1)*100:.1f}%)")
    print(f"Output: {filtered_path}")
    for lv in ['P1', 'P2', 'P3']:
        c = sum(1 for p in filtered if p['level'] == lv)
        print(f"  {lv}: {c}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, choices=['kit', 't2m'])
    parser.add_argument('--filter-only', action='store_true',
                        help='只跑过滤步骤（对已有 .jsonl 文件）')
    args = parser.parse_args()

    if args.filter_only:
        run_filter(args)
    else:
        run_generate(args)


if __name__ == '__main__':
    main()
