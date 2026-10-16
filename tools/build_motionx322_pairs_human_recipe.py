#!/usr/bin/env python
"""Run the HumanML3D full-pair builder against Motion-X-322."""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.HumanML3D import build_full_pairs as human_builder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("dataset/Motion-X-322"))
    parser.add_argument(
        "--output-raw",
        type=Path,
        default=Path("dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe.jsonl"),
    )
    parser.add_argument(
        "--output-filtered",
        type=Path,
        default=Path(
            "dataset/Motion-X-322/preserving_pairs_gemini31_human_recipe_filtered.jsonl"
        ),
    )
    parser.add_argument("--model", default="gemini-3.1-flash-lite")
    parser.add_argument("--base-url", default="https://di2lab.zeabur.app/v1")
    parser.add_argument("--api-key-env", default="DI2LAB_API_KEY")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--cooldown-buffer-seconds", type=int, default=15)
    parser.add_argument("--filter-batch-size", type=int, default=50)
    parser.add_argument("--filter-workers", type=int, default=16)
    parser.add_argument("--filter-max-retries", type=int, default=3)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--filter-only", action="store_true")
    return parser.parse_args()


def install_cooldown_aware_generation_call(args):
    def call_llm(system_prompt, user_content, max_retries=3):
        attempt = 0
        while attempt < max_retries:
            try:
                client = human_builder.get_client()
                response = client.chat.completions.create(
                    model=human_builder.MODEL,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.7,
                    max_tokens=4000,
                    response_format={"type": "json_object"},
                )
                text = response.choices[0].message.content
                if "```" in text:
                    match = re.search(
                        r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL
                    )
                    if match:
                        text = match.group(1)
                return json.loads(text)
            except Exception as exc:
                message = str(exc)
                cooldown = re.search(r"['\"]reset_seconds['\"]:\s*(\d+)", message)
                if "model_cooldown" in message and cooldown:
                    wait_seconds = int(cooldown.group(1)) + args.cooldown_buffer_seconds
                    print(
                        f"    MODEL COOLDOWN: waiting {wait_seconds}s before retry",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue
                attempt += 1
                if attempt >= max_retries:
                    print(f"    LLM FAILED: {exc}", flush=True)
                    return None
                time.sleep(5 * attempt)

    human_builder.call_llm = call_llm


def install_cooldown_aware_client(args):
    openai_client = human_builder.OpenAI

    class CompletionProxy:
        def __init__(self, completions):
            self._completions = completions

        def create(self, **kwargs):
            if args.disable_thinking:
                kwargs.setdefault("reasoning_effort", "none")
            while True:
                try:
                    return self._completions.create(**kwargs)
                except Exception as exc:
                    message = str(exc)
                    cooldown = re.search(
                        r"['\"]reset_seconds['\"]:\s*(\d+)", message
                    )
                    if "model_cooldown" not in message or not cooldown:
                        raise
                    wait_seconds = (
                        int(cooldown.group(1)) + args.cooldown_buffer_seconds
                    )
                    print(
                        f"    MODEL COOLDOWN: waiting {wait_seconds}s before retry",
                        flush=True,
                    )
                    time.sleep(wait_seconds)

    class ChatProxy:
        def __init__(self, chat):
            self.completions = CompletionProxy(chat.completions)

    class ClientProxy:
        def __init__(self, client):
            self.chat = ChatProxy(client.chat)

    def get_client():
        return ClientProxy(
            openai_client(
                api_key=human_builder.API_KEY,
                base_url=human_builder.BASE_URL,
            )
        )

    human_builder.get_client = get_client


def configure(args):
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable {args.api_key_env}")

    dataset_root = args.dataset_root.resolve()
    text_dir = dataset_root / "texts"
    train_file = dataset_root / "train.txt"
    if not text_dir.is_dir():
        raise FileNotFoundError(text_dir)
    if not train_file.is_file():
        raise FileNotFoundError(train_file)

    args.output_raw.parent.mkdir(parents=True, exist_ok=True)
    args.output_filtered.parent.mkdir(parents=True, exist_ok=True)

    human_builder.API_KEY = api_key
    human_builder.BASE_URL = args.base_url
    human_builder.MODEL = args.model
    human_builder.FILTER_MODEL = args.model
    human_builder.BATCH_SIZE = args.batch_size
    human_builder.MAX_WORKERS = args.workers
    human_builder.TEXT_DIR = str(text_dir)
    human_builder.TRAIN_FILE = str(train_file)
    human_builder.OUTPUT_RAW = str(args.output_raw.resolve())
    human_builder.OUTPUT_FILTERED = str(args.output_filtered.resolve())
    human_builder.N_PRESERVE_CAPS = args.limit
    human_builder.N_ALTER_CAPS = args.limit
    install_cooldown_aware_client(args)
    install_cooldown_aware_generation_call(args)
    random.seed(args.seed)


def write_metadata(args):
    output = args.output_filtered if args.filter_only else args.output_raw
    if not output.is_file():
        return
    counts = {"preserve": 0, "alter": 0, "P1": 0, "P2": 0, "P3": 0}
    motions = set()
    pairs = 0
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            pairs += 1
            motions.add(row.get("motion_id"))
            relation = row.get("relation")
            level = row.get("level")
            if relation in counts:
                counts[relation] += 1
            if level in counts:
                counts[level] += 1
    metadata = {
        "dataset_root": str(args.dataset_root.resolve()),
        "split": "train",
        "generator": "dataset/HumanML3D/build_full_pairs.py",
        "model": args.model,
        "base_url": args.base_url,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "filtered": args.filter_only,
        "pairs": pairs,
        "unique_motions": len(motions),
        "counts": counts,
    }
    output.with_suffix(output.suffix + ".meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main():
    args = parse_args()
    configure(args)
    if args.filter_only:
        human_builder.run_filter(args)
    else:
        human_builder.run_generate(args)
    write_metadata(args)


if __name__ == "__main__":
    main()
