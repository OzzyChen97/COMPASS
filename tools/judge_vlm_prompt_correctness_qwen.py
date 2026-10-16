import argparse
import base64
import csv
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import openai
from PIL import Image
from PIL import ImageDraw


SYSTEM_PROMPT = (
    "You are an expert judge for text-to-motion stick-figure animations. "
    "The image is a contact sheet of sampled animation frames ordered left to right and top to bottom. "
    "Judge only whether the visible motion matches the prompt. Ignore rendering quality. "
    "Return only valid compact JSON."
)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def data_url(path):
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def sample_indices(num_frames, num_samples):
    if num_frames <= num_samples:
        return list(range(num_frames))
    return sorted(set(np.linspace(0, num_frames - 1, num_samples, dtype=int).tolist()))


def make_contact_sheet(gif_path, out_path, num_samples=12, cols=6, thumb_size=192):
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    frames = imageio.mimread(gif_path)
    if not frames:
        raise RuntimeError(f"empty gif: {gif_path}")
    indices = sample_indices(len(frames), num_samples)
    thumbs = []
    for idx in indices:
        img = Image.fromarray(np.asarray(frames[idx])[..., :3]).convert("RGB")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        x = (thumb_size - img.width) // 2
        y = (thumb_size - img.height) // 2
        canvas.paste(img, (x, y))
        thumbs.append(canvas)
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), "white")
    draw = ImageDraw.Draw(sheet)
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * thumb_size
        y = (i // cols) * thumb_size
        sheet.paste(thumb, (x, y))
        draw.rectangle([x, y, x + thumb_size - 1, y + thumb_size - 1], outline=(220, 220, 220), width=1)
        draw.text((x + 5, y + 5), str(i + 1), fill=(90, 90, 90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path


def build_user_prompt(prompt):
    schema = {
        "matches_prompt": "boolean",
        "confidence": "number from 0 to 1",
        "rationale": "short string, at most 10 words",
    }
    return (
        "Does this animation match the prompt?\n\n"
        f"Prompt: {prompt}\n\n"
        "Return exactly one compact JSON object with this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False)}\n"
        "Do not include analysis. Keep rationale under 10 words."
    )


def create_client(base_url, api_key, timeout):
    return openai.OpenAI(
        api_key=api_key or "EMPTY",
        base_url=base_url.rstrip("/") + "/",
        timeout=timeout,
        max_retries=2,
    )


def post_chat(client, model, messages, max_tokens):
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    try:
        return client.chat.completions.create(**kwargs)
    except openai.BadRequestError:
        kwargs.pop("response_format", None)
        return client.chat.completions.create(**kwargs)


def extract_json(text):
    text = text.strip()
    if text.lower() in {"true", "yes", "correct"}:
        return {"matches_prompt": True, "confidence": 0.5, "rationale": text}
    if text.lower() in {"false", "no", "incorrect"}:
        return {"matches_prompt": False, "confidence": 0.5, "rationale": text}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    parsed = json.loads(text)
    if isinstance(parsed, bool):
        return {"matches_prompt": parsed, "confidence": 0.5, "rationale": str(parsed)}
    if isinstance(parsed, str):
        lowered = parsed.lower()
        if lowered in {"true", "yes", "correct"}:
            return {"matches_prompt": True, "confidence": 0.5, "rationale": parsed}
        if lowered in {"false", "no", "incorrect"}:
            return {"matches_prompt": False, "confidence": 0.5, "rationale": parsed}
    if not isinstance(parsed, dict):
        raise ValueError(f"unexpected response JSON type: {type(parsed).__name__}")
    return parsed


def normalize_judgement(row):
    if "matches_prompt" not in row:
        for key in ("match", "matches", "correct", "is_correct", "prompt_matches"):
            if key in row:
                row["matches_prompt"] = row[key]
                break
    if "rationale" not in row and "rationally" in row:
        row["rationale"] = row.get("rationally")
    if "rationale" not in row and "reason" in row:
        row["rationale"] = row.get("reason")
    return row


def judge_row(row, args, client):
    sheet_subdir = args.sheet_dir / row["model"] / row["edit_type"]
    sheet = make_contact_sheet(
        row["gif"],
        sheet_subdir / f"{row['pair_uid']}_{row['side']}.png",
        num_samples=args.num_samples,
    )
    out = {
        "model": row["model"],
        "edit_type": row["edit_type"],
        "pair_uid": row["pair_uid"],
        "side": row["side"],
        "motion_id": row.get("motion_id"),
        "prompt": row.get("prompt", ""),
        "gif": row["gif"],
        "sheet": str(sheet),
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(row.get("prompt", ""))},
                {"type": "image_url", "image_url": {"url": data_url(sheet)}},
            ],
        },
    ]
    try:
        response = post_chat(
            client,
            args.model,
            messages,
            max_tokens=args.max_tokens,
        )
        content = response.choices[0].message.content or ""
        parsed = normalize_judgement(extract_json(content))
        out.update(parsed)
        out["_raw_response"] = content
    except Exception as exc:  # noqa: BLE001
        out["error"] = repr(exc)
    return out


def bool_score(row, key):
    return 1 if row.get(key) is True else 0


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else ""


def write_summary(results_path, summary_path):
    rows = [r for r in read_jsonl(results_path) if not r.get("error")]
    groups = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["edit_type"])].append(row)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "edit_type",
                "n",
                "accuracy",
                "orig_n",
                "orig_accuracy",
                "edit_n",
                "edit_accuracy",
                "mean_confidence",
            ],
        )
        writer.writeheader()
        for (model, edit_type), items in sorted(groups.items()):
            orig = [x for x in items if x.get("side") == "orig"]
            edit = [x for x in items if x.get("side") == "edit"]
            writer.writerow(
                {
                    "model": model,
                    "edit_type": edit_type,
                    "n": len(items),
                    "accuracy": mean(bool_score(x, "matches_prompt") for x in items),
                    "orig_n": len(orig),
                    "orig_accuracy": mean(bool_score(x, "matches_prompt") for x in orig),
                    "edit_n": len(edit),
                    "edit_accuracy": mean(bool_score(x, "matches_prompt") for x in edit),
                    "mean_confidence": mean(float(x.get("confidence", 0) or 0) for x in items),
                }
            )
    print(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-manifest", type=Path, default=Path("ablation/vlm_judge/visuals_20_each/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("ablation/vlm_judge/qwen35a3b_prompt_correctness.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("ablation/vlm_judge/qwen35a3b_prompt_correctness_summary.csv"))
    parser.add_argument("--sheet-dir", type=Path, default=Path("ablation/vlm_judge/contact_sheets_prompt_correctness"))
    parser.add_argument("--base-url", default=os.environ.get("QWEN_VLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--model", default=os.environ.get("QWEN_VLM_MODEL", "Qwen3.5-35B-A3B"))
    parser.add_argument("--api-key-env", default="QWEN_VLM_API_KEY")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    client = create_client(args.base_url, api_key, args.timeout)
    done = set()
    if args.resume and args.out.exists():
        for row in read_jsonl(args.out):
            if not row.get("error"):
                done.add((row.get("model"), row.get("edit_type"), row.get("pair_uid"), row.get("side")))

    rows = list(read_jsonl(args.visual_manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = [r for r in rows if (r.get("model"), r.get("edit_type"), r.get("pair_uid"), r.get("side")) not in done]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        if args.workers <= 1:
            for idx, row in enumerate(rows, 1):
                out = judge_row(row, args, client)
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{idx}/{len(rows)} {out['model']} {out['edit_type']} {out['pair_uid']} {out['side']}", flush=True)
                time.sleep(0.1)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(judge_row, row, args, client) for row in rows]
                for idx, future in enumerate(as_completed(futures), 1):
                    out = future.result()
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"{idx}/{len(rows)} {out['model']} {out['edit_type']} {out['pair_uid']} {out['side']}", flush=True)

    write_summary(args.out, args.summary)


if __name__ == "__main__":
    main()
