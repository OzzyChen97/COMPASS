import argparse
import base64
import csv
import json
import os
import re
import time
from collections import Counter
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw


SYSTEM_PROMPT = (
    "You are an expert judge for text-to-motion stick-figure animations. "
    "The images are sampled frames from animations, ordered left to right and top to bottom. "
    "Judge the visible motion, not rendering quality. Return only valid JSON."
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


def group_cases(rows):
    grouped = defaultdict(dict)
    meta = {}
    for row in rows:
        key = (row["model"], row["edit_type"], row["pair_uid"])
        grouped[key][row["side"]] = row
        meta[key] = row
    for key in sorted(grouped):
        sides = grouped[key]
        if "orig" not in sides or "edit" not in sides:
            continue
        base = dict(meta[key])
        base.pop("side", None)
        base.pop("prompt", None)
        yield key, base, sides["orig"], sides["edit"]


def expected_keys(edit_type):
    if edit_type == "preserve":
        return {
            "orig_matches_prompt": "boolean",
            "edit_matches_prompt": "boolean",
            "same_motion": "boolean",
            "non_edited_attributes_preserved": "boolean",
            "confidence": "number from 0 to 1",
            "rationale": "short string, at most 12 words",
        }
    return {
        "orig_matches_prompt": "boolean",
        "edit_matches_prompt": "boolean",
        "intended_change_visible": "boolean",
        "non_edited_attributes_preserved": "boolean",
        "confidence": "number from 0 to 1",
        "rationale": "short string, at most 12 words",
    }


def build_user_prompt(edit_type, orig_prompt, edit_prompt):
    schema = json.dumps(expected_keys(edit_type), ensure_ascii=False)
    if edit_type == "preserve":
        task = (
            "The two prompts should describe the same motion. "
            "Decide whether both animations match their own prompts and whether the edited/paraphrased animation keeps the same motion as the original."
        )
    elif edit_type == "direction":
        task = (
            "The edited prompt changes the intended direction or facing/path direction. "
            "Decide whether the edited animation visibly follows the edited direction while preserving unrelated action attributes."
        )
    elif edit_type == "speed":
        task = (
            "The edited prompt changes motion speed. "
            "Decide whether the edited animation visibly changes speed in the requested way while preserving unrelated action attributes."
        )
    elif edit_type == "bodypart":
        task = (
            "The edited prompt changes the involved body part or side. "
            "Decide whether the edited animation visibly uses the requested body part/side while preserving unrelated action attributes."
        )
    else:
        task = "Judge whether the edited animation satisfies the edited prompt."
    return (
        f"{task}\n\n"
        f"Edit type: {edit_type}\n"
        f"Original prompt: {orig_prompt}\n"
        f"Edited prompt: {edit_prompt}\n\n"
        "Image 1 is the original animation contact sheet. Image 2 is the edited animation contact sheet.\n"
        f"Return only one compact JSON object with this schema: {schema}\n"
        "Do not include analysis. Keep rationale under 12 words."
    )


def post_chat(base_url, model, api_key, messages, timeout, max_tokens):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)


def normalize_judgement(row):
    if "rationale" not in row and "rationally" in row:
        row["rationale"] = row.get("rationally")
    if "rationale" not in row and "reason" in row:
        row["rationale"] = row.get("reason")
    return row


def judge_case(base_url, model, api_key, edit_type, orig_prompt, edit_prompt, orig_sheet, edit_sheet, timeout, max_tokens):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_user_prompt(edit_type, orig_prompt, edit_prompt)},
                {"type": "image_url", "image_url": {"url": data_url(orig_sheet)}},
                {"type": "image_url", "image_url": {"url": data_url(edit_sheet)}},
            ],
        },
    ]
    response = post_chat(base_url, model, api_key, messages, timeout=timeout, max_tokens=max_tokens)
    content = response["choices"][0]["message"]["content"]
    parsed = extract_json(content)
    parsed = normalize_judgement(parsed)
    parsed["_raw_response"] = content
    return parsed


def bool_score(row, key):
    value = row.get(key)
    return 1 if value is True else 0


def write_summary(results_path, summary_path):
    rows = list(read_jsonl(results_path))
    groups = defaultdict(list)
    for row in rows:
        if row.get("error"):
            continue
        groups[(row["model"], row["edit_type"])].append(row)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "edit_type",
                "n",
                "orig_match_acc",
                "edit_match_acc",
                "same_motion_acc",
                "intended_change_acc",
                "non_edited_preserved_acc",
                "mean_confidence",
            ],
        )
        writer.writeheader()
        for (model, edit_type), items in sorted(groups.items()):
            n = len(items)
            writer.writerow(
                {
                    "model": model,
                    "edit_type": edit_type,
                    "n": n,
                    "orig_match_acc": sum(bool_score(x, "orig_matches_prompt") for x in items) / n,
                    "edit_match_acc": sum(bool_score(x, "edit_matches_prompt") for x in items) / n,
                    "same_motion_acc": (
                        sum(bool_score(x, "same_motion") for x in items) / n if edit_type == "preserve" else ""
                    ),
                    "intended_change_acc": (
                        sum(bool_score(x, "intended_change_visible") for x in items) / n
                        if edit_type != "preserve"
                        else ""
                    ),
                    "non_edited_preserved_acc": sum(
                        bool_score(x, "non_edited_attributes_preserved") for x in items
                    )
                    / n,
                    "mean_confidence": sum(float(x.get("confidence", 0) or 0) for x in items) / n,
                }
            )
    print(summary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-manifest", type=Path, default=Path("ablation/vlm_judge/visuals_20_each/manifest.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("ablation/vlm_judge/qwen35vl_results.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("ablation/vlm_judge/qwen35vl_summary.csv"))
    parser.add_argument("--sheet-dir", type=Path, default=Path("ablation/vlm_judge/contact_sheets"))
    parser.add_argument("--base-url", default=os.environ.get("QWEN_VLM_BASE_URL", "http://127.0.0.1:8001/v1"))
    parser.add_argument("--model", default=os.environ.get("QWEN_VLM_MODEL", "qwen3.5-vl"))
    parser.add_argument("--api-key-env", default="QWEN_VLM_API_KEY")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    done = set()
    if args.resume and args.out.exists():
        for row in read_jsonl(args.out):
            done.add((row.get("model"), row.get("edit_type"), row.get("pair_uid")))

    rows = list(read_jsonl(args.visual_manifest))
    cases = list(group_cases(rows))
    if args.limit is not None:
        cases = cases[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    mode = "a" if args.resume else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for idx, (key, base, orig, edit) in enumerate(cases, 1):
            model_name, edit_type, pair_uid = key
            if key in done:
                continue
            sheet_subdir = args.sheet_dir / model_name / edit_type
            orig_sheet = make_contact_sheet(
                orig["gif"],
                sheet_subdir / f"{pair_uid}_orig.png",
                num_samples=args.num_samples,
            )
            edit_sheet = make_contact_sheet(
                edit["gif"],
                sheet_subdir / f"{pair_uid}_edit.png",
                num_samples=args.num_samples,
            )
            out = {
                "model": model_name,
                "edit_type": edit_type,
                "pair_uid": pair_uid,
                "motion_id": base.get("motion_id"),
                "text_orig": base.get("text_orig"),
                "text_pair": base.get("text_pair"),
                "orig_gif": orig["gif"],
                "edit_gif": edit["gif"],
                "orig_sheet": str(orig_sheet),
                "edit_sheet": str(edit_sheet),
            }
            try:
                out.update(
                    judge_case(
                        args.base_url,
                        args.model,
                        api_key,
                        edit_type,
                        base.get("text_orig", ""),
                        base.get("text_pair", ""),
                        orig_sheet,
                        edit_sheet,
                        timeout=args.timeout,
                        max_tokens=args.max_tokens,
                    )
                )
            except (Exception, URLError) as exc:  # noqa: BLE001
                out["error"] = repr(exc)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            counts[(model_name, edit_type)] += 1
            print(f"{idx}/{len(cases)} {model_name} {edit_type} {pair_uid}", flush=True)
            time.sleep(0.1)

    write_summary(args.out, args.summary)
    print(dict(counts))


if __name__ == "__main__":
    main()
