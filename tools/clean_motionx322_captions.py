#!/usr/bin/env python
"""Sanitize LLM-annotation artifacts (Input:/Output:/Ouput: prefixes and
ChatGPT meta-responses) that leaked into Motion-X 322D captions during
preparation. Recoverable captions are rewritten with regenerated spaCy pos
tags; unrecoverable meta-responses are dropped from the splits and reported.
"""
import argparse
import json
import re
import zipfile
from pathlib import Path

HASH = re.compile(r"^\s*#+\s*")
# A glued prefix (e.g. the "Ou" in the "OuOutput:" double-typo) is absorbed by
# the leading [a-z]* before the label keyword; the colon keeps this safe.
LABEL = re.compile(
    r"^\s*(?:revised\s+)?[a-z]*(?:input|output|ouput)\s*:?\s*", re.I)
LEADIN = re.compile(
    r"^\s*(?:sure[,!.:]?\s+)?(?:here'?s?|here is)\s+[^:]{0,80}:\s*", re.I)
LEADIN_SURE = re.compile(r"^\s*sure[,!.]?\s+i\s+[^:]{0,80}[:.]\s*", re.I)
QUOTES = re.compile(r'^["\u201c](.*)["\u201d]$')
TRAILANN = re.compile(
    r"\s*\((?:subject|verb|object|complete\s+sentence)\b.*?\)\s*$", re.I)
META = re.compile(
    r"(i apologize|i'm sorry|i cannot|as an ai|please provide|please give me|"
    r"provide me with|the subject of the sentence|is a term used|"
    r"complete the sentence|is not a clear sentence|assuming you meant|"
    r"give me an input|valid input|i'd be happy|i'm happy to help|"
    r"more context or information|what is your input|"
    r"provide (?:me )?(?:with )?(?:an?|the|your) input)", re.I)
# A source line needs cleaning if it opens with a hash/label/"Sure," lead-in.
INDICATOR = re.compile(r"^\s*#|(?:input|output|ouput)\s*:|^\s*sure[,!.]", re.I)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sanitize(text):
    text = re.sub(r"\s+", " ", text.strip())
    previous = None
    while text != previous:
        previous = text
        text = HASH.sub("", text)
        text = LABEL.sub("", text)
        text = LEADIN.sub("", text)
        text = LEADIN_SURE.sub("", text)
    text = TRAILANN.sub("", text).strip()
    match = QUOTES.match(text)
    return match.group(1).strip() if match else text


def is_unrecoverable(caption):
    return (not caption) or len(caption.split()) < 3 or bool(META.search(caption))


def tokenize_caption(nlp, caption):
    tokens = []
    for token in nlp(caption):
        if token.is_space or token.is_punct:
            continue
        word = token.text.lower()
        if token.pos_ in {"NOUN", "VERB"} and word != "left" and token.lemma_:
            word = token.lemma_.lower()
        tokens.append(f"{word}/{token.pos_ or 'OTHER'}")
    return tokens or ["unk/OTHER"]


def raw_first_line(text_zip, member):
    if not member:
        return ""
    raw = text_zip.read(member).decode("utf-8", errors="replace")
    for line in raw.splitlines():
        if line.strip():
            return line
    return ""


def main():
    args = parse_args()
    root = args.dataset_root
    meta = json.loads((root / "dataset_meta.json").read_text())
    text_zip = zipfile.ZipFile(meta["text_archive"])

    member_of = {}
    for line in (root / "manifest.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            member_of[row["motion_id"]] = row.get("text_member")

    import spacy
    nlp = spacy.load(args.spacy_model, disable=["parser", "ner"])

    cleaned, dropped = [], []
    new_split_counts = {}
    for split in ("train", "test", "val"):
        ids = [line.strip() for line in
               (root / f"{split}.txt").read_text().splitlines() if line.strip()]
        keep = []
        for motion_id in ids:
            raw = raw_first_line(text_zip, member_of.get(motion_id))
            if not INDICATOR.search(raw):
                keep.append(motion_id)
                continue
            caption = sanitize(raw)
            if is_unrecoverable(caption):
                dropped.append({"motion_id": motion_id, "split": split,
                                "raw": raw[:200]})
                continue
            pos = " ".join(tokenize_caption(nlp, caption))
            if not args.dry_run:
                (root / "texts" / f"{motion_id}.txt").write_text(
                    f"{caption}#{pos}#0.0#0.0\n", encoding="utf-8")
            cleaned.append(motion_id)
            keep.append(motion_id)
        new_split_counts[split] = len(keep)
        if not args.dry_run:
            (root / f"{split}.txt").write_text("\n".join(keep) + "\n",
                                               encoding="utf-8")

    dropped_ids = {d["motion_id"] for d in dropped}
    manifest_removed = 0
    if not args.dry_run and dropped_ids:
        manifest_path = root / "manifest.jsonl"
        kept_lines = []
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            if json.loads(line)["motion_id"] in dropped_ids:
                manifest_removed += 1
                continue
            kept_lines.append(line)
        manifest_path.write_text("\n".join(kept_lines) + "\n", encoding="utf-8")

    report = {
        "cleaned": len(cleaned),
        "dropped": len(dropped),
        "dropped_by_split": {s: sum(d["split"] == s for d in dropped)
                             for s in ("train", "test", "val")},
        "new_split_counts": new_split_counts,
        "manifest_removed": manifest_removed,
        "dropped_records": dropped,
    }
    if not args.dry_run:
        (root / "caption_clean_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8")
        meta["split_counts"] = new_split_counts
        meta["artifact_cleaned"] = len(cleaned)
        meta["artifact_dropped"] = len(dropped)
        (root / "dataset_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    summary = {k: v for k, v in report.items() if k != "dropped_records"}
    print(("DRY-RUN " if args.dry_run else "") + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
