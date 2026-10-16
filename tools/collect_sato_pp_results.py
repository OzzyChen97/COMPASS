#!/usr/bin/env python
"""Collect SATO++ experiment evidence from logs and evaluation summaries.

The script is intentionally read-only. It turns the scattered training logs and
perturbed-eval summaries under output_GPT_Final into JSON/CSV/LaTeX artifacts
that can be used by the TIP manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path


EVAL_RE = re.compile(
    r"Eva\. Iter\s+(?P<iter>\d+)\s*:?,\s+FID\.\s+(?P<fid>[0-9.]+),"
    r".*?Diversity\.\s+(?P<diversity>[0-9.]+),"
    r".*?R_precision\.\s+\[(?P<top1>[0-9.eE+-]+)\s+"
    r"(?P<top2>[0-9.eE+-]+)\s+(?P<top3>[0-9.eE+-]+)\],"
    r"\s+matching_score_real\.\s+(?P<matching_real>[0-9.eE+-]+),"
    r"\s+matching_score_pred\.\s+(?P<matching>[0-9.eE+-]+)"
)

TRAIN_BEST_RE = re.compile(
    r"Train\. Iter\s+(?P<iter>\d+)\s*:\s*FID\.\s+(?P<fid>[0-9.]+),"
    r"\s*Diversity\.\s+(?P<diversity>[0-9.]+),"
    r"\s*TOP1\.\s+(?P<top1>[0-9.]+),\s*TOP2\.\s+(?P<top2>[0-9.]+),"
    r"\s*TOP3\.\s+(?P<top3>[0-9.]+)"
)

PERT_RE = re.compile(
    r"FID\.?\s+(?P<fid>[0-9.]+),\s*FID_P\.?\s+(?P<fid_p>[0-9.]+),"
    r"\s*FID_D\.?\s+(?P<fid_d>[0-9.]+)"
)


EXPERIMENTS = [
    {
        "key": "baseline_pretrained",
        "name": "T2M-GPT",
        "kind": "baseline",
        "log": "output_GPT_Final/baseline_eval_perb_deterministic_v2/run.log",
        "note": "Original pretrained T2M-GPT on deterministic perturbation pairs.",
    },
    {
        "key": "baseline_aug_syn",
        "name": "T2M-GPT + synonym aug.",
        "kind": "baseline",
        "log": "output_GPT_Final/baseline_aug_syn/run.log",
        "perturbed": "output_GPT_Final/baseline_aug_syn/eval_perb_deterministic/summary.json",
        "note": "Generation baseline fine-tuned with synonym augmentation.",
    },
    {
        "key": "sato_v3a_kl_strong_low",
        "name": "SATO++ v3a",
        "kind": "ablation",
        "log": "output_GPT_Final/sato_v3a_strong_teach_low_sato_2gpu/run.log",
        "note": "Strong KL teacher, low SATO++ loss weights.",
    },
    {
        "key": "sato_v3b_kl_medium",
        "name": "SATO++ v3b",
        "kind": "ablation",
        "log": "output_GPT_Final/sato_v3b_med_teach_med_sato_2gpu/run.log",
        "note": "Medium KL teacher, medium SATO++ loss weights.",
    },
    {
        "key": "sato_v3c_l1_low",
        "name": "SATO++ v3c",
        "kind": "main_candidate",
        "log": "output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/run.log",
        "perturbed": "output_GPT_Final/sato_v3c_l1_teach_low_sato_2gpu/eval_perb_v3c/summary.json",
        "note": "L1 teacher, low SATO++ loss weights. Current best quality-retention run.",
    },
    {
        "key": "sato_v4a_kl_balanced",
        "name": "SATO++ v4a",
        "kind": "main_candidate",
        "log": "output_GPT_Final/sato_v4a_kl_teach_balanced_2gpu/run.log",
        "note": "Balanced KL teacher run; perturbation eval pending.",
    },
    {
        "key": "sato_v4b_kl_strong",
        "name": "SATO++ v4b",
        "kind": "ablation",
        "log": "output_GPT_Final/sato_v4b_kl_teach_sato_strong_2gpu/run.log",
        "perturbed": "output_GPT_Final/sato_v4b_kl_teach_sato_strong_2gpu/eval_perb/summary.json",
        "note": "Stronger SATO++ losses with KL teacher.",
    },
    {
        "key": "sato_v2_fullpair",
        "name": "SATO++ v2",
        "kind": "ablation",
        "log": "output_GPT_Final/sato_pp_v2_fullpair_6gpu_200k/run.log",
        "perturbed": "output_GPT_Final/sato_pp_v2_fullpair_6gpu_200k/eval_best_fid_perb/summary.json",
        "note": "Earlier full-pair training run.",
    },
]


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def best_standard_from_log(path: Path) -> dict:
    text = read_text(path)
    rows = []
    for m in EVAL_RE.finditer(text):
        row = {k: float(v) for k, v in m.groupdict().items() if k != "iter"}
        row["iter"] = int(m.group("iter"))
        rows.append(row)
    for m in TRAIN_BEST_RE.finditer(text):
        row = {k: float(v) for k, v in m.groupdict().items() if k != "iter"}
        row["iter"] = int(m.group("iter"))
        rows.append(row)
    if not rows:
        return {}
    best = min(rows, key=lambda x: x["fid"])
    best["source"] = str(path)
    return best


def perturbed_from_log(path: Path) -> dict:
    text = read_text(path)
    matches = list(PERT_RE.finditer(text))
    if not matches:
        return {}
    m = matches[-1]
    row = {k: float(v) for k, v in m.groupdict().items()}
    row["source"] = str(path)
    return row


def best_perturbed_from_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        return {}
    best = min(rows, key=lambda x: x.get("fid", float("inf")))
    out = {
        "iter": best.get("iter"),
        "fid": best.get("fid"),
        "fid_p": best.get("fid_p"),
        "fid_d": best.get("fid_d"),
        "matched_pairs": best.get("matched_pairs"),
        "total_pairs": best.get("total_pairs"),
        "checkpoint": best.get("checkpoint"),
        "source": str(path),
    }
    return out


def collect(root: Path) -> list[dict]:
    results = []
    for exp in EXPERIMENTS:
        row = {k: v for k, v in exp.items() if k not in {"log", "perturbed"}}
        log = root / exp["log"]
        row["standard"] = best_standard_from_log(log)

        perturbed = {}
        if "perturbed" in exp:
            perturbed = best_perturbed_from_summary(root / exp["perturbed"])
        if not perturbed:
            perturbed = perturbed_from_log(log)
        row["perturbed"] = perturbed
        results.append(row)
    return results


def write_csv(results: list[dict], path: Path) -> None:
    fields = [
        "key", "name", "kind", "std_iter", "fid", "diversity", "top1",
        "top2", "top3", "matching", "pert_iter", "fid_p", "fid_d",
        "matched_pairs", "total_pairs", "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in results:
            std = item.get("standard") or {}
            pert = item.get("perturbed") or {}
            writer.writerow({
                "key": item["key"],
                "name": item["name"],
                "kind": item["kind"],
                "std_iter": std.get("iter", ""),
                "fid": std.get("fid", pert.get("fid", "")),
                "diversity": std.get("diversity", ""),
                "top1": std.get("top1", ""),
                "top2": std.get("top2", ""),
                "top3": std.get("top3", ""),
                "matching": std.get("matching", ""),
                "pert_iter": pert.get("iter", ""),
                "fid_p": pert.get("fid_p", ""),
                "fid_d": pert.get("fid_d", ""),
                "matched_pairs": pert.get("matched_pairs", ""),
                "total_pairs": pert.get("total_pairs", ""),
                "note": item.get("note", ""),
            })


def fmt(x: object, digits: int = 3) -> str:
    if x in ("", None):
        return "--"
    try:
        return f"{float(x):.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def write_latex_table(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by tools/collect_sato_pp_results.py.",
        "\\begin{tabular}{lccccccc}",
        "\\toprule",
        "Method & FID $\\downarrow$ & R@1 $\\uparrow$ & Div. $\\rightarrow$ & FID$_P$ $\\downarrow$ & FID$_D$ $\\downarrow$ & Std. Iter & Pert. Iter \\\\",
        "\\midrule",
    ]
    for item in results:
        std = item.get("standard") or {}
        pert = item.get("perturbed") or {}
        if not std and not pert:
            continue
        fid = std.get("fid", pert.get("fid"))
        std_iter = std.get("iter", "--")
        pert_iter = pert.get("iter", "--")
        lines.append(
            f"{item['name']} & {fmt(fid)} & {fmt(std.get('top1'))} & "
            f"{fmt(std.get('diversity'))} & {fmt(pert.get('fid_p'))} & "
            f"{fmt(pert.get('fid_d'))} & {std_iter} & {pert_iter} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--out-dir", default="paper_tip/tables")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = collect(root)
    (out_dir / "sato_pp_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    write_csv(results, out_dir / "sato_pp_results.csv")
    write_latex_table(results, out_dir / "main_results_auto.tex")
    print(f"Wrote {out_dir / 'sato_pp_results.json'}")
    print(f"Wrote {out_dir / 'sato_pp_results.csv'}")
    print(f"Wrote {out_dir / 'main_results_auto.tex'}")


if __name__ == "__main__":
    main()
