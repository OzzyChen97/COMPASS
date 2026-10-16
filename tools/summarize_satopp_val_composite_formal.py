#!/usr/bin/env python
"""Summarize the validation-selected SATO++ formal Table 3 retest."""

import argparse
import json
from pathlib import Path


METRICS = ("fid", "fid_p", "fid_d", "top1", "top2", "top3", "matching")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--previous-summary", type=Path, required=True)
    return parser.parse_args()


def by_kind(summary, kind):
    rows = [row for row in summary["summary_wide"] if row.get("kind") == kind]
    if len(rows) != 1:
        raise ValueError(f"Expected one {kind} summary row, got {len(rows)}")
    return rows[0]


def main():
    args = parse_args()
    root = args.root.resolve()
    selection = json.loads(
        (root / "selection/result/selection_result.json").read_text(encoding="utf-8")
    )
    protocol = json.loads(
        (root / "selection/protocol_manifest.json").read_text(encoding="utf-8")
    )
    val_meta = json.loads(
        (
            root
            / "validation/val_pairs_train_gemini_exact_mixed.json.meta.json"
        ).read_text(encoding="utf-8")
    )
    formal_summary = json.loads(
        (root / "formal/table3_repeat20/combined/summary.json").read_text(
            encoding="utf-8"
        )
    )
    previous_summary = json.loads(args.previous_summary.read_text(encoding="utf-8"))

    current = by_kind(formal_summary, "sato_pp")
    previous = by_kind(previous_summary, "sato_pp")
    baseline = by_kind(previous_summary, "baseline")
    if int(current["n"]) != 20:
        raise ValueError(f"Formal retest has n={current['n']}, expected 20")
    if int(selection["selected_iter"]) != int(
        Path(current["path"]).name.split("_")[-1].split(".")[0]
    ):
        if int(selection["selected_iter"]) != int(
            formal_summary["runs"][0]["iter"]
        ):
            raise ValueError("Formal summary does not use the selected checkpoint")
    expected_seeds = protocol["formal_test"]["seeds"]
    actual_seeds = sorted(int(row["seed"]) for row in formal_summary["runs"])
    if actual_seeds != expected_seeds:
        raise ValueError(f"Formal seed mismatch: {actual_seeds}")
    expected_samples = int(protocol["formal_test"]["sample_count"])
    for row in formal_summary["runs"]:
        if int(row["matched_pairs"]) != expected_samples:
            raise ValueError("Formal matched-pair count mismatch")
        if int(row["total_pairs"]) != expected_samples:
            raise ValueError("Formal total-pair count mismatch")

    deltas_previous = {
        metric: current[f"{metric}_mean"] - previous[f"{metric}_mean"]
        for metric in METRICS
    }
    deltas_baseline = {
        metric: current[f"{metric}_mean"] - baseline[f"{metric}_mean"]
        for metric in METRICS
    }
    payload = {
        "schema_version": 1,
        "selection": {
            "type": selection["selection_type"],
            "selected_candidate_id": selection["selected_candidate_id"],
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_iter": selection["selected_iter"],
            "selected_validation_metrics": selection[
                "selected_validation_metrics"
            ],
            "test_metrics_used_for_selection": selection[
                "test_metrics_used_for_selection"
            ],
        },
        "validation_protocol": {
            "sample_count": protocol["validation"]["sample_count"],
            "full_validation_split_count": val_meta["target_split_motions"],
            "coverage_fraction": val_meta["coverage_fraction"],
            "seeds": protocol["validation"]["seeds"],
            "level_counts": val_meta["output_level_counts"],
            "construction": val_meta["construction"],
            "reason": val_meta["reason"],
            "test_pair_files_used": val_meta["test_pair_files_used"],
        },
        "formal_protocol": protocol["formal_test"],
        "formal_result": current,
        "previous_endpoint_result": previous,
        "t2mgpt_baseline_reference": baseline,
        "delta_selected_minus_previous_endpoint": deltas_previous,
        "delta_selected_minus_t2mgpt_baseline": deltas_baseline,
    }
    (root / "final_report.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def metric_cell(row, metric):
        return f"{row[f'{metric}_mean']:.4f}±{row[f'{metric}_std']:.4f}"

    lines = [
        "# SATO++ Validation-Selected Formal Retest",
        "",
        "## Checkpoint selection",
        "",
        f"- Selected checkpoint: `{selection['selected_checkpoint']}`",
        f"- Selected iteration: `{selection['selected_iter']}`",
        "- Rule: equal-weight min-max composite of validation FID, FID_P, and FID_D; lower is better.",
        f"- Validation seeds: `{protocol['validation']['seeds']}`",
        f"- Validation subset: `{val_meta['output_pairs']}/{val_meta['target_split_motions']}` motions ({100 * val_meta['coverage_fraction']:.1f}%).",
        f"- Validation construction: {val_meta['construction']}.",
        "- Test metrics used for checkpoint selection: `false`.",
        "",
        "## Motion-X-322 Table 3 formal result",
        "",
        "| Result | n | FID | FID_P | FID_D | Top-1 | Matching |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| Validation-selected SATO++ | {current['n']} | "
            f"{metric_cell(current, 'fid')} | {metric_cell(current, 'fid_p')} | "
            f"{metric_cell(current, 'fid_d')} | {metric_cell(current, 'top1')} | "
            f"{metric_cell(current, 'matching')} |"
        ),
        (
            f"| Previous endpoint SATO++ | {previous['n']} | "
            f"{metric_cell(previous, 'fid')} | {metric_cell(previous, 'fid_p')} | "
            f"{metric_cell(previous, 'fid_d')} | {metric_cell(previous, 'top1')} | "
            f"{metric_cell(previous, 'matching')} |"
        ),
        (
            f"| T2M-GPT baseline reference | {baseline['n']} | "
            f"{metric_cell(baseline, 'fid')} | {metric_cell(baseline, 'fid_p')} | "
            f"{metric_cell(baseline, 'fid_d')} | {metric_cell(baseline, 'top1')} | "
            f"{metric_cell(baseline, 'matching')} |"
        ),
        "",
        "Positive deltas below mean the selected result is numerically larger; FID-family metrics are lower-is-better.",
        "",
        "| Metric | Selected − previous endpoint | Selected − T2M-GPT baseline |",
        "| --- | ---: | ---: |",
    ]
    for metric in METRICS:
        lines.append(
            f"| {metric.upper()} | {deltas_previous[metric]:+.6f} | "
            f"{deltas_baseline[metric]:+.6f} |"
        )
    lines.append("")
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()