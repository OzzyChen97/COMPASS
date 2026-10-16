#!/usr/bin/env python
"""Select one SATO++ checkpoint from validation-only FID/FID_P/FID_D."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


METRICS = ("fid", "fid_p", "fid_d")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--formal-spec", type=Path, required=True)
    return parser.parse_args()


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    fields = [
        "rank",
        "candidate_id",
        "iter",
        "checkpoint",
        "n",
        "fid_mean",
        "fid_std",
        "fid_p_mean",
        "fid_p_std",
        "fid_d_mean",
        "fid_d_std",
        "fid_normalized",
        "fid_p_normalized",
        "fid_d_normalized",
        "composite_score",
        "selected",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def close_enough(a, b, atol=1e-12):
    return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=atol)


def main():
    args = parse_args()
    candidate_manifest = json.loads(
        args.candidate_manifest.read_text(encoding="utf-8")
    )
    protocol = json.loads(args.protocol_manifest.read_text(encoding="utf-8"))
    if protocol.get("selection_status") != "frozen_before_validation_metrics":
        raise ValueError("Protocol manifest was not frozen before validation metrics")

    rule = protocol.get("checkpoint_selection_rule", {})
    expected_formula = (
        "mean((metric-min_across_candidates)/(max_across_candidates-"
        "min_across_candidates)) for metric in [FID,FID_P,FID_D]"
    )
    if rule.get("formula") != expected_formula:
        raise ValueError(f"Unexpected selection formula: {rule.get('formula')!r}")
    if rule.get("zero_range_value") != 0.0:
        raise ValueError("Unexpected zero-range convention")
    if rule.get("tie_break") != ["lower_validation_fid", "earlier_iter", "candidate_id"]:
        raise ValueError("Unexpected tie-break rule")

    expected_seeds = [int(seed) for seed in protocol["validation"]["seeds"]]
    expected_pairs = int(protocol["validation"]["sample_count"])
    candidates = {
        row["candidate_id"]: row for row in candidate_manifest["candidates"]
    }
    if set(candidates) != set(protocol["candidate_ids"]):
        raise ValueError("Candidate set differs from the frozen protocol manifest")

    rows_by_candidate = {candidate_id: {} for candidate_id in candidates}
    for runs_path in sorted(args.shard_root.glob("*/**/runs.csv")):
        for row in read_csv(runs_path):
            candidate_id = row["model"]
            if candidate_id not in rows_by_candidate:
                raise ValueError(f"Unexpected validation model {candidate_id} in {runs_path}")
            seed = int(row["seed"])
            if seed in rows_by_candidate[candidate_id]:
                raise ValueError(f"Duplicate {candidate_id} seed {seed}")
            candidate = candidates[candidate_id]
            actual_path = str(Path(row["path"]).resolve())
            if actual_path != str(Path(candidate["canonical_path"]).resolve()):
                raise ValueError(
                    f"Checkpoint path mismatch for {candidate_id}: {actual_path}"
                )
            if int(row["iter"]) != int(candidate["checkpoint_iter"]):
                raise ValueError(f"Checkpoint iter mismatch for {candidate_id}")
            if int(row["matched_pairs"]) != expected_pairs:
                raise ValueError(f"matched_pairs mismatch for {candidate_id} seed {seed}")
            if int(row["total_pairs"]) != expected_pairs:
                raise ValueError(f"total_pairs mismatch for {candidate_id} seed {seed}")
            rows_by_candidate[candidate_id][seed] = row

    summary = []
    for candidate_id, candidate in candidates.items():
        seed_rows = rows_by_candidate[candidate_id]
        if sorted(seed_rows) != expected_seeds:
            raise ValueError(
                f"Seed mismatch for {candidate_id}: got {sorted(seed_rows)}, "
                f"expected {expected_seeds}"
            )
        item = {
            "candidate_id": candidate_id,
            "iter": int(candidate["checkpoint_iter"]),
            "checkpoint": candidate["canonical_path"],
            "generation_sha256": candidate["generation_sha256"],
            "n": len(seed_rows),
            "seeds": expected_seeds,
            "runs": [seed_rows[seed] for seed in expected_seeds],
        }
        for metric in METRICS:
            values = [float(seed_rows[seed][metric]) for seed in expected_seeds]
            item[f"{metric}_mean"] = statistics.fmean(values)
            item[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(item)

    ranges = {}
    for metric in METRICS:
        values = [row[f"{metric}_mean"] for row in summary]
        ranges[metric] = {"min": min(values), "max": max(values)}
        span = ranges[metric]["max"] - ranges[metric]["min"]
        for row in summary:
            row[f"{metric}_normalized"] = (
                (row[f"{metric}_mean"] - ranges[metric]["min"]) / span
                if span > 0.0
                else 0.0
            )

    for row in summary:
        row["composite_score"] = statistics.fmean(
            row[f"{metric}_normalized"] for metric in METRICS
        )

    summary.sort(
        key=lambda row: (
            row["composite_score"],
            row["fid_mean"],
            row["iter"],
            row["candidate_id"],
        )
    )
    selected = summary[0]
    for rank, row in enumerate(summary, start=1):
        row["rank"] = rank
        row["selected"] = row is selected

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "selection_type": "post_training_independent_validation_selection",
        "candidate_manifest": str(args.candidate_manifest.resolve()),
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "checkpoint_selection_rule": rule,
        "metric_ranges": ranges,
        "selected_candidate_id": selected["candidate_id"],
        "selected_checkpoint": selected["checkpoint"],
        "selected_iter": selected["iter"],
        "selected_generation_sha256": selected["generation_sha256"],
        "selected_validation_metrics": {
            key: selected[key]
            for key in (
                "fid_mean",
                "fid_std",
                "fid_p_mean",
                "fid_p_std",
                "fid_d_mean",
                "fid_d_std",
                "fid_normalized",
                "fid_p_normalized",
                "fid_d_normalized",
                "composite_score",
            )
        },
        "ranking": summary,
        "test_metrics_used_for_selection": False,
    }
    (output_dir / "selection_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_rows = []
    for row in summary:
        csv_rows.append(
            {
                "rank": row["rank"],
                "candidate_id": row["candidate_id"],
                "iter": row["iter"],
                "checkpoint": row["checkpoint"],
                "n": row["n"],
                "fid_mean": row["fid_mean"],
                "fid_std": row["fid_std"],
                "fid_p_mean": row["fid_p_mean"],
                "fid_p_std": row["fid_p_std"],
                "fid_d_mean": row["fid_d_mean"],
                "fid_d_std": row["fid_d_std"],
                "fid_normalized": row["fid_normalized"],
                "fid_p_normalized": row["fid_p_normalized"],
                "fid_d_normalized": row["fid_d_normalized"],
                "composite_score": row["composite_score"],
                "selected": row["selected"],
            }
        )
    write_csv(output_dir / "validation_ranking.csv", csv_rows)

    lines = [
        "# SATO++ Validation Checkpoint Selection",
        "",
        "The checkpoint was selected only from the independent validation subset; no test metric entered the selection rule.",
        "",
        "| Rank | Candidate | Iter | FID mean±std | FID_P mean±std | FID_D mean±std | Composite | Selected |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['rank']} | `{row['candidate_id']}` | {row['iter']} | "
            f"{row['fid_mean']:.6f}±{row['fid_std']:.6f} | "
            f"{row['fid_p_mean']:.6f}±{row['fid_p_std']:.6f} | "
            f"{row['fid_d_mean']:.6f}±{row['fid_d_std']:.6f} | "
            f"{row['composite_score']:.9f} | "
            f"{'yes' if row['selected'] else ''} |"
        )
    lines.extend(
        [
            "",
            f"Selected checkpoint: `{selected['checkpoint']}` (iter {selected['iter']}).",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")

    formal_spec = [
        {
            "name": (
                "motionx322_satopp_val_composite_selected_"
                f"iter{selected['iter']}"
            ),
            "kind": "sato_pp",
            "checkpoint": selected["checkpoint"],
            "description": (
                "SATO++ selected after training by independent-validation "
                "equal-weight min-max composite of FID/FID_P/FID_D"
            ),
        }
    ]
    args.formal_spec.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.formal_spec.resolve().write_text(
        json.dumps(formal_spec, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()