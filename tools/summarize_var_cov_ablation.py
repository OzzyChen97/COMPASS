#!/usr/bin/env python
"""Merge repeat-20 shards and report variance/covariance ablation effects."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

METRICS = ("fid", "fid_p", "fid_d", "diversity", "top1",
           "top2", "top3", "matching")
ORDER = ("full", "no_var", "no_cov", "no_both")
START_SEED = 123
NUM_SEEDS = 20
EXPECTED_SEEDS = set(range(START_SEED, START_SEED + NUM_SEEDS))
EXPECTED_PAIRS = {"humanml3d": 4192, "motionx322": 7733}
EXPECTED_ITERS = {"humanml3d": 150000, "motionx322": 100000}


def canonical_sha256(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_rows(eval_root: Path, dataset: str):
    rows = {}
    run_rows = {}
    expected_pairs = EXPECTED_PAIRS[dataset]
    expected_iter = EXPECTED_ITERS[dataset]
    for gpu in range(4):
        path = eval_root / f"gpu{gpu}" / "summary_wide.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            data = list(csv.DictReader(handle))
        if len(data) != 1 or int(data[0]["n"]) != NUM_SEEDS:
            raise ValueError(f"expected one 20-seed row in {path}, got {data}")
        row = data[0]
        prefix = f"{dataset}_"
        if not row["model"].startswith(prefix):
            raise ValueError(f"unexpected model {row['model']} in {path}")
        variant = row["model"][len(prefix):]
        if variant not in ORDER:
            raise ValueError(f"unexpected variant {variant} in {path}")
        if variant in rows:
            raise ValueError(f"duplicate variant {variant} in {path}")
        runs_path = eval_root / f"gpu{gpu}" / row["model"] / "runs.csv"
        with runs_path.open(newline="", encoding="utf-8") as handle:
            runs = list(csv.DictReader(handle))
        seeds = [int(run["seed"]) for run in runs]
        run_models = {run["model"] for run in runs}
        if run_models != {row["model"]}:
            raise ValueError(
                f"unexpected run models in {runs_path}: {sorted(run_models)}"
            )
        if len(runs) != NUM_SEEDS:
            raise ValueError(
                f"expected {NUM_SEEDS} rows in {runs_path}, got {len(runs)}"
            )
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"duplicate seeds in {runs_path}: {seeds}")
        if set(seeds) != EXPECTED_SEEDS:
            raise ValueError(
                f"expected seeds {sorted(EXPECTED_SEEDS)} in {runs_path}, "
                f"got {sorted(seeds)}"
            )
        if int(row["n"]) != len(runs):
            raise ValueError(
                f"summary count {row['n']} does not match {len(runs)} runs in {runs_path}"
            )
        run_indices = [int(run["run_idx"]) for run in runs]
        if set(run_indices) != set(range(1, NUM_SEEDS + 1)):
            raise ValueError(
                f"expected run_idx 1..{NUM_SEEDS} in {runs_path}, "
                f"got {sorted(run_indices)}"
            )
        for run in runs:
            seed = int(run["seed"])
            run_idx = int(run["run_idx"])
            if run_idx != seed - START_SEED + 1:
                raise ValueError(
                    f"run_idx/seed mismatch in {runs_path}: "
                    f"run_idx={run_idx}, seed={seed}"
                )
            if int(run["iter"]) != expected_iter:
                raise ValueError(
                    f"checkpoint iteration mismatch in {runs_path} seed {seed}: "
                    f"expected {expected_iter}, got {run['iter']}"
                )
            matched = int(run["matched_pairs"])
            total = int(run["total_pairs"])
            if matched != expected_pairs or total != expected_pairs:
                raise ValueError(
                    f"incomplete pair coverage in {runs_path} seed {run['seed']}: "
                    f"expected {expected_pairs}/{expected_pairs}, got "
                    f"{matched}/{total}"
                )
            run_hash = run["sample_index_sha256"]
            if not run_hash:
                raise ValueError(
                    f"missing sample-index hash in {runs_path} seed {seed}"
                )
            index_path = (
                eval_root / f"gpu{gpu}" / row["model"] / "sample_indices"
                / f"seed_{seed}.json"
            )
            payload = json.loads(index_path.read_text(encoding="utf-8"))
            samples = payload.get("samples")
            if not isinstance(samples, list) or len(samples) != expected_pairs:
                raise ValueError(
                    f"expected {expected_pairs} sample-index rows in {index_path}, "
                    f"got {0 if not isinstance(samples, list) else len(samples)}"
                )
            if int(payload.get("seed", -1)) != seed:
                raise ValueError(
                    f"sample-index seed mismatch in {index_path}: "
                    f"expected {seed}, got {payload.get('seed')}"
                )
            if payload.get("model") != row["model"]:
                raise ValueError(
                    f"sample-index model mismatch in {index_path}: "
                    f"expected {row['model']}, got {payload.get('model')}"
                )
            if int(payload.get("sample_count", -1)) != expected_pairs:
                raise ValueError(
                    f"sample-index count metadata mismatch in {index_path}: "
                    f"expected {expected_pairs}, got {payload.get('sample_count')}"
                )
            computed_hash = canonical_sha256(samples)
            recorded_hash = payload.get("sample_index_sha256")
            if computed_hash != run_hash or recorded_hash != run_hash:
                raise ValueError(
                    f"sample-index checksum mismatch in {index_path}: "
                    f"run={run_hash}, recorded={recorded_hash}, "
                    f"computed={computed_hash}"
                )
        rows[variant] = row
        run_rows[variant] = {int(run["seed"]): run for run in runs}
    missing = set(ORDER) - set(rows)
    if missing:
        raise ValueError(f"missing variants: {sorted(missing)}")
    reference = run_rows[ORDER[0]]
    for variant in ORDER[1:]:
        for seed in sorted(EXPECTED_SEEDS):
            ref_hash = reference[seed]["sample_index_sha256"]
            actual_hash = run_rows[variant][seed]["sample_index_sha256"]
            if actual_hash != ref_hash:
                raise ValueError(
                    f"sample-index mismatch for seed {seed}: full={ref_hash}, "
                    f"{variant}={actual_hash}"
                )
    return rows


def effect_rows(rows):
    effects = []
    for metric in METRICS:
        values = {key: float(rows[key][f"{metric}_mean"]) for key in ORDER}
        effects.append({
            "metric": metric,
            "delta_remove_var": values["no_var"] - values["full"],
            "delta_remove_cov": values["no_cov"] - values["full"],
            "delta_remove_both": values["no_both"] - values["full"],
            "interaction": (values["no_both"] - values["no_var"]
                            - values["no_cov"] + values["full"]),
        })
    return effects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=("humanml3d", "motionx322"), required=True)
    parser.add_argument(
        "--no-second-order-attribution",
        action="store_true",
        help=("record that attribution_create_graph was disabled, so variance/"
              "covariance gradients cannot reach the generation backbone"),
    )
    args = parser.parse_args()
    rows = read_rows(args.eval_root, args.dataset)
    effects = effect_rows(rows)

    summary_path = args.eval_root / "summary.csv"
    fields = ["model", "kind", "checkpoint", "path", "n"]
    for metric in METRICS:
        fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_var"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for variant in ORDER:
            writer.writerow(rows[variant])

    effect_path = args.eval_root / "effects.csv"
    with effect_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(effects[0]))
        writer.writeheader(); writer.writerows(effects)

    interpretation = None
    if args.no_second_order_attribution:
        interpretation = (
            "The training recipe disabled attribution_create_graph. Variance "
            "and covariance losses therefore update the attribution pooling "
            "and predictor heads, but their gradients cannot propagate through "
            "the attribution computation into the generation transformer."
        )
    audit = {
        "variants": list(ORDER),
        "seed_start": START_SEED,
        "seed_end": START_SEED + NUM_SEEDS - 1,
        "seeds_per_variant": NUM_SEEDS,
        "run_idx_start": 1,
        "run_idx_end": NUM_SEEDS,
        "expected_pairs_per_seed": EXPECTED_PAIRS[args.dataset],
        "expected_checkpoint_iter": EXPECTED_ITERS[args.dataset],
        "exact_pair_coverage_verified": True,
        "sample_index_count_verified": True,
        "sample_index_checksums_verified": True,
        "cross_variant_sample_index_alignment_verified": True,
    }
    payload = {"dataset": args.dataset, "seeds": sorted(EXPECTED_SEEDS),
               "rows": [rows[v] for v in ORDER], "effects": effects,
               "audit": audit, "interpretation": interpretation}
    (args.eval_root / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [f"# {args.dataset} variance/covariance ablation", "",
             "All values are mean ± sample standard deviation over seeds 123--142.", "",
             "## Formal audit", "",
             f"- Variants: `{', '.join(ORDER)}`",
             f"- Seeds per variant: `{NUM_SEEDS}` (`{START_SEED}`--"
             f"`{START_SEED + NUM_SEEDS - 1}`)",
             f"- Run indices: `1`--`{NUM_SEEDS}` with exact seed mapping",
             f"- Checkpoint iteration: `{EXPECTED_ITERS[args.dataset]}`",
             f"- Pair coverage per seed: `{EXPECTED_PAIRS[args.dataset]}`/"
             f"`{EXPECTED_PAIRS[args.dataset]}`",
             "- Sample-index counts and SHA-256 checksums: verified",
             "- Cross-variant sample-index alignment for each seed: verified", "",
             "## Metrics", "",
             "| Variant | FID | FID-P | FID-D | Diversity | Top-1 | Matching |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for variant in ORDER:
        row = rows[variant]
        cell = lambda m: f"{float(row[f'{m}_mean']):.4f} ± {float(row[f'{m}_std']):.4f}"
        lines.append(f"| {variant} | {cell('fid')} | {cell('fid_p')} | {cell('fid_d')} | "
                     f"{cell('diversity')} | {cell('top1')} | {cell('matching')} |")
    lines += ["", "Interaction is the difference-in-differences: "
              "no_both - no_var - no_cov + full.", "",
              "| Metric | Δ remove Var | Δ remove Cov | Δ remove Both | Interaction |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for row in effects:
        lines.append("| {metric} | {delta_remove_var:.6f} | {delta_remove_cov:.6f} | "
                     "{delta_remove_both:.6f} | {interaction:.6f} |".format(**row))
    if interpretation is not None:
        lines += ["", "## Interpretation", "", interpretation]
    (args.eval_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
