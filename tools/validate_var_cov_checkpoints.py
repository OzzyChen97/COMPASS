#!/usr/bin/env python
"""Validate formal SATO++ variance/covariance checkpoints before evaluation."""

import argparse
from pathlib import Path


EXPECTED = {
    "full": (0.5, 0.04),
    "no_var": (0.0, 0.04),
    "no_cov": (0.5, 0.0),
    "no_both": (0.0, 0.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-prefix", default="humanml3d")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--iteration", type=int, required=True)
    args = parser.parse_args()

    import torch

    for variant, (expected_var, expected_cov) in EXPECTED.items():
        path = (
            args.run_root
            / f"{args.dataset_prefix}_{variant}_seed{args.seed}"
            / "sato_pp_final.pth"
        )
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing final checkpoint: {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        metadata = checkpoint.get("args", {})
        actual = {
            "iter": int(checkpoint.get("iter", -1)),
            "seed": int(metadata.get("seed", -1)),
            "lambda_var": float(metadata.get("lambda_var", float("nan"))),
            "lambda_cov": float(metadata.get("lambda_cov", float("nan"))),
            "attribution_create_graph": bool(
                metadata.get("attribution_create_graph", False)
            ),
        }
        expected = {
            "iter": args.iteration,
            "seed": args.seed,
            "lambda_var": expected_var,
            "lambda_cov": expected_cov,
            "attribution_create_graph": False,
        }
        if actual != expected:
            raise ValueError(
                f"invalid checkpoint metadata for {variant} at {path}: "
                f"expected {expected}, got {actual}"
            )
        print(
            f"OK {variant} iter={actual['iter']} seed={actual['seed']} "
            f"lambda_var={actual['lambda_var']} lambda_cov={actual['lambda_cov']} "
            f"attribution_create_graph={actual['attribution_create_graph']} path={path}"
        )


if __name__ == "__main__":
    main()
