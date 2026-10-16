#!/usr/bin/env python

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--max-fallback-rate", type=float, default=0.02)
    args = parser.parse_args()
    payload = json.loads(args.meta.read_text())
    motions = int(payload["unique_motions"])
    fallback = sum(int(value) for value in payload.get("fallback_counts", {}).values())
    total = motions * len(payload.get("levels", ["P1", "P2", "P3"]))
    rate = fallback / total if total else 1.0
    print(json.dumps({"fallback": fallback, "total": total, "rate": rate}, indent=2))
    if rate > args.max_fallback_rate:
        raise RuntimeError(
            f"Pair fallback rate {rate:.4%} exceeds {args.max_fallback_rate:.4%}"
        )


if __name__ == "__main__":
    main()
