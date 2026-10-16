#!/usr/bin/env python
import argparse
import json
from pathlib import Path


def parse_model(value):
    name, separator, checkpoint = value.partition("=")
    if not separator or not name or not checkpoint:
        raise argparse.ArgumentTypeError("model must be NAME=CHECKPOINT")
    return name, checkpoint


def main():
    parser = argparse.ArgumentParser(
        description="Write an evaluation spec comparing SATO++ checkpoints."
    )
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    specs = []
    for name, checkpoint in args.model:
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append(
            {
                "name": name,
                "kind": "sato_pp",
                "checkpoint": checkpoint,
                "description": f"SATO++ checkpoint {name}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(specs, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
