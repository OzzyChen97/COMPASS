#!/usr/bin/env python
"""Download the required Motion-X++ archives from an authorized bypy account."""

import argparse
import json
import subprocess
from pathlib import Path


MOTION_ARCHIVES = (
    "animation.zip", "fitness.zip", "game_motion.zip", "haa500.zip",
    "humman.zip", "idea400.zip", "kungfu.zip", "music.zip", "perform.zip",
)
TEXT_ARCHIVES = (
    "animation.zip", "haa500.zip", "humman.zip", "idea400.zip",
    "kungfu.zip", "music.zip", "perform.zip",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bypy", default="bypy")
    parser.add_argument("--remote-root", default="Motion-X++")
    parser.add_argument("--output", type=Path, default=Path("dataset/Motion-X/raw/bypy"))
    return parser.parse_args()


def download(binary, remote, local):
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([binary, "downfile", remote, str(local)], check=True)
    if not local.is_file() or local.stat().st_size == 0:
        raise RuntimeError(f"bypy did not produce {local}")
    return {"remote": remote, "local": str(local.resolve()), "bytes": local.stat().st_size}


def main():
    args = parse_args()
    records = []
    motion_local = args.output / "motion/motion_generation/smplx322"
    text_local = args.output / "text/semantic_label"
    for name in MOTION_ARCHIVES:
        records.append(download(
            args.bypy,
            f"{args.remote_root}/motion/motion_generation/smplx322/{name}",
            motion_local / name,
        ))
    for name in TEXT_ARCHIVES:
        records.append(download(
            args.bypy,
            f"{args.remote_root}/text/semantic_label/{name}",
            text_local / name,
        ))
    manifest = {
        "transport": "bypy",
        "remote_root": args.remote_root,
        "total_bytes": sum(record["bytes"] for record in records),
        "files": records,
    }
    path = args.output / "download_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
