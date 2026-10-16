#!/usr/bin/env python3
"""Build and post-process a fixed MotionGPT baseline/SATO++ Blender demo set.

The builder consumes JSON/JSONL candidate metadata, deterministically selects
3 preserve cases, 3 alter cases (direction, speed, body-part/action), and one
failure case, then writes a reproducible manifest and a shell command file.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


REPO_ROOT = Path(__file__).resolve().parents[1]
MOTIONGPT_ROOT = REPO_ROOT / "MotionGPT"
SCHEMA_VERSION = 1
MODEL_ALIASES = {
    "baseline": ("baseline", "motiongpt", "motiongpt_baseline"),
    "sato_pp": ("sato_pp", "sato", "motiongpt_sato_pp"),
}


class ManifestError(ValueError):
    """Candidate metadata is incomplete or violates matched-demo rules."""


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text[0] in "[{":
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            if "Extra data" not in str(exc):
                raise
        else:
            if isinstance(parsed, dict):
                if "candidates" in parsed or "cases" in parsed:
                    parsed = parsed.get("candidates", parsed.get("cases", []))
                else:
                    parsed = [parsed]
            if not isinstance(parsed, list):
                raise ManifestError("candidate JSON must be a list or contain 'candidates'")
            return parsed
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _first(mapping: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return default


def _model_record(row: dict[str, Any], canonical_name: str) -> dict[str, Any]:
    for key in MODEL_ALIASES[canonical_name]:
        value = row.get(key)
        if isinstance(value, dict):
            return value
    outputs = row.get("outputs")
    if isinstance(outputs, dict):
        for key in MODEL_ALIASES[canonical_name]:
            value = outputs.get(key)
            if isinstance(value, dict):
                return value
    raise ManifestError(f"missing {canonical_name} output metadata")


def _resolve_path(value: Any, base_dir: Path) -> str | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path.resolve())


def _normalize_edit_type(value: Any) -> str:
    text = str(value or "").lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "bodypart": "body_part",
        "body_parts": "body_part",
        "body": "body_part",
        "tempo": "speed",
        "directional": "direction",
    }
    return aliases.get(text, text)


def normalize_candidate(row: dict[str, Any], index: int, base_dir: Path) -> dict[str, Any]:
    case_id = str(_first(row, ("case_id", "id", "motion_id"), f"case_{index:04d}"))
    common_prompt = _first(row, ("prompt", "text", "caption", "text_pair"))
    common_seed = _first(row, ("seed", "generation_seed"))
    common_length = _first(row, ("length", "motion_length", "num_frames"))
    relation = str(_first(row, ("relation", "pair_type"), "")).lower()
    edit_type = _normalize_edit_type(_first(row, ("edit_type", "semantic_type", "category")))
    failure = bool(row.get("is_failure", False)) or str(
        _first(row, ("outcome", "status"), "")
    ).lower() == "failure" or relation == "failure"

    models: dict[str, dict[str, Any]] = {}
    for model_name in ("baseline", "sato_pp"):
        source = _model_record(row, model_name)
        prompt = _first(source, ("prompt", "text", "caption"), common_prompt)
        seed = _first(source, ("seed", "generation_seed"), common_seed)
        length = _first(source, ("length", "motion_length", "num_frames"), common_length)
        motion_path = _resolve_path(
            _first(source, ("motion_path", "npy_path", "joints_path", "result_path")), base_dir
        )
        attribution_path = _resolve_path(
            _first(source, ("attribution_path", "attribution", "heatmap_path")), base_dir
        )
        models[model_name] = {
            "prompt": prompt,
            "seed": seed,
            "length": length,
            "motion_path": motion_path,
            "attribution_path": attribution_path,
        }

    left, right = models["baseline"], models["sato_pp"]
    for field in ("prompt", "seed", "length"):
        if left[field] is None or right[field] is None:
            raise ManifestError(f"{case_id}: {field} must be present for both models")
        if left[field] != right[field]:
            raise ManifestError(
                f"{case_id}: baseline/SATO++ {field} mismatch "
                f"({left[field]!r} != {right[field]!r})"
            )
    for model_name, model in models.items():
        if not model["motion_path"]:
            raise ManifestError(f"{case_id}: missing {model_name} motion_path")

    return {
        "case_id": case_id,
        "dataset": _first(row, ("dataset", "dataname")),
        "relation": relation,
        "edit_type": edit_type,
        "is_failure": failure,
        "selection_score": float(_first(row, ("selection_score", "improvement", "score"), 0.0)),
        "failure_score": float(_first(row, ("failure_score", "severity"), 0.0)),
        "prompt_orig": _first(row, ("prompt_orig", "text_orig", "source_prompt")),
        "prompt": left["prompt"],
        "seed": left["seed"],
        "length": left["length"],
        "models": models,
    }


def _rank(rows: Iterable[dict[str, Any]], score_key: str = "selection_score") -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row[score_key]), row["case_id"]))


def select_cases(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ids = [row["case_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ManifestError("case_id values must be unique")

    preserve = _rank(
        row for row in candidates if row["relation"] == "preserve" and not row["is_failure"]
    )
    if len(preserve) < 3:
        raise ManifestError(f"need at least 3 non-failure preserve candidates; found {len(preserve)}")

    alter = [
        row for row in candidates if row["relation"] == "alter" and not row["is_failure"]
    ]
    chosen_alter: list[dict[str, Any]] = []
    for label, allowed in (
        ("direction", {"direction"}),
        ("speed", {"speed"}),
        ("body_part/action", {"body_part", "action", "body_part_action"}),
    ):
        pool = _rank(row for row in alter if row["edit_type"] in allowed)
        if not pool:
            raise ManifestError(f"need a non-failure alter candidate for {label}")
        chosen_alter.append(pool[0])

    used = {row["case_id"] for row in preserve[:3] + chosen_alter}
    failures = _rank(
        (row for row in candidates if row["is_failure"] and row["case_id"] not in used),
        score_key="failure_score",
    )
    if not failures:
        raise ManifestError("need at least one distinct failure candidate")

    selected = preserve[:3] + chosen_alter + failures[:1]
    reasons = [
        "top-ranked preserve 1/3",
        "top-ranked preserve 2/3",
        "top-ranked preserve 3/3",
        "top-ranked alter: direction",
        "top-ranked alter: speed",
        "top-ranked alter: body_part/action",
        "top-ranked declared failure",
    ]
    return [{**row, "selection_reason": reason} for row, reason in zip(selected, reasons)]


def detect_blender(explicit: str | None = None) -> dict[str, Any]:
    candidates: list[tuple[str, str | None]] = [
        ("--blender", explicit),
        ("BLENDER_PATH", os.environ.get("BLENDER_PATH")),
        ("MotionGPT portable", str(MOTIONGPT_ROOT / "libs/blender-2.93.2-linux-x64/blender")),
        ("repository portable", str(REPO_ROOT / "libs/blender-2.93.2-linux-x64/blender")),
        ("PATH", shutil.which("blender")),
    ]
    checked: list[str] = []
    for source, raw_path in candidates:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        checked.append(str(path))
        if path.is_file() and os.access(path, os.X_OK):
            version_hint = "2.93.2" if "2.93.2" in str(path) else "unknown"
            return {
                "status": "ready" if version_hint == "2.93.2" else "version_unverified",
                "path": str(path),
                "source": source,
                "required_version": "2.93.2",
                "version_hint": version_hint,
                "checked": checked,
            }
    return {
        "status": "missing",
        "path": None,
        "required_version": "2.93.2",
        "message": (
            "Blender 2.93.2 was not found. Install the portable build at "
            "MotionGPT/libs/blender-2.93.2-linux-x64/blender or pass --blender; "
            "the manifest remains usable and rendering was not attempted."
        ),
        "checked": checked,
    }


def _shell_command(parts: list[str]) -> str:
    return shlex.join(parts)


def _expected_video(motion_path: str) -> str:
    path = Path(motion_path)
    return str(path.with_suffix(".mp4"))


def add_commands(
    selected: list[dict[str, Any]], blender: dict[str, Any], output_dir: Path, render_cfg: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    commands: list[str] = []
    enriched: list[dict[str, Any]] = []
    script_path = Path(__file__).resolve()
    blender_path = blender.get("path") or "BLENDER_2_93_2_NOT_FOUND"

    for case in selected:
        case_copy = dict(case)
        case_copy["models"] = {}
        for model_name, model in case["models"].items():
            model_copy = dict(model)
            motion_path = model["motion_path"]
            video_path = _expected_video(motion_path)
            render_parts = [
                blender_path,
                "--background",
                "--python",
                str(MOTIONGPT_ROOT / "render.py"),
                "--",
                f"--cfg={render_cfg}",
                f"--npy={motion_path}",
                "--mode=video",
            ]
            render_command = _shell_command(render_parts)
            artifact_dir = output_dir / "artifacts" / case["case_id"] / model_name
            contact_sheet = artifact_dir / "contact_sheet.png"
            trajectory = artifact_dir / "root_trajectory.png"
            postprocess: list[dict[str, Any]] = []

            for kind, input_path, output_path in (
                ("contact-sheet", video_path, contact_sheet),
                ("root-trajectory", motion_path, trajectory),
            ):
                parts = [
                    sys.executable,
                    str(script_path),
                    "postprocess",
                    "--kind",
                    kind,
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
                entry = {"kind": kind, "command": _shell_command(parts), "output": str(output_path)}
                postprocess.append(entry)

            if model.get("attribution_path"):
                heatmap = artifact_dir / "attribution_heatmap.png"
                parts = [
                    sys.executable,
                    str(script_path),
                    "postprocess",
                    "--kind",
                    "attribution-heatmap",
                    "--input",
                    model["attribution_path"],
                    "--output",
                    str(heatmap),
                ]
                postprocess.append(
                    {"kind": "attribution-heatmap", "command": _shell_command(parts), "output": str(heatmap)}
                )

            model_copy.update(
                {
                    "expected_video_path": video_path,
                    "render_status": "ready" if blender.get("path") else "blocked_missing_blender",
                    "render_command": render_command,
                    "postprocess": postprocess,
                }
            )
            commands.append(render_command)
            commands.extend(item["command"] for item in postprocess)
            case_copy["models"][model_name] = model_copy
        enriched.append(case_copy)
    return enriched, commands


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    candidate_path = Path(args.candidates).expanduser().resolve()
    rows = _read_records(candidate_path)
    normalized = [normalize_candidate(row, idx, candidate_path.parent) for idx, row in enumerate(rows)]
    selected = select_cases(normalized)
    blender = detect_blender(args.blender)
    output_dir = Path(args.output_dir).expanduser().resolve()
    render_cfg = Path(args.render_config).expanduser().resolve()
    selected, commands = add_commands(selected, blender, output_dir, render_cfg)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "selection_protocol": "3 preserve + direction + speed + body_part/action + 1 failure",
        "source_candidates": str(candidate_path),
        "render_config": str(render_cfg),
        "blender": blender,
        "selected_cases": selected,
        "summary": {
            "candidate_count": len(normalized),
            "selected_count": len(selected),
            "render_command_count": len(selected) * 2,
        },
    }

    if args.dry_run:
        return manifest

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    command_path = output_dir / "run_demo_commands.sh"
    manifest["manifest_path"] = str(manifest_path)
    manifest["command_file"] = str(command_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    command_path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n"
        + f"cd {shlex.quote(str(MOTIONGPT_ROOT))}\n\n"
        + "\n".join(commands)
        + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o755)

    if args.run_blender:
        if not blender.get("path"):
            manifest["execution"] = {"status": "skipped_missing_blender"}
        else:
            render_commands = [
                model["render_command"]
                for case in selected
                for model in case["models"].values()
            ]
            for command in render_commands:
                subprocess.run(command, shell=True, check=True, cwd=MOTIONGPT_ROOT)
            manifest["execution"] = {"status": "completed", "rendered": len(render_commands)}
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def _contact_sheet(input_path: Path, output_path: Path, samples: int, cols: int) -> None:
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    reader = imageio.get_reader(str(input_path))
    try:
        try:
            frame_count = reader.count_frames()
        except AttributeError:
            frame_count = reader.get_length()
        if not frame_count or frame_count == float("inf"):
            frames = [Image.fromarray(frame).convert("RGB") for frame in reader]
            frame_count = len(frames)
            indices = [round(i * (frame_count - 1) / max(samples - 1, 1)) for i in range(samples)]
            images = [frames[index] for index in indices]
        else:
            indices = [round(i * (frame_count - 1) / max(samples - 1, 1)) for i in range(samples)]
            images = [Image.fromarray(reader.get_data(index)).convert("RGB") for index in indices]
    finally:
        reader.close()
    width, height = images[0].size
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (height + 24)), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (index, frame) in enumerate(zip(indices, images)):
        x, y = (position % cols) * width, (position // cols) * (height + 24)
        sheet.paste(frame, (x, y))
        draw.text((x + 5, y + height + 4), f"frame {index}", fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def _root_trajectory(input_path: Path, output_path: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    motion = np.load(input_path)
    while motion.ndim > 3 and motion.shape[0] == 1:
        motion = motion[0]
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ManifestError(f"root trajectory expects [frames, joints, 3], got {motion.shape}")
    root = motion[:, 0]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.plot(root[:, 0], root[:, 2], linewidth=2)
    ax.scatter(root[0, 0], root[0, 2], c="green", label="start")
    ax.scatter(root[-1, 0], root[-1, 2], c="red", label="end")
    ax.set(xlabel="x", ylabel="z", title="Root trajectory")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _load_attribution(path: Path) -> tuple[list[str], Any]:
    import numpy as np

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        tokens = data.get("tokens") or data.get("labels")
        values = data.get("values") or data.get("attribution") or data.get("scores")
        if tokens is None or values is None:
            raise ManifestError("attribution JSON needs tokens/labels and values/scores")
        return [str(token) for token in tokens], np.asarray(values, dtype=float)
    data = np.load(path, allow_pickle=False)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = set(data.files)
        value_key = next((key for key in ("values", "attribution", "scores") if key in keys), None)
        if value_key is None:
            raise ManifestError("attribution NPZ needs values, attribution, or scores")
        values = data[value_key]
        tokens = data["tokens"].tolist() if "tokens" in keys else [str(i) for i in range(values.shape[-1])]
        return [str(token) for token in tokens], values
    values = data
    return [str(i) for i in range(values.shape[-1])], values


def _attribution_heatmap(input_path: Path, output_path: Path) -> None:
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tokens, values = _load_attribution(input_path)
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ManifestError(f"attribution heatmap expects 1D/2D values, got {values.shape}")
    if values.shape[1] != len(tokens):
        raise ManifestError("attribution token count does not match value width")
    fig_width = max(6.0, 0.45 * len(tokens))
    fig, ax = plt.subplots(figsize=(fig_width, max(2.2, 0.35 * values.shape[0])), dpi=160)
    image = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(tokens)), labels=tokens, rotation=45, ha="right")
    ax.set_ylabel("slot/frame")
    ax.set_title("Token / slot attribution")
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_postprocess(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise ManifestError(f"postprocess input does not exist: {input_path}")
    if args.kind == "contact-sheet":
        _contact_sheet(input_path, output_path, args.samples, args.cols)
    elif args.kind == "root-trajectory":
        _root_trajectory(input_path, output_path)
    else:
        _attribution_heatmap(input_path, output_path)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="select cases and build the demo manifest")
    build.add_argument("--candidates", required=True, help="candidate JSON or JSONL metadata")
    build.add_argument("--output-dir", default="ablation/blender_demo")
    build.add_argument("--render-config", default=str(MOTIONGPT_ROOT / "configs/render.yaml"))
    build.add_argument("--blender", help="explicit Blender 2.93.2 executable")
    build.add_argument("--dry-run", action="store_true", help="validate and print without writing files")
    build.add_argument("--run-blender", action="store_true", help="execute render commands after writing")

    post = subparsers.add_parser("postprocess", help="build one visualization artifact")
    post.add_argument("--kind", required=True, choices=("contact-sheet", "root-trajectory", "attribution-heatmap"))
    post.add_argument("--input", required=True)
    post.add_argument("--output", required=True)
    post.add_argument("--samples", type=int, default=12)
    post.add_argument("--cols", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_manifest(args)
            print(json.dumps(manifest, indent=2, ensure_ascii=False))
        else:
            run_postprocess(args)
    except ManifestError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
