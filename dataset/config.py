"""Shared dataset metadata for T2M-GPT and SATO++ pipelines."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    root: Path
    joints: int
    dim_pose: int
    fps: float
    min_motion_len: int
    evaluator_name: str
    representation: str = "humanml"
    variable_length: bool = False
    motion_dir_name: str = "new_joint_vecs"
    evaluator_drop_tail: int = 4
    evaluator_bias_slices: tuple[tuple[int, int], ...] = ((0, 4), (-4, 0))
    velocity_slices: tuple[tuple[int, int], ...] = ()

    @property
    def motion_dir(self) -> Path:
        return self.root / self.motion_dir_name

    @property
    def text_dir(self) -> Path:
        return self.root / "texts"

    @property
    def meta_dir(self) -> Path:
        return self.root


DATASETS = {
    "t2m": DatasetSpec(
        name="t2m",
        root=Path("dataset/HumanML3D"),
        joints=22,
        dim_pose=263,
        fps=20.0,
        min_motion_len=40,
        evaluator_name="t2m",
    ),
    "kit": DatasetSpec(
        name="kit",
        root=Path("dataset/KIT-ML"),
        joints=21,
        dim_pose=251,
        fps=12.5,
        min_motion_len=24,
        evaluator_name="kit",
    ),
    "motionx": DatasetSpec(
        name="motionx",
        root=Path("dataset/Motion-X"),
        joints=22,
        dim_pose=263,
        fps=20.0,
        min_motion_len=40,
        evaluator_name="motionx",
    ),
    "motionx322": DatasetSpec(
        name="motionx322",
        root=Path("dataset/Motion-X-322"),
        joints=55,
        dim_pose=322,
        fps=30.0,
        min_motion_len=1,
        evaluator_name="motionx322",
        representation="smplx322",
        variable_length=True,
        evaluator_drop_tail=0,
        evaluator_bias_slices=((0, 3), (309, 312)),
        velocity_slices=((0, 209), (309, 312)),
    ),
    "motionx322_75": DatasetSpec(
        name="motionx322_75",
        root=Path("dataset/Motion-X-322-MaxToken75"),
        joints=55,
        dim_pose=322,
        fps=30.0,
        min_motion_len=1,
        evaluator_name="motionx322",
        representation="smplx322",
        variable_length=True,
        evaluator_drop_tail=0,
        evaluator_bias_slices=((0, 3), (309, 312)),
        velocity_slices=((0, 209), (309, 312)),
    ),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported dataset {name!r}; expected one of {sorted(DATASETS)}"
        ) from exc


def evaluator_opt_path(name: str) -> str:
    evaluator_name = get_dataset_spec(name).evaluator_name
    return f"checkpoints/{evaluator_name}/Comp_v6_KLD005/opt.txt"


def dataset_mean_std_paths(name: str) -> tuple[Path, Path]:
    spec = get_dataset_spec(name)
    if name == "motionx" or spec.representation == "smplx322":
        return spec.root / "Mean.npy", spec.root / "Std.npy"
    meta = Path(
        f"checkpoints/{name}/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    )
    return meta / "mean.npy", meta / "std.npy"


def evaluator_mean_std_paths(name: str) -> tuple[Path, Path]:
    evaluator_name = get_dataset_spec(name).evaluator_name
    if evaluator_name in {"motionx", "motionx322"}:
        meta = Path(
            f"checkpoints/{evaluator_name}/Decomp_SP001_SM001_H512/meta"
        )
        return meta / "mean.npy", meta / "std.npy"
    meta = Path(
        f"checkpoints/{evaluator_name}/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    )
    return meta / "mean.npy", meta / "std.npy"


def dataset_metadata(name: str) -> dict:
    spec = get_dataset_spec(name)
    path = spec.root / "dataset_meta.json"
    if not path.is_file():
        return {}
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def dataset_max_motion_length(name: str) -> int:
    metadata = dataset_metadata(name)
    if "max_frames" in metadata:
        return int(metadata["max_frames"])
    return 196


def dataset_max_token_length(name: str, unit_length: int) -> int:
    max_frames = dataset_max_motion_length(name)
    return (max_frames + unit_length - 1) // unit_length + 1


def configure_evaluator_source(opt, source_name: str):
    """Attach source/target normalization arrays for cross-dataset evaluation."""
    source_mean_path, source_std_path = dataset_mean_std_paths(source_name)
    target_mean_path, target_std_path = evaluator_mean_std_paths(source_name)
    opt.source_dataset_name = source_name
    opt.source_mean = np.load(source_mean_path)
    opt.source_std = np.load(source_std_path)
    opt.evaluator_mean = np.load(target_mean_path)
    opt.evaluator_std = np.load(target_std_path)
    return opt
