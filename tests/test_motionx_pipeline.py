from pathlib import Path

import numpy as np
import pytest

from dataset.config import get_dataset_spec
from tools.finalize_sato_test_levels import choose_mixed_level
from tools.prepare_motionx_dataset import (
    build_split_name_index,
    choose_group_split,
    key_candidates,
    processed_source_candidates,
    read_splits,
    stable_motion_id,
    text_key_candidates,
)
from utils.humanml_preprocess import humanml_mean_std, process_positions, target_offsets


def test_motionx_dataset_spec_is_humanml_compatible():
    spec = get_dataset_spec("motionx")
    assert spec.root == Path("dataset/Motion-X")
    assert spec.joints == 22
    assert spec.dim_pose == 263
    assert spec.fps == 20.0
    assert spec.evaluator_name == "motionx"


def test_motionx_archive_aliases_cover_official_split_names():
    animation = key_candidates("game_motion/subset_0001/example.npy", "animation")
    perform = key_candidates("subset_0002/example.npy", "perform")
    assert "game_motion/subset_0001/example" in animation
    assert "fitness/subset_0002/example" in perform


def test_motionx_split_overlap_preserves_evaluation_sets(tmp_path):
    (tmp_path / "train.txt").write_text("train_only.npy\nshared_test.npy\nshared_val.npy\n")
    (tmp_path / "val.txt").write_text("val_only.npy\nshared_val.npy\n")
    (tmp_path / "test.txt").write_text("test_only.npy\nshared_test.npy\n")

    mapping, counts, deduplication = read_splits(tmp_path)

    assert counts == {"train": 3, "val": 2, "test": 2}
    assert mapping["shared_test"] == "test"
    assert mapping["shared_val"] == "val"
    assert deduplication["cross_split_paths"] == 2
    assert deduplication["dropped_unique_paths_from_lower_priority"] == {"train": 2}
    assert deduplication["effective_counts"] == {"train": 1, "val": 2, "test": 2}


def test_processed_motion_name_recovers_official_source_clip():
    candidates = processed_source_candidates("music/Play_Ruan_11_clip1.npy", "music")
    assert candidates[0] == "playruanclip11"


def test_unmatched_processed_group_gets_deterministic_split():
    index = build_split_name_index({"music/subset_0001/Play_Ruan_clip_11": "train"})
    official = choose_group_split("music/Play_Ruan_11_clip1.npy", "music", index)
    first = choose_group_split("music/Unknown_7_clip1.npy", "music", index)
    second = choose_group_split("music/Unknown_7_clip2.npy", "music", index)
    assert official[:2] == ("train", "official")
    assert first[:3] == second[:3]


def test_motion_identity_and_text_keys_preserve_case():
    upper = text_key_candidates("idea400/Thumbs_Up_clip1.txt", "idea400")[0]
    lower = text_key_candidates("idea400/Thumbs_up_clip1.txt", "idea400")[0]
    assert upper != lower
    assert stable_motion_id(upper) != stable_motion_id(lower)


def test_humanml_statistics_use_groupwise_scaling():
    rng = np.random.default_rng(123)
    motions = [rng.normal(size=(50, 263)).astype(np.float32) for _ in range(3)]
    mean, std = humanml_mean_std(motions)
    assert mean.shape == (263,)
    assert std.shape == (263,)
    for start, end in ((1, 3), (4, 67), (67, 193), (193, 259), (259, 263)):
        assert np.allclose(std[start:end], std[start])
    assert np.all(std > 0)


def test_mixed_level_selection_is_deterministic_and_unbalanced():
    first = [choose_mixed_level(f"mx_{index}", (0.46, 0.27, 0.27), 123) for index in range(1000)]
    second = [choose_mixed_level(f"mx_{index}", (0.46, 0.27, 0.27), 123) for index in range(1000)]
    assert first == second
    assert first.count("P1") > first.count("P2")
    assert first.count("P1") > first.count("P3")


def test_body_joint_processing_produces_263d_features():
    path = Path("dataset/HumanML3D/new_joints/000021.npy")
    if not path.is_file():
        pytest.skip("HumanML3D reference joints are unavailable")
    joints = np.load(path)[:80]
    features = process_positions(joints, target_offsets(joints))
    assert features.shape == (79, 263)
    assert np.isfinite(features).all()
