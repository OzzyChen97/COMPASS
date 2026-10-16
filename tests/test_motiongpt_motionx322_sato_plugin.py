import json
from pathlib import Path

import numpy as np
import torch
import yaml

from MotionGPT.mGPT.data.MotionX322 import MotionX322TextTokenDataset
from MotionGPT.mGPT.data.utils import humanml3d_collate


ROOT = Path(__file__).parents[1]


def _load_sato_helpers():
    import importlib.util
    import sys

    motiongpt_root = ROOT / "MotionGPT"
    sys.path.insert(0, str(motiongpt_root))
    try:
        path = motiongpt_root / "mGPT" / "models" / "mgpt_sato_pp.py"
        spec = importlib.util.spec_from_file_location("motiongpt_sato_helpers", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_grad_input_uses_early_teacher_forced_motion_likelihood():
    module = _load_sato_helpers()
    embeddings = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    logits = torch.stack(
        [embeddings[..., 0], embeddings[..., 1]], dim=-1)
    forward = {
        "outputs": type("Output", (), {"logits": logits})(),
        "labels": torch.tensor([[0, 1]]),
        "input_embeddings": embeddings,
        "attention_mask": torch.tensor([[True, True]]),
    }
    alpha = module._grad_input_attribution(
        forward, steps=1, tau=1.0, create_graph=True)
    assert alpha.shape == (1, 2)
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(1))
    # K=1 makes only the first target likelihood contribute, so its source
    # position receives more attribution than the untouched second position.
    assert alpha[0, 0] > alpha[0, 1]
    alpha[0, 0].backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_teacher_distillation_is_zero_for_identical_logits():
    module = _load_sato_helpers()
    logits = torch.randn(2, 3, 7)
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])
    loss = module._teacher_distillation(logits, logits, labels, "kl")
    assert abs(float(loss)) < 1.0e-6


def test_var_cov_uses_full_ddp_batch_when_local_batch_is_one(monkeypatch):
    module = _load_sato_helpers()
    local_orig = torch.tensor([[0.0, 1.0]], requires_grad=True)
    local_pair = torch.tensor([[1.0, 2.0]], requires_grad=True)
    remote_orig = torch.tensor([[2.0, 3.0]])
    remote_pair = torch.tensor([[3.0, 4.0]])
    gathered_features = iter(
        [(local_orig, remote_orig), (local_pair, remote_pair)]
    )
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        module.dist_nn, "all_gather", lambda tensor: next(gathered_features)
    )

    def gather_mask(outputs, value):
        outputs[0].copy_(value)
        outputs[1].fill_(True)

    monkeypatch.setattr(module.dist, "all_gather", gather_mask)
    views, preserve_count = module._gather_preserve_views(
        local_orig, local_pair, torch.tensor([True])
    )
    assert preserve_count == 2
    assert views.shape == (4, 2)
    loss_var = module._loss_variance(views, threshold=2.0)
    loss_cov = module._loss_covariance(views)
    assert float(loss_var) > 0
    assert float(loss_cov) > 0
    (loss_var + loss_cov).backward()
    assert local_orig.grad is not None and torch.isfinite(local_orig.grad).all()
    assert local_pair.grad is not None and torch.isfinite(local_pair.grad).all()


def _relation_mean_worker(rank, world_size, init_path, result_dir):
    module = _load_sato_helpers()
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        alter_source = torch.tensor([8.0], requires_grad=True)
        alter_values = alter_source if rank == 0 else alter_source[:0]

        preserve_source = torch.tensor([2.0, 4.0, 6.0], requires_grad=True)
        if rank == 0:
            preserve_values = preserve_source[:1]
        elif rank == 1:
            preserve_values = preserve_source[1:]
        else:
            preserve_values = preserve_source[:0]

        alter_mean = module._global_mean(alter_values, distributed=True)
        preserve_mean = module._global_mean(preserve_values, distributed=True)
        (alter_mean + preserve_mean).backward()

        linear = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            linear.weight.fill_(1.0)
        ddp = torch.nn.parallel.DistributedDataParallel(linear)
        ddp_input = (
            torch.tensor([[8.0]])
            if rank == 0
            else torch.empty((0, 1))
        )
        ddp_mean = module._global_mean(
            ddp(ddp_input).flatten(), distributed=True
        )
        ddp_mean.backward()
        Path(result_dir, f"rank{rank}.json").write_text(json.dumps({
            "alter_mean": float(alter_mean.detach()),
            "preserve_mean": float(preserve_mean.detach()),
            "alter_grad_finite": bool(torch.isfinite(alter_source.grad).all()),
            "preserve_grad_finite": bool(torch.isfinite(preserve_source.grad).all()),
            "ddp_mean": float(ddp_mean.detach()),
            "ddp_weight_grad": float(ddp.module.weight.grad),
        }))
    finally:
        torch.distributed.destroy_process_group()


def test_relation_losses_use_global_valid_count_across_four_ranks(tmp_path):
    world_size = 4
    init_path = tmp_path / "relation_mean_init"
    result_dir = tmp_path / "relation_mean_results"
    result_dir.mkdir()
    torch.multiprocessing.start_processes(
        _relation_mean_worker,
        args=(world_size, str(init_path), str(result_dir)),
        nprocs=world_size,
        join=True,
        start_method="spawn",
    )
    results = [
        json.loads((result_dir / f"rank{rank}.json").read_text())
        for rank in range(world_size)
    ]
    assert [row["alter_mean"] for row in results] == [8.0] * world_size
    assert [row["preserve_mean"] for row in results] == [4.0] * world_size
    assert [row["ddp_mean"] for row in results] == [8.0] * world_size
    assert [row["ddp_weight_grad"] for row in results] == [8.0] * world_size
    assert all(row["alter_grad_finite"] for row in results)
    assert all(row["preserve_grad_finite"] for row in results)


def test_relation_loss_global_mean_has_single_process_fallback(monkeypatch):
    module = _load_sato_helpers()
    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: False)
    values = torch.tensor([1.0, 3.0], requires_grad=True)
    loss = module._global_mean(values, distributed=True)
    assert float(loss) == 2.0
    loss.backward()
    assert torch.allclose(values.grad, torch.tensor([0.5, 0.5]))


def test_motionx_sato_dataset_emits_pair_contract(tmp_path):
    (tmp_path / "texts").mkdir()
    (tmp_path / "codes").mkdir()
    motion_id = "mx322_unit"
    caption = "A person walks forward."
    (tmp_path / "train.txt").write_text(motion_id + "\n", encoding="utf-8")
    (tmp_path / "texts" / f"{motion_id}.txt").write_text(
        caption + "#a/DET person/NOUN walks/VERB forward/ADV#0#0\n",
        encoding="utf-8",
    )
    np.save(tmp_path / "codes" / f"{motion_id}.npy", np.array([[1, 2, 3]]))
    task_path = tmp_path / "tasks.json"
    task_path.write_text(json.dumps({
        "Text-to-Motion": {
            "caption": {
                "class": "t2m",
                "input": ["<Caption_Placeholder>"],
                "output": ["<Motion_Placeholder>"],
            },
            "length": {
                "class": "l2m",
                "input": ["<Frame_Placeholder>"],
                "output": ["<Motion_Placeholder>"],
            },
        }
    }), encoding="utf-8")
    pair_path = tmp_path / "pairs.jsonl"
    pair_path.write_text(json.dumps({
        "motion_id": motion_id,
        "text_orig": caption,
        "text_pair": "A person walks backward.",
        "relation": "alter",
        "edit_type": "direction",
    }) + "\n", encoding="utf-8")

    dataset = MotionX322TextTokenDataset(
        data_root=tmp_path,
        split="train",
        mean=np.zeros(322),
        std=np.ones(322),
        code_path="codes",
        task_path=task_path,
        stage="lm_sato_pp",
        pair_data_paths=[pair_path],
        preserve_ratio=0.0,
    )
    assert len(dataset.tasks) == 1
    row = dataset[0]
    assert len(row) == 12
    assert row[9:] == ("A person walks backward.", "alter", "direction")
    batch = humanml3d_collate([row])
    assert batch["text_pair"] == ["A person walks backward."]
    assert batch["relation"] == ["alter"]


def test_motionx_sato_formal_config_is_a_four_gpu_baseline_plugin():
    config = yaml.safe_load((
        ROOT / "MotionGPT" / "configs" / "config_motionx322_sato_pp.yaml"
    ).read_text(encoding="utf-8"))
    assert config["DEVICE"] == [0, 1, 2, 3]
    assert config["TRAIN"]["STAGE"] == "lm_sato_pp"
    assert config["TRAIN"]["STRICT_LOAD"] is False
    assert "Instruct_MOTIONX322_OFFICIAL_FRESH_R2" in config["TRAIN"]["PRETRAINED"]
    assert config["model"]["target"].endswith("MotionGPTSATOPP")
    assert config["SATO"]["ATTRIBUTION_CREATE_GRAPH"] is True
    assert config["SATO"]["ATTRIBUTION_K"] == 16
    assert config["CHECKPOINT_SELECTION"]["EXPECTED_SAMPLES"] == 2552
    assert config["SATO"]["DISTRIBUTED_VAR_COV"] is True
    assert config["SATO"]["DISTRIBUTED_RELATION_LOSSES"] is True
    assert config["CHECKPOINT_SELECTION"]["SPLIT"] == "val"
