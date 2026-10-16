import torch

from utils.plateau_control import PlateauController


def test_plateau_reduces_lr_and_stops_after_global_patience():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    controller = PlateauController(
        factor=0.5, lr_patience=2, stop_patience=4,
        min_index=10, min_lr=1e-5,
    )

    assert controller.step(1.0, 1, [optimizer]).improved
    assert not controller.step(1.1, 2, [optimizer]).reduce_lr
    decision = controller.step(1.2, 3, [optimizer])
    assert decision.reduce_lr
    assert optimizer.param_groups[0]["lr"] == 5e-4
    assert not controller.step(1.3, 9, [optimizer]).stop
    assert not controller.step(1.4, 10, [optimizer]).stop
    assert not controller.step(1.5, 11, [optimizer]).stop
    assert not controller.step(1.6, 12, [optimizer]).stop
    assert controller.step(1.7, 13, [optimizer]).stop


def test_plateau_state_round_trip():
    controller = PlateauController(mode="max", min_index=3)
    controller.best_metric = 0.7
    controller.bad_evals = 2
    state = controller.state_dict()
    restored = PlateauController()
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_plateau_uses_per_group_min_lr():
    first = torch.nn.Parameter(torch.ones(()))
    second = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([
        {"params": [first], "lr": 1e-4, "min_lr": 5e-6},
        {"params": [second], "lr": 2e-6, "min_lr": 1e-7},
    ])
    controller = PlateauController(lr_patience=1, min_lr=5e-6)
    controller.step(1.0, 0, [optimizer])
    controller.step(2.0, 1, [optimizer])
    assert optimizer.param_groups[0]["lr"] == 5e-5
    assert optimizer.param_groups[1]["lr"] == 1e-6
