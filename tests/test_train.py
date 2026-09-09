import json

import pytest
import torch

from pinnlab.train import TrainConfig, load_model, train


@pytest.mark.parametrize("hard_ic", [True, False])
def test_tiny_training_checkpoint_replay(tmp_path, hard_ic, monkeypatch):
    import pinnlab.train as training_module

    exact_initial_condition = training_module.taylor_green_torch
    reference_calls = []

    def initial_condition_only(coords, nu, mode):
        assert torch.count_nonzero(coords[:, 2]) == 0, "future labels entered training"
        reference_calls.append(len(coords))
        return exact_initial_condition(coords, nu, mode)

    monkeypatch.setattr(training_module, "taylor_green_torch", initial_condition_only)
    config = TrainConfig(
        width=8, depth=1, batch_size=12, adam_steps=2, lbfgs_steps=2, hard_ic=hard_ic
    )
    metadata = train(config, tmp_path)
    assert reference_calls
    assert metadata["training_seconds"] > 0
    assert metadata["adam_updates"] == 2
    assert metadata["lbfgs_closure_evaluations"] >= 1
    assert (
        (tmp_path / "history.csv").read_text().startswith("step,phase,loss,momentum,divergence,ic")
    )
    assert json.loads((tmp_path / "training.json").read_text())["config"]["seed"] == 42
    model, saved_config = load_model(tmp_path / "checkpoint.pt")
    reloaded, _ = load_model(tmp_path / "checkpoint.pt")
    assert saved_config["hard_ic"] == hard_ic
    coords = torch.randn(5, 3, dtype=torch.float64)
    torch.testing.assert_close(model(coords), reloaded(coords), atol=0, rtol=0)
    assert torch.isfinite(model(coords)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nu": 0},
        {"tmax": -1},
        {"width": 0},
        {"depth": 0},
        {"mode": 1.5},
        {"dtype": "float16"},
        {"adam_steps": -1},
        {"adam_steps": 0, "lbfgs_steps": 0},
        {"batch_size": 0},
        {"learning_rate": float("nan")},
        {"threads": 0},
        {"seed": -1},
    ],
)
def test_invalid_training_config(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)
