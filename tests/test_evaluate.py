import numpy as np
import pytest

from pinnlab.evaluate import align_pressure, relative_l2


def test_pressure_gauge_is_per_time():
    p = np.arange(24, dtype=float).reshape(2, 3, 4)
    q = p + np.array([7.0, -900.0])[:, None, None]
    np.testing.assert_allclose(align_pressure(p), align_pressure(q), atol=1e-12)
    np.testing.assert_allclose(align_pressure(q).mean(axis=(-2, -1)), 0, atol=1e-12)


def test_velocity_error_detects_direction_not_only_speed():
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert relative_l2(-reference, reference) == pytest.approx(2.0)
    assert relative_l2(np.zeros_like(reference), reference) == pytest.approx(1.0)


def test_evaluate_checkpoint(tmp_path):
    import hashlib
    from pinnlab.train import TrainConfig, train
    from pinnlab.evaluate import evaluate

    train(
        TrainConfig(adam_steps=1, lbfgs_steps=0, batch_size=8, width=8, depth=1, threads=1),
        tmp_path,
    )
    metrics = evaluate(tmp_path, n=8, frames=3, residual_points=8, dt=0.04)
    assert metrics["audit"]["heldout_points"] == 8
    assert metrics["audit"]["heldout_initial_velocity_relative_l2"] < 1e-12
    assert metrics["zero_field_control"]["initial_velocity_relative_l2"] == 1.0
    assert len(metrics["checkpoint_sha256"]) == 64
    assert (
        metrics["fields_sha256"]
        == hashlib.sha256((tmp_path / "fields.npz").read_bytes()).hexdigest()
    )
    assert metrics["evaluation"]["threads"] == 1
    data = np.load(tmp_path / "fields.npz", allow_pickle=False)
    assert data["pinn_uv"].shape == (3, 8, 8, 2)
    assert data["particles"].shape == (3, 48, 2)
    np.testing.assert_allclose(data["pinn_p"].mean(axis=(-2, -1)), 0, atol=1e-12)


def test_invalid_metrics_preserve_existing_artifacts(tmp_path, monkeypatch):
    import pinnlab.evaluate as evaluation
    from pinnlab.train import TrainConfig, train

    train(TrainConfig(adam_steps=1, lbfgs_steps=0, batch_size=8, width=8, depth=1), tmp_path)
    evaluation.evaluate(tmp_path, n=8, frames=2, residual_points=4)
    original = {name: (tmp_path / name).read_bytes() for name in ("fields.npz", "metrics.json")}
    monkeypatch.setattr(
        evaluation, "_residual_audit", lambda *args: {"heldout_momentum_rms": float("nan")}
    )
    with pytest.raises(ValueError, match="Out of range"):
        evaluation.evaluate(tmp_path, n=8, frames=2, residual_points=4)
    for name, contents in original.items():
        assert (tmp_path / name).read_bytes() == contents


@pytest.mark.parametrize("kwargs", [{"n": 7}, {"frames": 1}, {"residual_points": 0}])
def test_evaluation_invalid_configuration(tmp_path, kwargs):
    from pinnlab.evaluate import evaluate

    with pytest.raises(ValueError):
        evaluate(tmp_path, **kwargs)
