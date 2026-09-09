from pinnlab.cli import main


def test_train_rejects_overwrite(tmp_path):
    (tmp_path / "checkpoint.pt").write_bytes(b"preserve me")
    assert main(["train", "--output", str(tmp_path)]) == 2
    assert (tmp_path / "checkpoint.pt").read_bytes() == b"preserve me"


def test_invalid_run_is_actionable(tmp_path):
    assert main(["demo", "--run", str(tmp_path), "--no-open"]) == 2


def test_bundled_demo_recomputes_checkpoint(tmp_path, monkeypatch):
    import hashlib
    import json
    from pathlib import Path

    import numpy as np
    import pytest
    import pinnlab.visualize as visualization

    source = Path(__file__).resolve().parents[1] / "examples/taylor-green"
    recorded = json.loads((source / "metrics.json").read_text())
    # Actual inference, exact reference, Fourier integration and audits run.
    # Rendering itself is covered separately without duplicating GIF exports.
    monkeypatch.setattr(visualization, "render_report", lambda path: [])
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "fresh-demo"
    assert main(["demo", "--run", str(source), "--output", str(output), "--no-open"]) == 0
    fresh = json.loads((output / "metrics.json").read_text())
    assert fresh["checkpoint_sha256"] == recorded["checkpoint_sha256"]
    assert (
        fresh["fields_sha256"] == hashlib.sha256((output / "fields.npz").read_bytes()).hexdigest()
    )
    for key in ("pinn_velocity_relative_l2", "pinn_pressure_relative_l2"):
        assert fresh["per_time"][-1][key] == pytest.approx(recorded["per_time"][-1][key], rel=1e-5)
    with np.load(output / "fields.npz", allow_pickle=False) as fields:
        assert fields["pinn_uv"].shape == (21, 48, 48, 2)
        assert not np.array_equal(fields["pinn_uv"], fields["exact_uv"])
    # Existing output must not be silently overwritten by another replay.
    assert main(["demo", "--run", str(source), "--output", str(output), "--no-open"]) == 2
