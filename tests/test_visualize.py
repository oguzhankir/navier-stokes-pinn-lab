"""Renderer tests use deliberately synthetic fixtures, never showcase predictions."""

import hashlib
import json
from pathlib import Path
import re

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from pinnlab.visualize import _Comparison, _STYLE, _load, render_report, show


@pytest.fixture
def synthetic_run(tmp_path: Path) -> Path:
    n = 8
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    times = np.asarray([0.0, 0.5, 1.0])
    uv = np.stack(
        [
            np.stack((-np.cos(xx) * np.sin(yy), np.sin(xx) * np.cos(yy)), axis=-1)
            * np.exp(-0.2 * time)
            for time in times
        ]
    )
    p = np.stack(
        [-0.25 * (np.cos(2 * xx) + np.cos(2 * yy)) * np.exp(-0.4 * time) for time in times]
    )
    omega = np.stack([2 * np.cos(xx) * np.cos(yy) * np.exp(-0.2 * time) for time in times])
    fields = {"x": x, "y": x, "times": times, "particles": np.zeros((len(times), 4, 2))}
    for source, scale in (("exact", 1.0), ("numerical", 1.005), ("pinn", 0.93)):
        fields[f"{source}_uv"] = scale * uv
        fields[f"{source}_p"] = scale * p
        fields[f"{source}_omega"] = scale * omega
    np.savez_compressed(tmp_path / "fields.npz", **fields)
    (tmp_path / "metrics.json").write_text(
        json.dumps(
            {
                "config": {"nu": 0.1, "tmax": 1.0, "mode": "synthetic-test-fixture-not-trained"},
                "checkpoint_sha256": "synthetic-fixture-not-a-checkpoint",
                "per_time": [],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_report_exports_headlessly_and_html_has_no_external_assets(synthetic_run: Path) -> None:
    paths = render_report(synthetic_run)
    assert {path.name for path in paths} == {
        "comparison.png",
        "diagnostics.png",
        "preview.gif",
        "comparison.html",
    }
    for path in paths:
        assert path.exists() and path.stat().st_size > 100
    assert (synthetic_run / "preview.gif").stat().st_size <= 2_000_000
    with Image.open(synthetic_run / "preview.gif") as image:
        assert image.n_frames == 3
        assert image.width >= 600
    with Image.open(synthetic_run / "comparison.png") as image:
        assert image.width > image.height
    page = (synthetic_run / "comparison.html").read_text(encoding="utf-8")
    assert 'name="pinnlab-frame-count" content="3"' in page
    assert "data:image/png;base64," in page
    assert not re.search(r"(?:src|href)\s*=\s*['\"](?:https?:)?//", page, re.IGNORECASE)
    assert not re.search(r"url\(['\"]?(?:https?:)?//", page, re.IGNORECASE)
    assert "100% relative velocity initial-condition error" in page
    assert "synthetic-fixture-not-a-checkpoint" in page


def test_optional_exports(synthetic_run: Path) -> None:
    paths = render_report(synthetic_run, html=False, gif=False)
    assert [path.name for path in paths] == ["comparison.png", "diagnostics.png"]
    assert not (synthetic_run / "comparison.html").exists()
    assert not (synthetic_run / "preview.gif").exists()


def test_comparison_keeps_fixed_scales_and_preserves_xy_grid(synthetic_run: Path) -> None:
    fields, metrics = _load(synthetic_run)
    with mpl.rc_context(_STYLE):
        comparison = _Comparison(fields, metrics)
        initial_limits = [picture.get_clim() for picture in comparison.images]
        comparison.update(2)
        assert [picture.get_clim() for picture in comparison.images] == initial_limits
        np.testing.assert_allclose(
            comparison.images[0].get_array()[:-1, :-1], fields["pinn_omega"][2]
        )
        comparison.select(quantity="Speed", reference="Fourier solver")
        vector_error = np.linalg.norm(fields["pinn_uv"][2] - fields["numerical_uv"][2], axis=-1)
        np.testing.assert_allclose(comparison.images[2].get_array()[:-1, :-1], vector_error)
        comparison.select(control=True)
        assert "Diagnostic control (not trained)" in comparison.axes[0].get_title()
        assert "100% relative velocity IC error" in comparison.note.get_text()
        np.testing.assert_array_equal(comparison.images[0].get_array(), 0)
        assert not comparison.particles.get_visible()
        comparison.figure.clear()


def test_renderer_rejects_nonfinite_predictions(synthetic_run: Path) -> None:
    fields, _ = _load(synthetic_run)
    fields["pinn_uv"][0, 0, 0, 0] = np.nan
    np.savez_compressed(synthetic_run / "fields.npz", **fields)
    with pytest.raises(ValueError, match="pinn_uv contains nonfinite"):
        render_report(synthetic_run, html=False, gif=False)


def test_renderer_rejects_missing_field(synthetic_run: Path) -> None:
    fields, _ = _load(synthetic_run)
    del fields["pinn_omega"]
    np.savez_compressed(synthetic_run / "fields.npz", **fields)
    with pytest.raises(ValueError, match="missing: pinn_omega"):
        render_report(synthetic_run, html=False, gif=False)


def test_local_viewer_controls_without_gui(
    synthetic_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: None)
    show(synthetic_run)
    figure = plt.gcf()
    slider, quantities, references, models, play, timer = figure._pinnlab_controls
    quantities.set_active(1)
    references.set_active(1)
    slider.set_val(0.5)
    assert figure.axes[0].get_title() == "Trained PINN"
    assert figure.axes[1].get_title() == "Fourier solver reference"
    assert figure.axes[2].get_title() == "Absolute vector error"
    models.set_active(1)
    assert figure.axes[0].get_title() == "Diagnostic control (not trained)"
    play._observers.process("clicked", None)
    assert play.label.get_text() == "Pause"
    timer.callbacks[0][0]()
    assert slider.val == 1.0
    timer.stop()
    plt.close(figure)


def _record_fixture_hashes(run_dir: Path) -> None:
    """Record integrity metadata for intentionally non-model synthetic bytes."""
    (run_dir / "checkpoint.pt").write_bytes(b"synthetic-test-checkpoint-not-a-trained-model")
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    for name, key in (("fields.npz", "fields_sha256"), ("checkpoint.pt", "checkpoint_sha256")):
        metrics[key] = hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")


def test_renderer_accepts_matching_provenance_hashes(synthetic_run: Path) -> None:
    _record_fixture_hashes(synthetic_run)
    fields, metrics = _load(synthetic_run)
    assert fields["pinn_uv"].shape == (3, 8, 8, 2)
    assert len(metrics["fields_sha256"]) == 64


def test_renderer_rejects_changed_fields_before_render(synthetic_run: Path) -> None:
    _record_fixture_hashes(synthetic_run)
    fields, _ = _load(synthetic_run)
    fields["pinn_uv"][0, 0, 0, 0] += 1
    np.savez_compressed(synthetic_run / "fields.npz", **fields)
    with pytest.raises(ValueError, match=r"fields.npz SHA-256 mismatch:.*pinnlab evaluate"):
        render_report(synthetic_run, html=False, gif=False)
    assert not (synthetic_run / "comparison.png").exists()


def test_renderer_rejects_retrained_checkpoint_before_render(synthetic_run: Path) -> None:
    _record_fixture_hashes(synthetic_run)
    (synthetic_run / "checkpoint.pt").write_bytes(b"new-synthetic-checkpoint-after-retraining")
    with pytest.raises(ValueError, match=r"checkpoint.pt SHA-256 mismatch:.*pinnlab evaluate"):
        render_report(synthetic_run, html=False, gif=False)
    assert not (synthetic_run / "comparison.png").exists()


def test_renderer_allows_missing_optional_legacy_hashes(synthetic_run: Path) -> None:
    metrics_path = synthetic_run / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    del metrics["checkpoint_sha256"]
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    (synthetic_run / "checkpoint.pt").write_bytes(b"legacy-synthetic-fixture")
    fields, loaded_metrics = _load(synthetic_run)
    assert len(fields["times"]) == 3
    assert "checkpoint_sha256" not in loaded_metrics


@pytest.mark.parametrize("interactive", [False, True])
def test_colorbar_text_fits_export_and_viewer_canvases(
    synthetic_run: Path, interactive: bool
) -> None:
    fields, metrics = _load(synthetic_run)
    with mpl.rc_context(_STYLE):
        figure = Figure(figsize=(13, 7), dpi=90) if interactive else None
        if figure is not None:
            FigureCanvasAgg(figure)
        comparison = _Comparison(fields, metrics, figure=figure, interactive=interactive)
        for quantity in ("Vorticity", "Pressure", "Speed"):
            comparison.select(quantity=quantity)
            # Exercise longer decimal tick text like the actual trained showcase.
            comparison.images[2].set_clim(0, 0.0016)
            comparison.figure.canvas.draw()
            renderer = comparison.figure.canvas.get_renderer()
            for colorbar in comparison.colorbars:
                text_artists = [colorbar.ax.yaxis.label, *colorbar.ax.get_yticklabels()]
                for artist in text_artists:
                    bounds = artist.get_window_extent(renderer)
                    assert bounds.x0 >= 0
                    assert bounds.x1 <= comparison.figure.bbox.x1 - 2
        comparison.figure.clear()
