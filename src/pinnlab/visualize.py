"""Scientific field comparison, offline animation export, and local exploration.

All learned-field pixels come from ``fields.npz``; this module never evaluates an
analytic field as a substitute for a checkpoint prediction. Coordinates, time,
velocity, density-normalized pressure, and vorticity are nondimensional.
"""

from __future__ import annotations

import hashlib
import html as html_lib
import io
import json
from pathlib import Path
import re
from typing import Any

import matplotlib as mpl
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
import numpy as np


_BACKGROUND = "#101827"
_FOREGROUND = "#e9eef7"
_MUTED = "#aab8cf"
_ACCENTS = {"pinn": "#56d6c9", "exact": "#f4c06d", "numerical": "#b0a8ff"}
_REFERENCES = {"Exact": "exact", "Fourier solver": "numerical"}
_QUANTITIES = ("Vorticity", "Speed", "Pressure")
_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "text.color": _FOREGROUND,
    "axes.labelcolor": _MUTED,
    "axes.edgecolor": "#49566c",
    "axes.facecolor": _BACKGROUND,
    "figure.facecolor": _BACKGROUND,
    "savefig.facecolor": _BACKGROUND,
    "xtick.color": _MUTED,
    "ytick.color": _MUTED,
    "grid.color": "#49566c",
}


def _load(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load non-pickled fields, rejecting stale or modified evaluation artifacts."""
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    field_bytes = (run_dir / "fields.npz").read_bytes()
    expected_fields_hash = metrics.get("fields_sha256")
    if (
        expected_fields_hash is not None
        and hashlib.sha256(field_bytes).hexdigest() != expected_fields_hash
    ):
        raise ValueError(
            "fields.npz SHA-256 mismatch: evaluation artifacts are stale or modified. "
            "Run pinnlab evaluate before rendering."
        )
    checkpoint = run_dir / "checkpoint.pt"
    expected_checkpoint_hash = metrics.get("checkpoint_sha256")
    if (
        checkpoint.exists()
        and expected_checkpoint_hash is not None
        and hashlib.sha256(checkpoint.read_bytes()).hexdigest() != expected_checkpoint_hash
    ):
        raise ValueError(
            "checkpoint.pt SHA-256 mismatch: fields were evaluated from a different "
            "checkpoint. Run pinnlab evaluate before rendering."
        )
    # Parse the exact bytes checked above, not a second potentially changed read.
    with np.load(io.BytesIO(field_bytes), allow_pickle=False) as archive:
        fields = {name: archive[name] for name in archive.files}
    required = {"x", "y", "times"}
    required.update(
        f"{source}_{quantity}"
        for source in ("pinn", "exact", "numerical")
        for quantity in ("uv", "p", "omega")
    )
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"fields.npz is missing: {', '.join(sorted(missing))}")
    for name in ("x", "y", "times"):
        if fields[name].ndim != 1 or len(fields[name]) == 0:
            raise ValueError(f"{name} must be a nonempty 1D array")
        if not np.isfinite(fields[name]).all():
            raise ValueError(f"{name} contains nonfinite values")
        if len(fields[name]) > 1 and not np.all(np.diff(fields[name]) > 0):
            raise ValueError(f"{name} must be strictly increasing")
    if min(len(fields["x"]), len(fields["y"])) < 2:
        raise ValueError("At least two spatial grid points per axis are required")
    # Evaluation arrays use xy indexing: field[t, y_index, x_index].
    shape = (len(fields["times"]), len(fields["y"]), len(fields["x"]))
    for name in sorted(required - {"x", "y", "times"}):
        expected = shape + ((2,) if name.endswith("_uv") else ())
        if fields[name].shape != expected:
            raise ValueError(f"{name} has shape {fields[name].shape}; expected {expected}")
        if not np.isfinite(fields[name]).all():
            raise ValueError(
                f"{name} contains nonfinite values; inspect evaluation before rendering"
            )
    if "particles" in fields:
        particles = fields["particles"]
        if particles.ndim != 3 or particles.shape[0] != shape[0] or particles.shape[-1] != 2:
            raise ValueError("particles must have shape (number of times, number of particles, 2)")
        if not np.isfinite(particles).all():
            raise ValueError("particles contains nonfinite values")
    return fields, metrics


def _scalar(fields: dict[str, np.ndarray], source: str, quantity: str) -> np.ndarray:
    if quantity == "Speed":
        return np.linalg.norm(fields[f"{source}_uv"], axis=-1)
    return fields[f"{source}_{'p' if quantity == 'Pressure' else 'omega'}"]


def _relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm((prediction - reference).ravel()))
    denominator = float(np.linalg.norm(reference.ravel()))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


class _Comparison:
    """Three synchronized maps with time-independent normalization per selection."""

    def __init__(
        self,
        fields: dict[str, np.ndarray],
        metrics: dict[str, Any],
        figure: Figure | None = None,
        interactive: bool = False,
    ):
        self.fields = fields
        self.metrics = metrics
        self.figure = figure if figure is not None else Figure(figsize=(12, 5.3), dpi=90)
        if figure is None:
            FigureCanvasAgg(self.figure)
        self.index = 0
        self.quantity = "Vorticity"
        self.reference = "Exact"
        self.control = False
        self.figure.set_facecolor(_BACKGROUND)
        bottom = 0.40 if interactive else 0.29
        # Reserve room for long decimal tick labels and the rotated error label;
        # tight-bbox export alone would not fix GIF, HTML, or GUI canvas clipping.
        grid = self.figure.add_gridspec(
            1, 3, left=0.055, right=0.91, bottom=bottom, top=0.79, wspace=0.25
        )
        self.axes = [self.figure.add_subplot(grid[0, i]) for i in range(3)]
        x, y = fields["x"], fields["y"]
        dx, dy = x[1] - x[0], y[1] - y[0]
        # A periodic ghost row/column fills the upper half-cell without moving
        # sample centers. Explicit limits prevent tracer scatters changing axes.
        extent = (x[0] - dx / 2, x[-1] + 1.5 * dx, y[0] - dy / 2, y[-1] + 1.5 * dy)
        self.images = []
        self.colorbars = []
        for panel, axis in enumerate(self.axes):
            picture = axis.imshow(
                np.zeros((len(y) + 1, len(x) + 1)),
                origin="lower",
                extent=extent,
                interpolation="nearest",
                rasterized=True,
            )
            axis.set_xlabel("x (nondim.)")
            if panel == 0:
                axis.set_ylabel("y (nondim.)")
            axis.set_xticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
            axis.set_yticks([-np.pi, 0, np.pi], ["−π", "0", "π"])
            axis.set_xlim(float(x[0]), float(x[-1] + dx))
            axis.set_ylim(float(y[0]), float(y[-1] + dy))
            axis.set_autoscale_on(False)
            colorbar = self.figure.colorbar(picture, ax=axis, fraction=0.045, pad=0.035)
            colorbar.ax.tick_params(labelsize=8)
            self.images.append(picture)
            self.colorbars.append(colorbar)
        self.particles = self.axes[0].scatter(
            [], [], s=8, c="#ffffff", alpha=0.8, edgecolors="#243042", linewidths=0.2
        )
        self.figure.text(
            0.055,
            0.935,
            "NAVIER–STOKES / PINN FLOW LAB",
            fontsize=17,
            fontweight="bold",
            color=_FOREGROUND,
        )
        self.subtitle = self.figure.text(0.055, 0.875, "", fontsize=10, color=_MUTED)
        self.status = self.figure.text(
            0.055, 0.30 if interactive else 0.15, "", fontsize=10, color=_FOREGROUND
        )
        self.note = self.figure.text(
            0.055, 0.245 if interactive else 0.095, "", fontsize=8.5, color=_MUTED
        )
        digest = str(metrics.get("checkpoint_sha256", "not recorded"))[:12]
        self.figure.text(
            0.055,
            0.205 if interactive else 0.04,
            f"Checkpoint SHA-256: {digest}  ·  Scales fixed across saved times"
            "  ·  All units nondimensional",
            fontsize=8,
            color=_MUTED,
        )
        self.select()

    def select(
        self, quantity: str | None = None, reference: str | None = None, control: bool | None = None
    ) -> None:
        if quantity is not None:
            if quantity not in _QUANTITIES:
                raise ValueError(f"Unknown quantity: {quantity}")
            self.quantity = quantity
        if reference is not None:
            if reference not in _REFERENCES:
                raise ValueError(f"Unknown reference: {reference}")
            self.reference = reference
        if control is not None:
            self.control = control
        source = _REFERENCES[self.reference]
        predicted = _scalar(self.fields, "pinn", self.quantity)
        self.predicted = np.zeros_like(predicted) if self.control else predicted
        self.expected = _scalar(self.fields, source, self.quantity)
        if self.quantity == "Speed":
            uv = np.zeros_like(self.fields["pinn_uv"]) if self.control else self.fields["pinn_uv"]
            self.error = np.linalg.norm(uv - self.fields[f"{source}_uv"], axis=-1)
        else:
            self.error = np.abs(self.predicted - self.expected)
        field_max = max(
            float(np.max(np.abs(_scalar(self.fields, key, self.quantity))))
            for key in ("pinn", "exact", "numerical")
        )
        field_max = max(field_max, np.finfo(float).eps)
        limits = (0, field_max) if self.quantity == "Speed" else (-field_max, field_max)
        cmap = "viridis" if self.quantity == "Speed" else "coolwarm"
        for picture in self.images[:2]:
            picture.set_norm(Normalize(*limits))
            picture.set_cmap(cmap)
        self.images[2].set_norm(Normalize(0, max(float(self.error.max()), np.finfo(float).eps)))
        self.images[2].set_cmap("magma")
        field_label = {"Vorticity": "ω", "Speed": "|u|", "Pressure": "p / ρ"}[self.quantity]
        self.colorbars[0].set_label(field_label, size=8)
        self.colorbars[1].set_label(field_label, size=8)
        self.colorbars[2].set_label(
            "|u − u_ref|"
            if self.quantity == "Speed"
            else f"Absolute {self.quantity.lower()} error",
            size=8,
        )
        self.axes[0].set_title(
            "Diagnostic control (not trained)" if self.control else "Trained PINN",
            fontsize=11,
            color=_ACCENTS["pinn"],
            pad=12,
        )
        self.axes[1].set_title(
            f"{self.reference} reference", fontsize=11, color=_ACCENTS[source], pad=12
        )
        self.axes[2].set_title(
            "Absolute vector error" if self.quantity == "Speed" else "Absolute error",
            fontsize=11,
            pad=12,
        )
        self.particles.set_visible("particles" in self.fields and not self.control)
        self.update(self.index)

    def update(self, index: int) -> list[Any]:
        self.index = int(index)
        for picture, values in zip(self.images, (self.predicted, self.expected, self.error)):
            picture.set_data(np.pad(values[self.index], ((0, 1), (0, 1)), mode="wrap"))
        if "particles" in self.fields and not self.control:
            self.particles.set_offsets(self.fields["particles"][self.index])
        source = _REFERENCES[self.reference]
        predicted_uv = (
            np.zeros_like(self.fields["pinn_uv"][self.index])
            if self.control
            else self.fields["pinn_uv"][self.index]
        )
        reference_uv = self.fields[f"{source}_uv"][self.index]
        relative = _relative_l2(predicted_uv, reference_uv)
        energy = 0.5 * np.mean(np.sum(predicted_uv**2, axis=-1))
        time = float(self.fields["times"][self.index])
        viscosity = self.metrics.get("config", {}).get("nu", "not recorded")
        self.subtitle.set_text(
            f"2D periodic Taylor–Green · {self.quantity.lower()} · t = {time:.3f} · ν = {viscosity}"
        )
        self.status.set_text(
            f"Velocity relative L² error: {relative:.2%}   |   "
            f"Mean kinetic energy: {energy:.5f}   |   "
            f"Frame {self.index + 1}/{len(self.fields['times'])}"
        )
        if self.control:
            self.note.set_text(
                "Zero velocity + constant pressure: zero PDE residual, but "
                "100% relative velocity IC error for this nonzero initial flow."
            )
        else:
            self.note.set_text(
                "PINN predictions from the saved checkpoint. "
                + (
                    "White tracers follow the learned velocity. "
                    if "particles" in self.fields
                    else ""
                )
                + "Pressure gauge aligned at each time."
            )
        return [*self.images, self.particles, self.subtitle, self.status, self.note]


def _diagnostics(fields: dict[str, np.ndarray], metrics: dict[str, Any], output: Path) -> None:
    figure = Figure(figsize=(11.5, 5), dpi=110)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    figure.subplots_adjust(left=0.075, right=0.975, bottom=0.19, top=0.67, wspace=0.38)
    times = fields["times"]
    rows = metrics.get("per_time", [])

    # Metrics come from evaluation when present; tiny fixtures can use fields.
    def values(key: str, fallback: np.ndarray) -> np.ndarray:
        if len(rows) == len(times) and all(key in row for row in rows):
            return np.asarray([row[key] for row in rows], dtype=float)
        return fallback

    for source, label, style in (
        ("pinn", "Trained PINN", "-"),
        ("exact", "Exact", "--"),
        ("numerical", "Fourier solver", ":"),
    ):
        energy = 0.5 * np.mean(np.sum(fields[f"{source}_uv"] ** 2, axis=-1), axis=(1, 2))
        axes[0].plot(
            times,
            values(f"{source}_energy", energy),
            style,
            label=label,
            color=_ACCENTS[source],
            linewidth=2,
        )
    for source, label, style in (
        ("pinn", "Trained PINN", "-"),
        ("numerical", "Fourier solver", ":"),
    ):
        for axis, component, metric in ((axes[1], "uv", "velocity"), (axes[2], "p", "pressure")):
            errors = np.asarray(
                [
                    _relative_l2(pred, ref)
                    for pred, ref in zip(
                        fields[f"{source}_{component}"], fields[f"exact_{component}"]
                    )
                ]
            )
            # Linear error axes retain exact zero without inventing a log floor.
            axis.plot(
                times,
                values(f"{source}_{metric}_relative_l2", errors) * 100,
                style,
                label=label,
                color=_ACCENTS[source],
                linewidth=2,
            )
    for axis, title, ylabel in zip(
        axes,
        ("Energy decay", "Velocity error vs exact", "Pressure error vs exact"),
        ("Mean kinetic energy", "Relative L² error (%)", "Relative L² error (%)"),
    ):
        axis.set_title(title, fontsize=11, pad=12)
        axis.set_xlabel("Time (nondim.)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.3)
        axis.set_ylim(bottom=0)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    figure.text(
        0.075, 0.905, "PHYSICS CHECK / INDEPENDENT EVALUATION", fontsize=16, fontweight="bold"
    )
    figure.text(
        0.075,
        0.825,
        "Per-time errors on the saved evaluation grid; pressure is zero-mean at each time.",
        fontsize=9,
        color=_MUTED,
    )
    audit = metrics.get("audit", {})
    if "heldout_divergence_rms" in audit:
        figure.text(
            0.075,
            0.77,
            f"Independent space-time audit: divergence RMS = {audit['heldout_divergence_rms']:.3e}"
            "  (finite sampled check, not a proof)",
            fontsize=9,
            color=_MUTED,
        )
    figure.text(
        0.075,
        0.055,
        "Viscous flow dissipates energy. This 2D benchmark is not a turbulence or singularity test.",
        fontsize=9,
        color=_MUTED,
    )
    figure.savefig(output)


def _offline_html(animation: FuncAnimation, frame_count: int, metrics: dict[str, Any]) -> str:
    """Keep Matplotlib's embedded frames and controls, removing its icon CDN."""
    with mpl.rc_context({"animation.embed_limit": 120.0}):
        content = animation.to_jshtml(fps=8, embed_frames=True, default_mode="loop")
    content = re.sub(r"<link\b[^>]*>", "", content, flags=re.IGNORECASE)
    icons = {
        "fa-minus": "−",
        "fa-fast-backward": "|◀",
        "fa-step-backward": "◀|",
        "fa-play fa-flip-horizontal": "◀",
        "fa-pause": "Ⅱ",
        "fa-play": "▶",
        "fa-step-forward": "|▶",
        "fa-fast-forward": "▶|",
        "fa-plus": "+",
    }
    for name, symbol in icons.items():
        content = content.replace(f'<i class="fa {name}"></i>', symbol)
    content = re.sub(
        r'<img id="(_anim_img[^"]+)"',
        r'<img alt="PINN, exact vorticity, and absolute error across time" id="\1"',
        content,
    )
    digest = html_lib.escape(str(metrics.get("checkpoint_sha256", "not recorded")))
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<meta name="pinnlab-frame-count" content="{frame_count}">'
        "<title>Navier–Stokes PINN Lab — checkpoint replay</title>"
        "<style>body{margin:0 auto;padding:24px;max-width:1200px;background:#101827;"
        "color:#e9eef7;font-family:system-ui,sans-serif}h1{font-size:20px;font-weight:600}"
        "p{color:#aab8cf;font-size:14px;line-height:1.5}.animation{width:100%}"
        ".animation img{max-width:100%;height:auto}.anim-controls{max-width:100%!important}"
        ".anim-buttons{display:flex;flex-wrap:wrap;justify-content:center;gap:4px}"
        "button{font:inherit;min-width:36px;min-height:36px;background:#253149;color:#e9eef7;"
        "border:1px solid #66748b;border-radius:4px;cursor:pointer}"
        "button:focus-visible,input:focus-visible{outline:2px solid #56d6c9}"
        "code{overflow-wrap:anywhere}</style></head><body>"
        "<h1>Trained PINN / independent exact reference / absolute error</h1>"
        "<p>Offline checkpoint replay: all frames are embedded. Vorticity scales remain fixed "
        "through time. Use the local viewer to change quantity, reference, or diagnostic control.</p>"
        + content
        + f"<p>Checkpoint SHA-256: <code>{digest}</code></p>"
        "<p>The zero-field diagnostic in the local viewer is not trained: it has zero PDE residual "
        "but 100% relative velocity initial-condition error for this nonzero flow. "
        "No simulation here proves regularity or singularity formation.</p></body></html>\n"
    )


def render_report(run_dir: Path, html: bool = True, gif: bool = True) -> list[Path]:
    """Export maps, diagnostics, and optional offline replay from evaluated fields.

    ``comparison.png`` uses the final saved time. The GIF samples at most 24
    frames; HTML preserves every saved frame. Export requires no GUI or network.
    """
    run_dir = Path(run_dir)
    fields, metrics = _load(run_dir)
    outputs = [run_dir / "comparison.png", run_dir / "diagnostics.png"]
    with mpl.rc_context(_STYLE):
        comparison = _Comparison(fields, metrics)
        comparison.update(len(fields["times"]) - 1)
        comparison.figure.savefig(outputs[0], dpi=130)
        _diagnostics(fields, metrics, outputs[1])
        if gif:
            gif_path = run_dir / "preview.gif"
            frame_ids = np.unique(
                np.linspace(0, len(fields["times"]) - 1, min(24, len(fields["times"])), dtype=int)
            )
            movie = FuncAnimation(
                comparison.figure, comparison.update, frames=frame_ids, interval=125, blit=False
            )
            movie.save(gif_path, writer=PillowWriter(fps=8), dpi=70)
            # Keep the README preview small without changing stored simulation fields.
            if gif_path.stat().st_size > 2_000_000:
                movie.save(gif_path, writer=PillowWriter(fps=8), dpi=52)
            outputs.append(gif_path)
        if html:
            html_path = run_dir / "comparison.html"
            movie = FuncAnimation(
                comparison.figure,
                comparison.update,
                frames=len(fields["times"]),
                interval=125,
                blit=False,
            )
            html_path.write_text(
                _offline_html(movie, len(fields["times"]), metrics), encoding="utf-8"
            )
            outputs.append(html_path)
        comparison.figure.clear()
    return outputs


def show(run_dir: Path) -> None:
    """Open the local Matplotlib viewer (requires an interactive backend/display)."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RadioButtons, Slider

    fields, metrics = _load(Path(run_dir))
    with mpl.rc_context(_STYLE):
        figure = plt.figure(figsize=(13, 7))
        comparison = _Comparison(fields, metrics, figure=figure, interactive=True)
        slider_axis = figure.add_axes((0.12, 0.135, 0.67, 0.025))
        times = fields["times"]
        slider = Slider(
            slider_axis,
            "Time",
            float(times[0]),
            float(times[-1]) if len(times) > 1 else float(times[0] + 1),
            valinit=float(times[0]),
            valstep=times,
            valfmt="%.3f",
            color=_ACCENTS["pinn"],
        )
        quantity_buttons = RadioButtons(
            figure.add_axes((0.055, 0.018, 0.20, 0.088)), _QUANTITIES, activecolor=_ACCENTS["pinn"]
        )
        reference_buttons = RadioButtons(
            figure.add_axes((0.29, 0.018, 0.22, 0.088)),
            tuple(_REFERENCES),
            activecolor=_ACCENTS["exact"],
        )
        model_buttons = RadioButtons(
            figure.add_axes((0.55, 0.018, 0.27, 0.088)),
            ("Trained PINN", "Zero-field control"),
            activecolor=_ACCENTS["pinn"],
        )
        play = Button(
            figure.add_axes((0.855, 0.115, 0.095, 0.055)),
            "Play",
            color="#253149",
            hovercolor="#344563",
        )
        playing = {"value": False}

        def changed_time(value: float) -> None:
            comparison.update(int(np.argmin(np.abs(times - value))))
            figure.canvas.draw_idle()

        def changed_quantity(value: str) -> None:
            comparison.select(quantity=value)
            figure.canvas.draw_idle()

        def changed_reference(value: str) -> None:
            comparison.select(reference=value)
            figure.canvas.draw_idle()

        def changed_model(value: str) -> None:
            comparison.select(control=value == "Zero-field control")
            figure.canvas.draw_idle()

        def toggle_play(_: Any) -> None:
            playing["value"] = not playing["value"]
            play.label.set_text("Pause" if playing["value"] else "Play")
            figure.canvas.draw_idle()

        def advance() -> None:
            if playing["value"]:
                slider.set_val(times[(comparison.index + 1) % len(times)])

        slider.on_changed(changed_time)
        quantity_buttons.on_clicked(changed_quantity)
        reference_buttons.on_clicked(changed_reference)
        model_buttons.on_clicked(changed_model)
        play.on_clicked(toggle_play)
        timer = figure.canvas.new_timer(interval=125)
        timer.add_callback(advance)
        timer.start()
        # Retain widget/callback ownership while the window is open.
        figure._pinnlab_controls = (
            slider,
            quantity_buttons,
            reference_buttons,
            model_buttons,
            play,
            timer,
        )  # type: ignore[attr-defined]
        figure.canvas.mpl_connect("close_event", lambda _: timer.stop())
        plt.show()
