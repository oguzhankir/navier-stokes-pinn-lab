"""Independent model checks and explicitly labeled reference artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from .numerical import simulate
from .physics import navier_stokes_residual
from .reference import taylor_green_numpy, vorticity_numpy
from .train import load_model


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    """Vector/component error; do not compare only speed magnitudes."""
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((prediction - reference).ravel()))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return numerator / denominator


def align_pressure(pressure: np.ndarray) -> np.ndarray:
    """Remove the spatial mean separately at each time (last two dimensions)."""
    return pressure - pressure.mean(axis=(-2, -1), keepdims=True)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _predict(model, coordinates: np.ndarray, batch_size: int = 1024):
    param = next(model.parameters())
    predictions, curls = [], []
    for start in range(0, len(coordinates), batch_size):
        c = torch.as_tensor(
            coordinates[start : start + batch_size], dtype=param.dtype, device=param.device
        ).requires_grad_(True)
        with torch.enable_grad():
            q = model(c)
            du = torch.autograd.grad(q[:, 0].sum(), c, retain_graph=True)[0]
            dv = torch.autograd.grad(q[:, 1].sum(), c)[0]
        predictions.append(q.detach().cpu().numpy())
        curls.append((dv[:, 0] - du[:, 1]).detach().cpu().numpy())
    return np.concatenate(predictions), np.concatenate(curls)


def _residual_audit(model, coordinates: np.ndarray):
    param = next(model.parameters())
    values = []
    for start in range(0, len(coordinates), 256):
        c = torch.as_tensor(
            coordinates[start : start + 256], dtype=param.dtype, device=param.device
        ).requires_grad_(True)
        with torch.enable_grad():
            residual = navier_stokes_residual(model(c), c, model.nu)
        values.append(residual.detach().cpu().numpy())
    r = np.concatenate(values)
    return {
        "heldout_momentum_rms": float(np.sqrt(np.mean(r[:, :2] ** 2))),
        "heldout_momentum_max_abs": float(np.max(np.abs(r[:, :2]))),
        "heldout_divergence_rms": float(np.sqrt(np.mean(r[:, 2] ** 2))),
        "heldout_divergence_max_abs": float(np.max(np.abs(r[:, 2]))),
    }


def _periodic_audit(model, rng):
    param = next(model.parameters())
    value_error, derivative_error = 0.0, 0.0
    for axis in (0, 1):
        a = rng.uniform(-np.pi, np.pi, (128, 3))
        a[:, 2] = rng.uniform(0, model.tmax, 128)
        b = a.copy()
        a[:, axis], b[:, axis] = -np.pi, np.pi
        tensors = [
            torch.tensor(c, device=param.device, dtype=param.dtype, requires_grad=True)
            for c in (a, b)
        ]
        fields = [model(c) for c in tensors]
        value_error = max(value_error, float((fields[0] - fields[1]).abs().max().detach()))
        for component in range(3):
            derivatives = [
                torch.autograd.grad(f[:, component].sum(), c, retain_graph=True)[0]
                for f, c in zip(fields, tensors)
            ]
            derivative_error = max(
                derivative_error, float((derivatives[0] - derivatives[1]).abs().max().detach())
            )
    return {"periodic_value_max_abs": value_error, "periodic_derivative_max_abs": derivative_error}


def _particles(model, times: np.ndarray, seed: int):
    """RK4 tracers integrated through the learned field, not an extra fluid solver."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(-np.pi, np.pi, (48, 2))
    result = [p.copy()]
    param = next(model.parameters())

    def velocity(x, t):
        c = np.column_stack(((x + np.pi) % (2 * np.pi) - np.pi, np.full(len(x), t)))
        with torch.no_grad():
            return (
                model(torch.as_tensor(c, device=param.device, dtype=param.dtype))[:, :2]
                .cpu()
                .numpy()
            )

    for left, right in zip(times[:-1], times[1:]):
        steps = max(1, int(np.ceil((right - left) / 0.025)))
        h = (right - left) / steps
        for j in range(steps):
            t = left + j * h
            k1 = velocity(p, t)
            k2 = velocity(p + h * k1 / 2, t + h / 2)
            k3 = velocity(p + h * k2 / 2, t + h / 2)
            k4 = velocity(p + h * k3, t + h)
            p = (p + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6 + np.pi) % (2 * np.pi) - np.pi
        result.append(p.copy())
    return np.asarray(result)


def evaluate(
    run_dir: Path | str,
    n: int = 48,
    frames: int = 21,
    residual_points: int = 2048,
    dt: float = 0.02,
    device: str = "cpu",
) -> dict:
    """Write held-out metrics and raw model/reference arrays from a saved checkpoint."""
    run_dir = Path(run_dir)
    if n < 8 or n % 2 or frames < 2 or residual_points < 1:
        raise ValueError("Use an even grid >= 8, at least 2 frames, and positive residual_points.")
    checkpoint = run_dir / "checkpoint.pt"
    model, saved_config = load_model(checkpoint, device=device)
    model.eval()
    config = model.get_config()
    training_file = run_dir / "training.json"
    training = json.loads(training_file.read_text()) if training_file.exists() else {}
    times = np.linspace(0, model.tmax, frames)
    axis = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coords = np.stack([np.column_stack((xx.ravel(), yy.ravel(), np.full(n * n, t))) for t in times])
    param = next(model.parameters())
    _sync(param.device)
    started = time.perf_counter()
    predicted, omega = _predict(model, coords.reshape(-1, 3))
    _sync(param.device)
    evaluation_seconds = time.perf_counter() - started
    predicted = predicted.reshape(frames, n, n, 3)
    omega = omega.reshape(frames, n, n)
    exact = taylor_green_numpy(coords, model.nu, model.mode).reshape(frames, n, n, 3)
    exact_omega = vorticity_numpy(coords, model.nu, model.mode).reshape(frames, n, n)
    numerical = simulate(model.nu, times, n=n, dt=dt, mode=model.mode)
    predicted_p, exact_p = align_pressure(predicted[..., 2]), align_pressure(exact[..., 2])
    numerical_p = align_pressure(numerical["pressure"])
    if not all(
        np.isfinite(a).all() for a in [predicted, omega, numerical["velocity"], numerical_p]
    ):
        raise FloatingPointError("Non-finite output; refusing to publish evaluation artifacts.")
    per_time = []
    for i, t in enumerate(times):
        puv, euv, nuv = predicted[i, ..., :2], exact[i, ..., :2], numerical["velocity"][i]
        per_time.append(
            {
                "time": float(t),
                "pinn_velocity_relative_l2": relative_l2(puv, euv),
                "numerical_velocity_relative_l2": relative_l2(nuv, euv),
                "pinn_velocity_max_abs_error": float(np.max(np.abs(puv - euv))),
                "pinn_pressure_relative_l2": relative_l2(predicted_p[i], exact_p[i]),
                "numerical_pressure_relative_l2": relative_l2(numerical_p[i], exact_p[i]),
                "pinn_vorticity_relative_l2": relative_l2(omega[i], exact_omega[i]),
                "pinn_energy": float(0.5 * np.mean(np.sum(puv**2, axis=-1))),
                "exact_energy": float(0.5 * np.mean(np.sum(euv**2, axis=-1))),
                "numerical_energy": float(0.5 * np.mean(np.sum(nuv**2, axis=-1))),
            }
        )
    # Distinct RNG seed from training, fixed for reproducible held-out auditing.
    seed = int(saved_config.get("seed", 42)) + 1_000_003
    rng = np.random.default_rng(seed)
    heldout = rng.uniform(-np.pi, np.pi, (residual_points, 3))
    heldout[:, 2] = rng.uniform(0, model.tmax, residual_points)
    audit = _residual_audit(model, heldout)
    audit.update(_periodic_audit(model, rng))
    audit["initial_velocity_relative_l2"] = per_time[0]["pinn_velocity_relative_l2"]
    initial = heldout.copy()
    initial[:, 2] = 0
    initial_prediction, _ = _predict(model, initial)
    audit["heldout_initial_velocity_relative_l2"] = relative_l2(
        initial_prediction[:, :2], taylor_green_numpy(initial, model.nu, model.mode)[:, :2]
    )
    audit["heldout_seed"] = seed
    audit["heldout_points"] = residual_points
    audit["warning"] = "Sampled diagnostics are not a proof of a PDE solution or regularity."
    particles = _particles(model, times, seed)
    metrics = {
        "schema_version": 1,
        "config": config,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training": training,
        "evaluation": {
            "grid_n": n,
            "frames": frames,
            "dtype": str(param.dtype),
            "device": str(param.device),
            "threads": torch.get_num_threads(),
            "hardware": platform.machine(),
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(param.device)
            if param.device.type == "cuda"
            else None,
            "software": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "numpy": np.__version__,
            },
            "field_and_vorticity_seconds": evaluation_seconds,
            "numerical_solver_seconds": float(numerical["elapsed_seconds"]),
            "numerical_requested_dt": dt,
            "numerical_effective_dt_min": numerical["effective_dt_min"],
            "numerical_effective_dt_max": numerical["effective_dt_max"],
            "numerical_steps": numerical["steps"],
            "timing_note": "Single measurements, not a speed benchmark. PINN timing includes field and curl evaluation; excludes model loading, training, residual/IC/periodicity audits, particle integration, I/O, and rendering. Numerical timing includes integration/reconstruction.",
            "energy_definition": "0.5 * spatial_mean(u^2 + v^2)",
            "pressure_gauge": "independent spatial mean subtraction at each time",
        },
        "audit": audit,
        "per_time": per_time,
        "zero_field_control": {
            "label": "Diagnostic control, NOT a trained model",
            "momentum_rms": 0.0,
            "divergence_rms": 0.0,
            "initial_velocity_relative_l2": relative_l2(
                np.zeros_like(exact[0, ..., :2]), exact[0, ..., :2]
            ),
            "explanation": "The zero velocity field with constant pressure satisfies the unforced PDE but violates this nonzero initial condition.",
        },
    }
    # Validate every metric before touching existing outputs. Stage both files on
    # the same filesystem; if interrupted between replacements, hash validation
    # in the renderer fails closed instead of silently mixing different runs.
    json.dumps(metrics, allow_nan=False)
    with tempfile.TemporaryDirectory(prefix=".evaluate-", dir=run_dir) as staging:
        staged_fields = Path(staging) / "fields.npz"
        staged_metrics = Path(staging) / "metrics.json"
        np.savez_compressed(
            staged_fields,
            x=axis,
            y=axis,
            times=times,
            pinn_uv=predicted[..., :2],
            exact_uv=exact[..., :2],
            numerical_uv=numerical["velocity"],
            pinn_p=predicted_p,
            exact_p=exact_p,
            numerical_p=numerical_p,
            pinn_omega=omega,
            exact_omega=exact_omega,
            numerical_omega=numerical["vorticity"],
            particles=particles,
        )
        metrics["fields_sha256"] = hashlib.sha256(staged_fields.read_bytes()).hexdigest()
        staged_metrics.write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
        staged_fields.replace(run_dir / "fields.npz")
        staged_metrics.replace(run_dir / "metrics.json")
    return metrics
