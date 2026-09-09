"""Reproducible forward PINN training; no future reference data is sampled."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch

from pinnlab.model import FlowPINN
from pinnlab.physics import navier_stokes_residual, taylor_green_torch, validate_physics


@dataclass
class TrainConfig:
    nu: float = 0.1
    tmax: float = 1.0
    mode: int = 1
    width: int = 32
    depth: int = 3
    hard_ic: bool = True
    seed: int = 42
    dtype: str = "float64"
    adam_steps: int = 1000
    lbfgs_steps: int = 100
    batch_size: int = 512
    learning_rate: float = 0.001
    threads: int = 1
    device: str = "cpu"

    def __post_init__(self) -> None:
        validate_physics(self.nu, self.mode)
        for name in ("width", "depth", "batch_size", "threads"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("adam_steps", "lbfgs_steps", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.seed >= 2**32:
            raise ValueError("seed must be smaller than 2**32")
        if self.adam_steps + self.lbfgs_steps == 0:
            raise ValueError("at least one optimizer step is required")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if not isinstance(self.hard_ic, bool):
            raise ValueError("hard_ic must be a boolean")
        for name in ("tmax", "learning_rate"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        parsed_device = torch.device(self.device)
        if parsed_device.type not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda[:index]")


def _sample(config: TrainConfig, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    points = torch.rand(config.batch_size, 3, device=device, dtype=dtype)
    scales = points.new_tensor([2 * math.pi, 2 * math.pi, config.tmax])
    offsets = points.new_tensor([-math.pi, -math.pi, 0])
    return (points * scales + offsets).requires_grad_(True)


def _loss(
    model: FlowPINN, points: torch.Tensor, initial_points: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    residual = navier_stokes_residual(model(points), points, model.nu)
    momentum = residual[:, :2].square().mean()
    divergence = residual[:, 2].square().mean()
    # Evaluating the prescribed initial condition is allowed. No t>0 labels
    # occur here. The hard-IC variant gives exactly zero by construction.
    target = taylor_green_torch(initial_points, model.nu, model.mode)[:, :2]
    ic = (model(initial_points)[:, :2] - target).square().mean()
    total = momentum + divergence + 10 * ic
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite training loss; checkpoint was not saved")
    return total, {
        "loss": total.detach().item(),
        "momentum": momentum.detach().item(),
        "divergence": divergence.detach().item(),
        "ic": ic.detach().item(),
    }


def _check_gradients(model: FlowPINN) -> None:
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError("nonfinite parameter gradient; checkpoint was not saved")


def train(config: TrainConfig, output_dir: str | Path) -> dict:
    """Train, then save checkpoint.pt, loss history, and measured run provenance.

    Adam uses fresh uniform space-time points at each update. L-BFGS uses one
    fixed independent point set throughout its line searches. Recorded losses
    are optimizer evaluations, not independent validation metrics.
    """
    config.__post_init__()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)
    dtype = getattr(torch, config.dtype)
    constructor = {
        key: getattr(config, key) for key in ("nu", "tmax", "mode", "width", "depth", "hard_ic")
    }
    model = FlowPINN(**constructor).to(device=device, dtype=dtype)
    history: list[dict] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    initial_points = _sample(config, device, dtype).detach()
    initial_points[:, 2] = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    for step in range(1, config.adam_steps + 1):
        points = _sample(config, device, dtype)
        # Soft-IC training also refreshes its initial-condition points.
        if not config.hard_ic:
            initial_points = points.detach().clone()
            initial_points[:, 2] = 0
        optimizer.zero_grad(set_to_none=True)
        loss, parts = _loss(model, points, initial_points)
        loss.backward()
        _check_gradients(model)
        optimizer.step()
        history.append({"step": step, "phase": "adam", **parts})
        if step == 1 or step % 100 == 0 or step == config.adam_steps:
            print(f"Adam {step}/{config.adam_steps}: loss={parts['loss']:.6e}", flush=True)
    lbfgs_evaluations = 0
    if config.lbfgs_steps:
        points = _sample(config, device, dtype)
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=1,
            max_iter=config.lbfgs_steps,
            max_eval=config.lbfgs_steps * 2,
            history_size=50,
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            nonlocal lbfgs_evaluations
            optimizer.zero_grad(set_to_none=True)
            points.grad = None
            loss, parts = _loss(model, points, initial_points)
            loss.backward()
            _check_gradients(model)
            lbfgs_evaluations += 1
            history.append(
                {"step": config.adam_steps + lbfgs_evaluations, "phase": "lbfgs_eval", **parts}
            )
            if lbfgs_evaluations == 1 or lbfgs_evaluations % 100 == 0:
                print(
                    f"L-BFGS evaluation {lbfgs_evaluations}: loss={parts['loss']:.6e}", flush=True
                )
            return loss

        optimizer.step(closure)
        print(f"L-BFGS finished: {lbfgs_evaluations} closure evaluations", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("nonfinite model parameter; checkpoint was not saved")
    # Final collocation loss is only a diagnostic, explicitly not validation.
    _, final_loss = _loss(model, _sample(config, device, dtype), initial_points)
    checkpoint = {
        "model_config": model.get_config(),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": asdict(config),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    with (output / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("step", "phase", "loss", "momentum", "divergence", "ic")
        )
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "config": asdict(config),
        "seed": config.seed,
        "training_seconds": elapsed,
        "elapsed_seconds": elapsed,
        "adam_updates": config.adam_steps,
        "lbfgs_closure_evaluations": lbfgs_evaluations,
        "final_collocation_loss": final_loss,
        "training_data": "PDE residual and prescribed t=0 velocity only; no future reference labels",
        "pressure_gauge": "p(0,0,t)=0",
        "hardware": {
            "device": str(device),
            "cpu": platform.processor() or platform.machine(),
            "threads": config.threads,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_vram_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        },
        "software": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
        },
    }
    (output / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[FlowPINN, dict]:
    """Load a tensor-only checkpoint and return its model and training config."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = FlowPINN(**checkpoint["model_config"]).to(
        device=device, dtype=getattr(torch, config["dtype"])
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, config
