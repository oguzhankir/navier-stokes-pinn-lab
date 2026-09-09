"""Density-normalized, unforced incompressible Navier–Stokes in two dimensions."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_physics(nu: float, mode: int = 1) -> None:
    """Reject a nonphysical viscosity or a nonperiodic benchmark mode."""
    if not math.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be finite and positive")
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 1:
        raise ValueError("mode must be a positive integer")


def taylor_green_torch(coords: Tensor, nu: float, mode: int = 1) -> Tensor:
    """Return exact [u, v, p] at [x, y, t], with zero spatial-mean pressure.

    The box is [-pi, pi)^2; an integer mode is necessary for periodicity.
    This reference function is used at t=0 only during forward PINN training.
    """
    validate_physics(nu, mode)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must have shape (N, 3): [x, y, t]")
    x, y, t = coords.unbind(dim=1)
    x, y = mode * x, mode * y
    decay = torch.exp(-2 * nu * mode**2 * t)
    u = -torch.cos(x) * torch.sin(y) * decay
    v = torch.sin(x) * torch.cos(y) * decay
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * decay.square()
    return torch.stack((u, v, p), dim=1)


def gradient(field: Tensor, coords: Tensor) -> Tensor:
    """Differentiate a pointwise scalar field, retaining higher derivatives.

    A batch must contain independent pointwise predictions (no batch coupling).
    Constant fields and affine fields' second derivatives are valid zeros, not
    autograd errors. ``coords * 0`` keeps those zeros differentiable again.
    """
    if not coords.requires_grad:
        raise ValueError("coords must require gradients")
    if not field.requires_grad:
        return coords * 0
    result = torch.autograd.grad(
        field,
        coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return coords * 0 if result is None else result


def navier_stokes_residual(fields: Tensor, coords: Tensor, nu: float) -> Tensor:
    """Return [x-momentum, y-momentum, divergence] using exact autodiff.

    Momentum uses ``u_t + u*u_x + v*u_y + p_x - nu*(u_xx+u_yy)``.
    No detached predictions, finite differences, or reference labels are used.
    """
    validate_physics(nu)
    if coords.ndim != 2 or coords.shape[1] != 3 or fields.shape != coords.shape:
        raise ValueError("fields and coords must both have shape (N, 3)")
    u, v, p = fields.unbind(dim=1)
    du, dv, dp = (gradient(field, coords) for field in (u, v, p))
    uxx = gradient(du[:, 0], coords)[:, 0]
    uyy = gradient(du[:, 1], coords)[:, 1]
    vxx = gradient(dv[:, 0], coords)[:, 0]
    vyy = gradient(dv[:, 1], coords)[:, 1]
    rx = du[:, 2] + u * du[:, 0] + v * du[:, 1] + dp[:, 0] - nu * (uxx + uyy)
    ry = dv[:, 2] + u * dv[:, 0] + v * dv[:, 1] + dp[:, 1] - nu * (vxx + vyy)
    return torch.stack((rx, ry, du[:, 0] + dv[:, 1]), dim=1)
