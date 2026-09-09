"""Analytic Taylor–Green reference fields, independent of any neural model."""

from __future__ import annotations

import numpy as np


def _inputs(coords: np.ndarray, nu: float, mode: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim == 0 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (..., 3), ordered x, y, t")
    if not np.isfinite(coords).all():
        raise ValueError("coords must be finite")
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)) or mode < 1:
        raise ValueError("mode must be a positive integer")
    return coords


def taylor_green_numpy(coords: np.ndarray, nu: float = 0.1, mode: int = 1) -> np.ndarray:
    """Return (u, v, p) at (..., x/y/t) coordinates; pressure has zero spatial mean.

    The box is [-pi, pi)^2, density is one, and the initial velocity amplitude
    is one. The mode is an integer spatial wavenumber, not a Reynolds number.
    These exact fields are for initial conditions and independent validation,
    never future labels for forward PINN training or numerical time stepping.
    """
    coords = _inputs(coords, nu, mode)
    x, y, t = np.moveaxis(coords, -1, 0)
    decay = np.exp(-2.0 * nu * mode**2 * t)
    u = -np.cos(mode * x) * np.sin(mode * y) * decay
    v = np.sin(mode * x) * np.cos(mode * y) * decay
    p = -0.25 * (np.cos(2 * mode * x) + np.cos(2 * mode * y)) * decay**2
    return np.stack((u, v, p), axis=-1)


def vorticity_numpy(coords: np.ndarray, nu: float = 0.1, mode: int = 1) -> np.ndarray:
    """Return scalar vorticity omega = d_x(v) - d_y(u)."""
    coords = _inputs(coords, nu, mode)
    x, y, t = np.moveaxis(coords, -1, 0)
    return 2.0 * mode * np.cos(mode * x) * np.cos(mode * y) * np.exp(-2.0 * nu * mode**2 * t)
