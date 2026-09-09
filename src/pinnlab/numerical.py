"""A small Fourier/RK4 solver for periodic 2D incompressible Navier–Stokes.

This is a numerical baseline, not an analytic solution renderer. It evolves
omega_t = -(u omega_x + v omega_y) + nu Laplacian(omega), reconstructs the
zero-mean velocity by Biot–Savart, and recovers pressure through a Poisson
solve. The grid uses indexing='xy': arrays have axes (y, x).
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from pinnlab.reference import vorticity_numpy


def _grid(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 8 or n % 2:
        raise ValueError("n must be an even integer >= 8")
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    k = np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    k2 = kx**2 + ky**2
    # Strictly below n/3 avoids retaining the aliased cutoff interaction.
    cutoff = (n - 1) // 3
    mask = (np.abs(kx) <= cutoff) & (np.abs(ky) <= cutoff)
    return x, kx, ky, k2, mask


def _parameters(nu: float, times: np.ndarray, dt: float) -> np.ndarray:
    if not np.isfinite(nu) or nu <= 0:
        raise ValueError("nu must be positive and finite")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError("times must be a nonempty finite 1D array")
    if np.any(times < 0) or np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted and nonnegative")
    return times


def _velocity_hat(
    omega_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray, k2: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    inverse_k2 = np.divide(1.0, k2, out=np.zeros_like(k2), where=k2 != 0)
    return 1j * ky * inverse_k2 * omega_hat, -1j * kx * inverse_k2 * omega_hat


def _rhs(
    omega_hat: np.ndarray,
    nu: float,
    kx: np.ndarray,
    ky: np.ndarray,
    k2: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    omega_hat = omega_hat * mask
    u_hat, v_hat = _velocity_hat(omega_hat, kx, ky, k2)
    u, v = np.fft.ifft2(u_hat).real, np.fft.ifft2(v_hat).real
    wx = np.fft.ifft2(1j * kx * omega_hat).real
    wy = np.fft.ifft2(1j * ky * omega_hat).real
    nonlinear_hat = np.fft.fft2(u * wx + v * wy)
    return (-nonlinear_hat - nu * k2 * omega_hat) * mask


def spectral_rhs(omega_hat: np.ndarray, nu: float) -> np.ndarray:
    """Return d(FFT(omega))/dt, including dealiased nonlinear advection.

    This public primitive permits independent tests on non-Taylor–Green
    fields: Taylor–Green's vorticity advection alone is identically zero.
    """
    omega_hat = np.asarray(omega_hat)
    if omega_hat.ndim != 2 or omega_hat.shape[0] != omega_hat.shape[1]:
        raise ValueError("omega_hat must be a square 2D array")
    if not np.isfinite(omega_hat).all():
        raise ValueError("omega_hat must be finite")
    _parameters(nu, np.array([0.0]), 1.0)
    _, kx, ky, k2, mask = _grid(omega_hat.shape[0])
    return _rhs(omega_hat, nu, kx, ky, k2, mask)


def _fields(
    omega_hat: np.ndarray, kx: np.ndarray, ky: np.ndarray, k2: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_hat, v_hat = _velocity_hat(omega_hat, kx, ky, k2)
    velocity = np.stack((np.fft.ifft2(u_hat).real, np.fft.ifft2(v_hat).real), axis=-1)
    # Form the quadratic pressure source on a 3/2 padded grid, then project
    # back. Otherwise unresolved high modes would alias into low pressure
    # modes even though the vorticity evolution itself is dealiased.
    n = omega_hat.shape[0]
    padded_n = 3 * n // 2
    indices = np.rint(np.fft.fftfreq(n, d=1.0 / n)).astype(int) % padded_n

    def padded_derivative(derivative_hat: np.ndarray) -> np.ndarray:
        padded = np.zeros((padded_n, padded_n), dtype=np.complex128)
        padded[np.ix_(indices, indices)] = derivative_hat * (padded_n / n) ** 2
        return np.fft.ifft2(padded).real

    ux = padded_derivative(1j * kx * u_hat)
    uy = padded_derivative(1j * ky * u_hat)
    vx = padded_derivative(1j * kx * v_hat)
    vy = padded_derivative(1j * ky * v_hat)
    # Delta p = -(ux^2 + 2 uy vx + vy^2). The zero mode fixes the gauge.
    padded_source = np.fft.fft2(-(ux**2 + 2.0 * uy * vx + vy**2))
    source_hat = padded_source[np.ix_(indices, indices)] * (n / padded_n) ** 2
    # Exclude the ambiguous even-grid Nyquist modes in the projection.
    source_hat[n // 2, :] = 0.0
    source_hat[:, n // 2] = 0.0
    p_hat = np.divide(-source_hat, k2, out=np.zeros_like(source_hat), where=k2 != 0)
    pressure = np.fft.ifft2(p_hat).real
    return velocity, pressure, np.fft.ifft2(omega_hat).real


def simulate_vorticity(
    initial_vorticity: np.ndarray,
    nu: float,
    times: np.ndarray,
    dt: float = 0.01,
) -> dict:
    """Integrate a real, zero-mean initial vorticity on [-pi, pi)^2.

    The initial field is projected onto the rectangular 2/3 spectral band.
    Mean velocity is fixed to zero. Output times may repeat and need not
    start at zero; integration always starts at zero. The requested dt is
    an upper bound, capped dynamically by advection and diffusion limits.
    Pressure is a diagnostic Poisson solve with a 3/2-padded quadratic source,
    projected back to the output grid with its Nyquist modes excluded.
    """
    started = perf_counter()
    times = _parameters(nu, times, dt)
    initial = np.asarray(initial_vorticity)
    if initial.ndim != 2 or initial.shape[0] != initial.shape[1]:
        raise ValueError("initial_vorticity must be a square 2D array")
    if np.iscomplexobj(initial) or not np.isfinite(initial).all():
        raise ValueError("initial_vorticity must be real and finite")
    n = initial.shape[0]
    x, kx, ky, k2, mask = _grid(n)
    if abs(float(initial.mean())) > 1e-10 * (1.0 + float(np.abs(initial).max())):
        raise ValueError("periodic initial_vorticity must have zero spatial mean")
    omega_hat = np.fft.fft2(initial.astype(np.float64)) * mask
    omega_hat[0, 0] = 0.0
    velocity = np.empty((len(times), n, n, 2))
    pressure = np.empty((len(times), n, n))
    vorticity = np.empty((len(times), n, n))
    current_time = 0.0
    steps = 0
    step_min = np.inf
    step_max = 0.0
    dx = 2.0 * np.pi / n
    diffusion_cap = 2.0 / (nu * np.max(k2[mask]))

    for index, target in enumerate(times):
        while current_time < target:
            remaining = float(target - current_time)
            if remaining <= 8.0 * np.spacing(float(target)):
                # Do not report roundoff-only residual intervals as actual
                # adaptive time steps (e.g. 0.1 - ten additions of 0.01).
                current_time = float(target)
                break
            u_hat, v_hat = _velocity_hat(omega_hat, kx, ky, k2)
            max_speed_sum = float(
                np.abs(np.fft.ifft2(u_hat).real).max() + np.abs(np.fft.ifft2(v_hat).real).max()
            )
            advection_cap = 0.5 * dx / max(max_speed_sum, 1e-15)
            step = min(dt, diffusion_cap, advection_cap, remaining)
            if current_time + step == current_time:
                raise FloatingPointError("time step cannot advance floating-point time")
            k1 = _rhs(omega_hat, nu, kx, ky, k2, mask)
            k2_stage = _rhs(omega_hat + 0.5 * step * k1, nu, kx, ky, k2, mask)
            k3 = _rhs(omega_hat + 0.5 * step * k2_stage, nu, kx, ky, k2, mask)
            k4 = _rhs(omega_hat + step * k3, nu, kx, ky, k2, mask)
            omega_hat += step / 6.0 * (k1 + 2 * k2_stage + 2 * k3 + k4)
            omega_hat *= mask
            omega_hat[0, 0] = 0.0
            if not np.isfinite(omega_hat).all():
                raise FloatingPointError(
                    "non-finite numerical solution; reduce dt or increase resolution"
                )
            current_time = min(float(target), current_time + step)
            steps += 1
            step_min = min(step_min, step)
            step_max = max(step_max, step)
        velocity[index], pressure[index], vorticity[index] = _fields(omega_hat, kx, ky, k2)

    return {
        "x": x,
        "y": x.copy(),
        "times": times.copy(),
        "velocity": velocity,
        "pressure": pressure,
        "vorticity": vorticity,
        "elapsed_seconds": perf_counter() - started,
        "n": n,
        "dt": dt,
        "effective_dt_min": float(step_min) if steps else 0.0,
        "effective_dt_max": float(step_max),
        "steps": steps,
    }


def simulate(
    nu: float,
    times: np.ndarray,
    n: int = 48,
    dt: float = 0.01,
    mode: int = 1,
) -> dict:
    """Advance Taylor–Green initial vorticity using the general numerical solver.

    Only t=0 is evaluated analytically. All later states result from RK4
    integration. n > 4*mode is required to resolve the pressure's 2*mode.
    """
    _parameters(nu, times, dt)
    x, _, _, _, _ = _grid(n)
    if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)) or mode < 1:
        raise ValueError("mode must be a positive integer")
    if n <= 4 * mode:
        raise ValueError("n must exceed 4*mode to resolve the pressure wavenumber")
    xx, yy = np.meshgrid(x, x, indexing="xy")
    initial_coords = np.stack((xx, yy, np.zeros_like(xx)), axis=-1)
    initial = vorticity_numpy(initial_coords, nu=nu, mode=mode)
    return simulate_vorticity(initial, nu, times, dt=dt)
