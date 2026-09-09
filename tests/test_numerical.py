"""Independent analytic, nonlinear, and convergence checks for the baseline."""

import numpy as np
import pytest

from pinnlab.numerical import simulate, simulate_vorticity, spectral_rhs
from pinnlab.reference import taylor_green_numpy, vorticity_numpy


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_taylor_green_velocity_pressure_and_vorticity(mode):
    result = simulate(0.1, np.array([0.0, 0.1, 0.6]), n=32, dt=0.01, mode=mode)
    xx, yy = np.meshgrid(result["x"], result["y"], indexing="xy")
    for index, time in enumerate(result["times"]):
        coords = np.stack((xx, yy, np.full_like(xx, time)), axis=-1)
        exact = taylor_green_numpy(coords, nu=0.1, mode=mode)
        np.testing.assert_allclose(result["velocity"][index], exact[..., :2], atol=3e-9)
        np.testing.assert_allclose(result["pressure"][index], exact[..., 2], atol=3e-9)
        np.testing.assert_allclose(
            result["vorticity"][index], vorticity_numpy(coords, nu=0.1, mode=mode), atol=2e-8
        )
    assert result["elapsed_seconds"] >= 0
    assert 0 < result["effective_dt_max"] <= result["dt"]
    assert result["steps"] > 0


def test_divergence_and_pressure_gauge():
    result = simulate(0.1, [0.0, 0.2], n=24)
    k = np.fft.fftfreq(24, d=1 / 24)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    for velocity, pressure in zip(result["velocity"], result["pressure"]):
        div = np.fft.ifft2(
            1j * kx * np.fft.fft2(velocity[..., 0]) + 1j * ky * np.fft.fft2(velocity[..., 1])
        ).real
        assert np.abs(div).max() < 1e-13
        assert abs(pressure.mean()) < 1e-15


def test_multimode_rhs_has_correct_nonzero_convection():
    n = 24
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    omega = np.sin(xx) + np.sin(2 * yy)
    # Biot–Savart gives u=cos(2y)/2, v=-cos(x), hence
    # -(u*omega_x + v*omega_y) = 3*cos(x)*cos(2y)/2.
    convection = 1.5 * np.cos(xx) * np.cos(2 * yy)
    diffusion = -0.1 * np.sin(xx) - 0.4 * np.sin(2 * yy)
    actual = np.fft.ifft2(spectral_rhs(np.fft.fft2(omega), nu=0.1)).real
    assert np.linalg.norm(convection) > 1.0
    np.testing.assert_allclose(actual, convection + diffusion, atol=2e-13)


def test_rk4_timestep_convergence_on_nonlinear_flow():
    n = 16
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    initial = np.sin(xx) + np.sin(2 * yy)
    coarse = simulate_vorticity(initial, 0.1, [0.4], dt=0.04)["vorticity"]
    fine = simulate_vorticity(initial, 0.1, [0.4], dt=0.02)["vorticity"]
    reference = simulate_vorticity(initial, 0.1, [0.4], dt=0.0025)["vorticity"]
    coarse_error = np.linalg.norm(coarse - reference)
    fine_error = np.linalg.norm(fine - reference)
    assert fine_error > 1e-12
    assert coarse_error / fine_error > 10.0


def test_output_times_need_not_start_at_zero_and_may_repeat():
    direct = simulate(0.1, [0.2, 0.2], n=16)
    all_times = simulate(0.1, [0.0, 0.2], n=16)
    np.testing.assert_array_equal(direct["velocity"][0], direct["velocity"][1])
    np.testing.assert_array_equal(direct["velocity"][0], all_times["velocity"][1])


def test_step_caps_are_applied():
    result = simulate(1.0, [0.1], n=32, dt=10.0)
    assert result["effective_dt_max"] < 0.1
    assert result["steps"] > 1
    assert np.isfinite(result["velocity"]).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nu": 0},
        {"nu": -1},
        {"nu": np.nan},
        {"nu": np.inf},
        {"times": []},
        {"times": [0.2, 0.1]},
        {"times": [-0.1]},
        {"times": [np.nan]},
        {"times": [[0.0]]},
        {"dt": 0},
        {"dt": np.inf},
        {"n": 7},
        {"n": 15},
        {"n": True},
        {"mode": 0},
        {"mode": 1.5},
        {"mode": True},
        {"n": 16, "mode": 4},
    ],
)
def test_invalid_parameters(kwargs):
    params = {"nu": 0.1, "times": [0.0], "n": 16}
    params.update(kwargs)
    with pytest.raises(ValueError):
        simulate(**params)


def test_generic_initial_vorticity_validation():
    with pytest.raises(ValueError, match="zero spatial mean"):
        simulate_vorticity(np.ones((16, 16)), 0.1, [0.0])
    with pytest.raises(ValueError, match="square"):
        simulate_vorticity(np.zeros((16, 15)), 0.1, [0.0])
    with pytest.raises(ValueError, match="real and finite"):
        simulate_vorticity(np.full((16, 16), np.nan), 0.1, [0.0])


def test_two_thirds_spectral_projection():
    n = 24
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    initial = np.sin(xx) + np.sin(9 * yy)
    result = simulate_vorticity(initial, 0.1, [0.0])
    np.testing.assert_allclose(result["vorticity"][0], np.sin(xx), atol=3e-15)


def test_pressure_source_padding_prevents_high_mode_aliasing():
    n = 16
    x = np.linspace(-np.pi, np.pi, n, endpoint=False)
    xx, yy = np.meshgrid(x, x, indexing="xy")
    initial = 10.0 * np.cos(5 * xx) * np.cos(5 * yy)
    result = simulate_vorticity(initial, 0.1, [0.0])
    # This velocity's pressure has only wavenumber 10, outside the output
    # grid's resolved range. An unpadded product would alias it to mode 6.
    assert np.abs(result["pressure"]).max() < 2e-14


def test_analytic_reference_is_only_called_at_initial_time(monkeypatch):
    import pinnlab.numerical as numerical

    calls = []
    original = numerical.vorticity_numpy

    def guarded_reference(coords, nu, mode):
        assert np.all(coords[..., 2] == 0)
        calls.append(coords.shape)
        return original(coords, nu=nu, mode=mode)

    monkeypatch.setattr(numerical, "vorticity_numpy", guarded_reference)
    result = numerical.simulate(0.1, [0.2, 0.4], n=16)
    assert len(calls) == 1
    assert result["steps"] > 1
