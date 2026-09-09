import math

import pytest
import torch

from pinnlab.model import FlowPINN
from pinnlab.physics import gradient, navier_stokes_residual, taylor_green_torch


@pytest.mark.parametrize("mode", [1, 2])
def test_exact_solution_satisfies_equations(mode):
    torch.manual_seed(7)
    coords = torch.rand(128, 3, dtype=torch.float64, requires_grad=True)
    fields = taylor_green_torch(coords, nu=0.13, mode=mode)
    residual = navier_stokes_residual(fields, coords, nu=0.13)
    torch.testing.assert_close(residual, torch.zeros_like(residual), atol=3e-14, rtol=0)


def test_constant_and_affine_field_derivatives():
    coords = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
    constant = torch.zeros_like(coords)
    torch.testing.assert_close(navier_stokes_residual(constant, coords, 0.1), constant)
    x, y, t = coords.unbind(dim=1)
    # u=x+t, v=-y, p=0: div=0, rx=1+x+t, ry=y.
    fields = torch.stack((x + t, -y, torch.zeros_like(t)), dim=1)
    expected = torch.stack((1 + x + t, y, torch.zeros_like(t)), dim=1)
    torch.testing.assert_close(navier_stokes_residual(fields, coords, 0.1), expected)


def test_second_derivative_signs_and_parameter_gradient():
    coords = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    coefficient = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    x, y, t = coords.unbind(dim=1)
    fields = torch.stack(
        (coefficient * y.square(), torch.zeros_like(x), torch.zeros_like(t)), dim=1
    )
    residual = navier_stokes_residual(fields, coords, 0.2)
    torch.testing.assert_close(residual[:, 0], torch.full_like(x, -0.28))
    derivative = torch.autograd.grad(residual[:, 0].sum(), coefficient)[0]
    torch.testing.assert_close(derivative, torch.tensor(-0.4 * len(coords), dtype=torch.float64))


@pytest.mark.parametrize("mode", [1, 2])
def test_hard_initial_condition_and_pressure_gauge(mode):
    model = FlowPINN(mode=mode).double()
    coords = torch.randn(13, 3, dtype=torch.float64)
    coords[:, 2] = 0
    expected = taylor_green_torch(coords, model.nu, mode)
    torch.testing.assert_close(model(coords)[:, :2], expected[:, :2], atol=0, rtol=0)
    coords[:, :2] = 0
    coords[:, 2] = torch.linspace(0, model.tmax, len(coords))
    torch.testing.assert_close(model(coords)[:, 2], torch.zeros_like(coords[:, 2]), atol=0, rtol=0)


@pytest.mark.parametrize("axis", [0, 1])
def test_periodic_values_and_first_derivatives(axis):
    model = FlowPINN().double()
    left = torch.randn(15, 3, dtype=torch.float64)
    left[:, axis] = -math.pi
    right = left.clone()
    right[:, axis] = math.pi
    left.requires_grad_(True)
    right.requires_grad_(True)
    fleft, fright = model(left), model(right)
    torch.testing.assert_close(fleft, fright, atol=1e-13, rtol=1e-13)
    for component in range(3):
        torch.testing.assert_close(
            gradient(fleft[:, component], left),
            gradient(fright[:, component], right),
            atol=1e-13,
            rtol=1e-13,
        )


def test_model_derivative_matches_finite_difference():
    torch.manual_seed(1)
    model = FlowPINN(width=8, depth=2).double()
    coords = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    autodiff = gradient(model(coords)[:, 0], coords)
    epsilon = 1e-5
    for axis in range(3):
        shift = torch.zeros_like(coords)
        shift[:, axis] = epsilon
        finite_difference = (model(coords + shift)[:, 0] - model(coords - shift)[:, 0]) / (
            2 * epsilon
        )
        torch.testing.assert_close(autodiff[:, axis], finite_difference, atol=1e-9, rtol=1e-7)
