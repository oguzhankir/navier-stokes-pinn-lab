"""A small continuous-coordinate PINN with exact spatial periodicity."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from pinnlab.physics import validate_physics


class FlowPINN(nn.Module):
    """Map [x, y, t] to [u, v, p] without future analytical labels.

    Smooth periodic features enforce both value and derivative periodicity.
    The optional hard velocity IC uses only the prescribed t=0 field. Pressure
    is determined by the PDE up to a time-dependent constant; the architecture
    chooses the gauge p(0, 0, t)=0, independently at every time.
    """

    def __init__(
        self,
        nu: float = 0.1,
        tmax: float = 1.0,
        mode: int = 1,
        width: int = 32,
        depth: int = 3,
        hard_ic: bool = True,
    ) -> None:
        super().__init__()
        validate_physics(nu, mode)
        if not math.isfinite(tmax) or tmax <= 0:
            raise ValueError("tmax must be finite and positive")
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, depth)):
            raise ValueError("width and depth must be positive integers")
        if not isinstance(hard_ic, bool):
            raise ValueError("hard_ic must be a boolean")
        self.nu, self.tmax, self.mode = float(nu), float(tmax), mode
        self.width, self.depth, self.hard_ic = width, depth, hard_ic
        layers: list[nn.Module] = []
        for index in range(depth):
            layers.extend((nn.Linear(5 if index == 0 else width, width), nn.Tanh()))
        layers.append(nn.Linear(width, 3))
        self.network = nn.Sequential(*layers)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def _features(self, coords: Tensor) -> Tensor:
        x, y, t = coords.unbind(dim=1)
        return torch.stack((x.sin(), x.cos(), y.sin(), y.cos(), 2 * t / self.tmax - 1), dim=1)

    def forward(self, coords: Tensor) -> Tensor:
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("coords must have shape (N, 3): [x, y, t]")
        raw = self.network(self._features(coords))
        x, y, t = coords.unbind(dim=1)
        if self.hard_ic:
            u0 = -(self.mode * x).cos() * (self.mode * y).sin()
            v0 = (self.mode * x).sin() * (self.mode * y).cos()
            u = u0 + (t / self.tmax) * raw[:, 0]
            v = v0 + (t / self.tmax) * raw[:, 1]
        else:
            u, v = raw[:, 0], raw[:, 1]
        # The anchor depends on t, but has no spatial dependence. Autodiff
        # therefore preserves p_x and p_y exactly under this gauge choice.
        anchor = torch.stack((torch.zeros_like(t), torch.zeros_like(t), t), dim=1)
        p = raw[:, 2] - self.network(self._features(anchor))[:, 2]
        return torch.stack((u, v, p), dim=1)

    def get_config(self) -> dict:
        """Return only serializable constructor parameters."""
        return {
            "nu": self.nu,
            "tmax": self.tmax,
            "mode": self.mode,
            "width": self.width,
            "depth": self.depth,
            "hard_ic": self.hard_ic,
        }
