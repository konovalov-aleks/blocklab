import pytest
import torch

import blocklab
from blocklab.visualization import observation_grid


def test_observation_grid_returns_cpu_image_tensor():
    obs = torch.linspace(0.0, 1.0, 4 * 3 * 8 * 16, dtype=torch.float32).reshape(4, 3, 8, 16)
    grid = observation_grid(obs, max_images=4)
    assert grid.shape == (3, 16, 32)
    assert grid.device.type == "cpu"
    assert torch.all(grid >= 0.0)
    assert torch.all(grid <= 1.0)
