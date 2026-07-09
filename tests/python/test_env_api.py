import pytest
import torch

import blocklab
from blocklab.visualization import observation_grid


def test_native_inventory_symbols_are_reexported():
    if blocklab.AgentAction is None:
        pytest.skip("native BlockLab module is not built")

    assert blocklab.Item is not None
    assert blocklab.ItemType is not None
    assert blocklab.ItemType.Torch is not None


def test_native_observation_exposes_inventory_state():
    if blocklab.AgentAction is None:
        pytest.skip("native BlockLab module is not built")

    env = blocklab.make(num_envs=1, width=32, height=18, max_steps=None)
    obs, _ = env.reset(seed=1)

    inventory = obs.inventories[0]
    assert inventory.active_hotbar_slot == 0
    assert inventory.hotbar_slots[0].type == blocklab.ItemType.Torch
    assert inventory.hotbar_slots[0].count == 64
    assert inventory.hotbar_slots[1].type == blocklab.ItemType.Dirt
    assert inventory.hotbar_slots[2].type == blocklab.ItemType.Stone

    action = blocklab.AgentAction()
    action.active_hotbar_slot = 2
    obs, *_ = env.step(action)
    assert obs.inventories[0].active_hotbar_slot == 2


def test_observation_grid_returns_cpu_image_tensor():
    obs = torch.linspace(0.0, 1.0, 4 * 3 * 8 * 16, dtype=torch.float32).reshape(4, 3, 8, 16)
    grid = observation_grid(obs, max_images=4)
    assert grid.shape == (3, 16, 32)
    assert grid.device.type == "cpu"
    assert torch.all(grid >= 0.0)
    assert torch.all(grid <= 1.0)
