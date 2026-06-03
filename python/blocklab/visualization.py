from __future__ import annotations

import math
from typing import Any

import torch


def observation_grid(obs: torch.Tensor, *, max_images: int = 16) -> torch.Tensor:
    if obs.ndim != 4:
        raise ValueError(f"expected NCHW observation tensor, got shape {tuple(obs.shape)}")
    if obs.shape[1] not in (1, 3, 4):
        raise ValueError(f"expected 1, 3, or 4 channels, got {obs.shape[1]}")

    images = obs[:max_images].detach()
    if images.dtype == torch.uint8:
        images = images.float().mul_(1.0 / 255.0)
    else:
        images = images.float().clamp_(0.0, 1.0)
    images = images.cpu()
    count, channels, height, width = images.shape
    columns = math.ceil(math.sqrt(count))
    rows = math.ceil(count / columns)
    grid = torch.zeros(channels, rows * height, columns * width)
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        grid[:, row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return grid


def show_observation(obs: torch.Tensor, *, max_images: int = 16, title: str = "BlockLab observation") -> Any:
    import matplotlib.pyplot as plt

    grid = observation_grid(obs, max_images=max_images)
    image = grid.permute(1, 2, 0).numpy()
    if image.shape[2] == 1:
        image = image[:, :, 0]
    elif image.shape[2] == 4:
        image = image[:, :, :3]

    plt.figure(title)
    plt.imshow(image, cmap="gray" if grid.shape[0] == 1 else None, interpolation="nearest")
    plt.axis("off")
    plt.tight_layout()
    return plt.gcf()
