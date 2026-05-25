from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

import torch

import blocklab
from blocklab.visualization import observation_grid


NON_INTERACTIVE_BACKENDS = {"agg", "cairo",
                            "pdf", "pgf", "ps", "svg", "template"}


def import_pyplot(*, backend: str | None):
    import matplotlib.pyplot as plt

    if backend is not None:
        plt.switch_backend(backend)
        return plt

    if matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS:
        return plt

    for candidate in ("QtAgg", "TkAgg"):
        try:
            plt.switch_backend(candidate)
            return plt
        except ImportError:
            continue

    return plt


def is_interactive_backend() -> bool:
    return matplotlib.get_backend().lower() not in NON_INTERACTIVE_BACKENDS


def draw_observation(plt, obs, *, max_images: int, title: str) -> None:
    grid = observation_grid(obs, max_images=max_images)
    image = grid.permute(1, 2, 0).numpy()
    plt.clf()
    plt.imshow(image, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--device", default=None)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--backend", default=None,
                        help="matplotlib GUI backend, for example QtAgg or TkAgg")
    parser.add_argument("--save", type=Path, default=None,
                        help="save the last observation image instead of opening a window")
    args = parser.parse_args()

    plt = import_pyplot(backend=args.backend)
    if not is_interactive_backend() and args.save is None:
        raise RuntimeError(
            f"matplotlib selected non-interactive backend {matplotlib.get_backend()!r}. "
            "Install a GUI backend, for example python3-tk for TkAgg or PyQt/PySide for QtAgg, "
            "set MPLBACKEND=TkAgg/QtAgg, or pass --save out.png."
        )

    env = blocklab.make(num_envs=args.num_envs, width=args.width,
                        height=args.height, device=args.device)
    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        raise RuntimeError(
            "visualize requires CUDA tensor observations. The native binding is active, "
            "but zero-copy DLPack/PyTorch observation export is not implemented yet."
        )
    plt.figure("BlockLab observation")
    draw_observation(plt, obs, max_images=args.num_envs, title="step 0")
    if is_interactive_backend():
        plt.pause(0.001)

    for step in range(args.steps):
        obs, reward, terminated, truncated, info = env.step(
            env.sample_actions())
        del reward, terminated, truncated, info
        if step % args.interval != 0:
            continue

        draw_observation(plt, obs, max_images=args.num_envs,
                         title=f"step {step + 1}")
        if is_interactive_backend():
            plt.pause(0.001)

    if args.save is not None:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
