from __future__ import annotations

import argparse

import torch
from torch import nn

import blocklab


MAX_STEPS = 60 * 60 * 80  # 4 game days.


class TinyPolicy(nn.Module):
    def __init__(self, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    env = blocklab.make(
        num_envs=args.num_envs,
        width=args.width,
        height=args.height,
        device=args.device,
        max_steps=MAX_STEPS,
    )
    obs, _ = env.reset()
    obs = torch.from_dlpack(obs)
    policy = TinyPolicy(env.single_action_space_n).to(obs.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    for step in range(args.steps):
        logits = policy(obs)
        target = torch.round(obs[:, 2, 0, 0] * 3.0).long()
        loss = nn.functional.cross_entropy(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            actions = logits.argmax(dim=1)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            del terminated, truncated, info
        obs = torch.from_dlpack(next_obs).detach()

        if step % 50 == 0:
            print(f"step={step} loss={loss.item():.4f} reward={reward.mean().item():.3f} device={obs.device}")


if __name__ == "__main__":
    main()
