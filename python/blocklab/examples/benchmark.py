from __future__ import annotations

import argparse
import time

import blocklab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup-steps", type=int, default=128)
    args = parser.parse_args()

    env = blocklab.make(num_envs=args.num_envs, width=args.width, height=args.height, device=args.device)
    obs, _ = env.reset()
    info = {"zero_copy_ready": obs.data_ptr()}

    for _ in range(args.warmup_steps):
        actions = env.sample_actions()
        obs, reward, terminated, truncated, info = env.step(actions)
        del reward, terminated, truncated

    started = time.perf_counter()
    for _ in range(args.steps):
        actions = env.sample_actions()
        obs, reward, terminated, truncated, info = env.step(actions)
        del reward, terminated, truncated
    elapsed = time.perf_counter() - started

    print(f"device: {env.observation_spec.device}")
    print(f"observation_shape: {env.observation_spec.shape}")
    print(f"observation_dtype: {obs.dtype}")
    print(f"observation_ptr_stable: {obs.data_ptr() == info['zero_copy_ready']}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"steps_per_second: {args.steps * args.num_envs / elapsed:.0f}")


if __name__ == "__main__":
    main()
