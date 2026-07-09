from __future__ import annotations

import argparse
import random
import time

import torch

import blocklab


DEFAULT_MAX_STEPS = 60 * 60 * 80  # 4 game days.


class RandomAgent:
    def __init__(self, *, min_action_steps: int, max_action_steps: int) -> None:
        self._min_action_steps = min_action_steps
        self._max_action_steps = max_action_steps
        self._remaining_steps = 0
        self._forward = 0.0
        self._right = 0.0
        self._yaw_delta = 0.0
        self._pitch_delta = 0.0

    def next_action(self, rng: random.Random):
        if self._remaining_steps == 0:
            self._remaining_steps = rng.randint(self._min_action_steps, self._max_action_steps)
            self._forward = rng.uniform(0.45, 1.0) if rng.random() < 0.82 else 0.0
            self._right = rng.uniform(-0.35, 0.35)
            self._yaw_delta = rng.gauss(0.0, 0.012)
            self._pitch_delta = rng.gauss(0.0, 0.012) * 0.25

        action = blocklab.AgentAction()
        action.forward = self._forward
        action.right = self._right
        action.yaw_delta = self._yaw_delta
        action.pitch_delta = self._pitch_delta
        action.jump = rng.random() < 0.004
        action.attack = rng.random() < 0.004
        action.use = (not action.attack) and rng.random() < 0.003
        self._remaining_steps -= 1
        return action


def parse_action_steps(value: str) -> tuple[int, int]:
    parts = value.split(":", maxsplit=1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected MIN:MAX")
    min_steps = int(parts[0])
    max_steps = int(parts[1])
    if min_steps <= 0 or max_steps < min_steps:
        raise argparse.ArgumentTypeError("expected 0 < MIN <= MAX")
    return min_steps, max_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warmup-steps", type=int, default=128)
    parser.add_argument("--report-interval", type=float, default=1.0)
    parser.add_argument("--action-steps", type=parse_action_steps, default=(20, 160))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--materialize-observation", action="store_true")
    args = parser.parse_args()

    env = blocklab.make(
        num_envs=args.num_envs,
        width=args.width,
        height=args.height,
        device=args.device,
        seed=args.seed,
        max_steps=args.max_steps,
    )
    rng = random.Random(args.seed)
    random_agents = [
        RandomAgent(min_action_steps=args.action_steps[0], max_action_steps=args.action_steps[1])
        for _ in range(args.num_envs)
    ]
    obs, _ = env.reset()
    tensor = torch.from_dlpack(obs) if args.materialize_observation else None

    for step in range(args.warmup_steps):
        actions = [agent.next_action(rng) for agent in random_agents]
        obs, reward, terminated, truncated, info = env.step(actions)
        if args.materialize_observation:
            tensor = torch.from_dlpack(obs)
        if any(done or cut for done, cut in zip(info["terminated"], info["truncated"])):
            obs, _ = env.reset(seed=args.seed + step + 1)
            if args.materialize_observation:
                tensor = torch.from_dlpack(obs)

    started = time.perf_counter()
    last_report_at = started
    last_report_step = 0
    total_reward = 0.0
    total_episodes = 0
    step = 0
    while True:
        step += 1
        actions = [agent.next_action(rng) for agent in random_agents]
        obs, reward, terminated, truncated, info = env.step(actions)
        if args.materialize_observation:
            tensor = torch.from_dlpack(obs)
        total_reward += float(sum(info["reward"]))
        episode_count = sum(done or cut for done, cut in zip(info["terminated"], info["truncated"]))
        total_episodes += episode_count
        if episode_count > 0:
            obs, _ = env.reset(seed=args.seed + total_episodes)
            if args.materialize_observation:
                tensor = torch.from_dlpack(obs)
        now = time.perf_counter()
        if args.report_interval > 0.0 and now - last_report_at >= args.report_interval:
            interval = now - last_report_at
            interval_steps = (step - last_report_step) * args.num_envs
            total_steps = step * args.num_envs
            avg_reward = total_reward / max(1, total_steps)
            print(
                f"iterations={step} steps={total_steps} elapsed={now - started:.2f}s "
                f"steps/s={interval_steps / interval:.0f} avg_reward={avg_reward:.4f} episodes={total_episodes}",
                flush=True,
            )
            last_report_at = now
            last_report_step = step
        if args.steps > 0:
            if step >= args.steps:
                break
        elif now - started >= args.seconds:
            break
    elapsed = time.perf_counter() - started
    total_steps = step * args.num_envs

    print(f"device: {env.observation_spec.device}")
    print(f"observation_shape: {env.observation_spec.shape}")
    print(f"observation_dtype: {obs.dtype}")
    print(f"materialize_observation: {args.materialize_observation}")
    if tensor is not None:
        print(f"materialized_tensor_ptr: {tensor.data_ptr()}")
    print(f"warmup_steps: {args.warmup_steps}")
    print(f"iterations: {step}")
    print(f"total_steps: {total_steps}")
    print(f"avg_reward: {total_reward / max(1, total_steps):.6f}")
    print(f"episodes: {total_episodes}")
    print(f"steps_per_second: {total_steps / elapsed:.0f}")


if __name__ == "__main__":
    main()
