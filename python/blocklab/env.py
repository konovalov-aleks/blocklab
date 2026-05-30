from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import random

import torch

try:
    from . import _native
except ImportError as exc:
    _native = None
    _native_import_error = exc
else:
    _native_import_error = None

AgentAction = _native.AgentAction if _native is not None else None


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    device: torch.device


class BlockLabObservation:
    """Lazy DLPack producer for an observation frame.

    The Vulkan render is submitted by reset()/step(), but synchronization is
    delayed until a consumer calls torch.from_dlpack(observation).
    """

    def __init__(self, backend: "NativeBlockLabBackend", *, frame_indices: list[int], version: int) -> None:
        self._backend = backend
        self._frame_indices = frame_indices
        self.version = version
        self.shape = backend.observation_spec.shape
        self.dtype = backend.observation_spec.dtype
        self.device = backend.observation_spec.device

    def __dlpack_device__(self) -> tuple[int, int]:
        if self.device.type != "cuda":
            raise RuntimeError("BlockLab observations are CUDA-only")
        return (2, 0 if self.device.index is None else self.device.index)

    def __dlpack__(self, stream: int | None = None):
        return self._backend._env.observation_dlpack(self._frame_indices, 0 if stream is None else stream)

    def to_tensor(self) -> torch.Tensor:
        return torch.from_dlpack(self)


class NativeBlockLabBackend:
    """Thin Python wrapper over the pybind11 C++ BlockLab environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        width: int,
        height: int,
        device: str | torch.device,
        seed: int = 1,
        world_radius_chunks: int = 3,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if _native is None:
            raise RuntimeError(
                "blocklab._native is not built. Configure and build the CMake target `_native` first."
            ) from _native_import_error
        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            raise RuntimeError("native BlockLab backend is CUDA-only")

        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.device = resolved_device
        self._env = _native.NativeEnvironment(
            num_envs=num_envs,
            width=width,
            height=height,
            world_radius_chunks=world_radius_chunks,
            seed=seed,
        )
        self._rng = random.Random(seed)

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec((self.num_envs, 3, self.height, self.width), torch.float32, self.device)

    def reset(self, seed: int = 1):
        info = self._env.reset(seed)
        return self._observation_ref(info)

    def step(self, actions: Any):
        result = self._env.step(self._normalize_actions(actions))
        obs = self._observation_ref(result)
        reward = torch.as_tensor(result["reward"], device=self.device, dtype=torch.float32)
        terminated = torch.as_tensor(result["terminated"], device=self.device, dtype=torch.bool)
        truncated = torch.as_tensor(result["truncated"], device=self.device, dtype=torch.bool)
        return obs, reward, terminated, truncated, {
            "native": True,
            "terminated": result["terminated"],
            "truncated": result["truncated"],
            "observation_version": result["version"],
            "observation_frame_indices": result["frame_indices"],
        }

    def _observation_ref(self, info: Any) -> BlockLabObservation:
        return BlockLabObservation(
            self, frame_indices=list(info["frame_indices"]), version=int(info["version"])
        )

    def sample_action(self) -> int:
        if self.num_envs == 1:
            return self._rng.randrange(7)
        return torch.tensor([self._rng.randrange(7) for _ in range(self.num_envs)], dtype=torch.int32)

    @staticmethod
    def _normalize_actions(actions: Any) -> Any:
        if isinstance(actions, int):
            return _action_from_discrete(actions)
        if _native is not None and isinstance(actions, _native.AgentAction):
            return actions
        if (
            _native is not None
            and isinstance(actions, (list, tuple))
            and actions
            and isinstance(actions[0], _native.AgentAction)
        ):
            return actions
        actions = torch.as_tensor(actions, device="cpu", dtype=torch.int32).reshape(-1)
        if actions.numel() == 1:
            return _action_from_discrete(int(actions[0].item()))
        return [_action_from_discrete(int(action.item())) for action in actions]


class BlockLabEnv:
    def __init__(
        self,
        *,
        num_envs: int = 1,
        width: int = 160,
        height: int = 90,
        device: str | torch.device | None = None,
        seed: int = 1,
        backend: NativeBlockLabBackend | None = None,
        world_radius_chunks: int = 3,
    ) -> None:
        resolved_device = torch.device(device) if device is not None else torch.device("cuda")
        self.backend = backend or NativeBlockLabBackend(
            num_envs=num_envs,
            width=width,
            height=height,
            device=resolved_device,
            seed=seed,
            world_radius_chunks=world_radius_chunks,
        )
        self.num_envs = self.backend.num_envs
        self.single_action_space_n = 7
        self.observation_spec = self.backend.observation_spec
        self._done = False

    def reset(self, *, seed: int = 1):
        obs = self.backend.reset(seed)
        self._done = False
        return obs, {"native": True, "observation_version": obs.version}

    def step(self, actions: Any):
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")
        obs, reward, terminated, truncated, info = self.backend.step(actions)
        self._done = any(done or cut for done, cut in zip(info["terminated"], info["truncated"]))
        return obs, reward, terminated, truncated, info

    def sample_actions(self) -> int:
        return self.backend.sample_action()


def make(**kwargs: Any) -> BlockLabEnv:
    return BlockLabEnv(**kwargs)


def _action_from_discrete(action_id: int):
    if _native is None:
        raise RuntimeError("blocklab._native is not built") from _native_import_error
    action = _native.AgentAction()
    if action_id == 0:
        action.forward = 1.0
    elif action_id == 1:
        action.forward = -1.0
    elif action_id == 2:
        action.right = -1.0
    elif action_id == 3:
        action.right = 1.0
    elif action_id == 4:
        action.jump = True
    elif action_id == 5:
        action.dig = True
    elif action_id == 6:
        action.place = True
    else:
        raise ValueError(f"unknown discrete action: {action_id}")
    return action
