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


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, int, int, int]
    dtype: torch.dtype
    device: torch.device


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
        if num_envs != 1:
            raise ValueError("native BlockLab backend currently supports only num_envs == 1")
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
        self._obs: torch.Tensor | None = None
        self._rng = random.Random(seed)

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec((1, 4, self.height, self.width), torch.uint8, self.device)

    def reset(self, seed: int = 1):
        self._env.reset(seed)
        self._obs = self._observation_tensor()
        return self._obs

    def step(self, actions: Any):
        if self._obs is None:
            result = self._env.step_discrete(self._discrete_action_id(actions))
            self._obs = self._observation_tensor()
        else:
            result = self._env.step_discrete_sync(self._discrete_action_id(actions))
        obs = self._obs
        return obs, result.reward, result.terminated, result.truncated, {
            "native": True,
            "zero_copy_ready": obs.data_ptr(),
            "observation_handle": result.observation.handle,
            "observation_version": result.observation.version,
        }

    def _observation_tensor(self) -> torch.Tensor:
        return torch.utils.dlpack.from_dlpack(self._env.observation_dlpack())

    def sample_action(self) -> int:
        return self._rng.randrange(7)

    @staticmethod
    def _discrete_action_id(actions: Any) -> int:
        if isinstance(actions, int):
            return actions
        return int(torch.as_tensor(actions, device="cpu").reshape(-1)[0].item())


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

    def reset(self, *, seed: int = 1):
        obs = self.backend.reset(seed)
        return obs, {"native": True, "zero_copy_ready": obs.data_ptr()}

    def step(self, actions: Any):
        return self.backend.step(actions)

    def sample_actions(self) -> int:
        return self.backend.sample_action()


def make(**kwargs: Any) -> BlockLabEnv:
    return BlockLabEnv(**kwargs)
