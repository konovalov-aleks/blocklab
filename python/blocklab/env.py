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


class BlockLabObservation:
    """Lazy DLPack producer for an observation frame.

    The Vulkan render is submitted by reset()/step(), but synchronization is
    delayed until a consumer calls torch.from_dlpack(observation).
    """

    def __init__(self, backend: "NativeBlockLabBackend", *, frame_index: int, version: int) -> None:
        self._backend = backend
        self._frame_index = frame_index
        self.version = version
        self.shape = backend.observation_spec.shape
        self.dtype = backend.observation_spec.dtype
        self.device = backend.observation_spec.device

    def __dlpack_device__(self) -> tuple[int, int]:
        if self.device.type != "cuda":
            raise RuntimeError("BlockLab observations are CUDA-only")
        return (2, 0 if self.device.index is None else self.device.index)

    def __dlpack__(self, stream: int | None = None):
        return self._backend._env.observation_dlpack(self._frame_index)

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
        self._rng = random.Random(seed)

    @property
    def observation_spec(self) -> ObservationSpec:
        return ObservationSpec((1, 4, self.height, self.width), torch.uint8, self.device)

    def reset(self, seed: int = 1):
        observation = self._env.reset(seed)
        return self._observation_ref(observation)

    def step(self, actions: Any):
        result = self._env.step(self._discrete_action_id(actions))
        obs = self._observation_ref(result.observation)
        return obs, result.reward, result.terminated, result.truncated, {
            "native": True,
            "observation_handle": result.observation.handle,
            "observation_version": result.observation.version,
            "observation_frame_index": obs._frame_index,
        }

    def _observation_ref(self, observation: Any) -> BlockLabObservation:
        return BlockLabObservation(
            self, frame_index=self._env.last_observation_frame_index(), version=observation.version
        )

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
        return obs, {"native": True, "observation_version": obs.version}

    def step(self, actions: Any):
        return self.backend.step(actions)

    def sample_actions(self) -> int:
        return self.backend.sample_action()


def make(**kwargs: Any) -> BlockLabEnv:
    return BlockLabEnv(**kwargs)
