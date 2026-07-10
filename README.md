# BlockLab RL Environment

BlockLab is an experimental Minecraft-like reinforcement learning environment written in C++20/CUDA/Vulkan. The project explores a GPU-first simulation and observation pipeline: voxel world generation, lighting, offscreen rendering, and CUDA/PyTorch observation handoff are designed to stay on the GPU.

## Current status

<img width="320" alt="screenshot" src="https://github.com/user-attachments/assets/43d78423-8797-45e9-a6e2-99d59fda3800" />
<img width="320" alt="screenshot" src="https://github.com/user-attachments/assets/2f2cd40e-ee57-4f02-9cff-8da7a1147831" />

BlockLab currently provides:
- Fully asynchronous GPU pipeline for world generation, light computation, rendering, and observation conversion.
- Voxel lighting system: day/night sky lighting with sunrise/sunset transitions, directional sky shading, and block light propagation from torches.
- Batched offscreen Vulkan rendering for RL observations.
- Zero-copy CUDA/PyTorch integration for float NCHW observation tensors.
- Memory-efficient voxel world representation with sparse modifications.
- Agent physics with collisions, jumping, camera control, digging, and block placement.
- Basic NPC support.

## Short-term roadmap

Near-term work is focused on making the environment less prototype-like:

- ~~Inventory and dropped items.~~ ✅
- Interaction with NPC.
- NPC spawn.
- Health / hunger management.
- More world content: trees, water, lava, and additional block/entity types.
- More complete terrain generation.
- Tools.
- Craft.
- Smooth vertex lighting for less blocky light gradients.
- Sun and moon rendering.
  
## Build

Requirements:

- CMake 3.24+
- C++20 compiler
- CUDA toolkit
- Vulkan loader and headers
- GLFW 3
- `glslc` for HLSL-to-SPIR-V shader compilation

```bash
cmake -S . -B build
cmake --build build -j 16
./build/blocklab
```

## Python API

The build produces a native Python extension module, `blocklab._native`, and the `blocklab` Python package wraps it with a small environment API.

The main entry point is `blocklab.BlockLabEnv`:

```python
import blocklab

env = blocklab.BlockLabEnv(num_envs=128, width=160, height=90, device="cuda")

obs, info = env.reset(seed=1)
obs_tensor = obs.to_tensor()  # CUDA float32 tensor, shape: (N, 3, H, W)

actions = env.sample_actions()
obs, reward, terminated, truncated, info = env.step(actions)
```

`BlockLabEnv` methods:

- `reset(seed=1)` - resets all environments and returns `(observation, info)`.
- `step(actions)` - advances the batch and returns `(observation, reward, terminated, truncated, info)`.
- `sample_actions()` - returns a random discrete action or a batch of random discrete actions, depending on `num_envs`.

Observations are represented by `BlockLabObservation`. They are converted to PyTorch tensors through DLPack:

```python
tensor = obs.to_tensor()
```

The tensor is CUDA-backed, `float32`, and laid out as `NCHW`: `(num_envs, 3, height, width)`.

Actions can be passed either as `AgentAction` objects or as discrete action IDs:

- `0` - forward
- `1` - backward
- `2` - left
- `3` - right
- `4` - jump
- `5` - attack
- `6` - use selected hotbar slot

Examples:

- `python/blocklab/examples/benchmark.py`
- `python/blocklab/examples/demo_classifier.py`
- `python/blocklab/examples/visualize.py`

## Tools

- `blocklab` runs an interactive debug environment.
- `blocklab_benchmark` measures native environment throughput.
- `python -m blocklab.examples.benchmark` measures the Python/PyTorch training-facing path.

## Benchmark

```bash
./build/blocklab_benchmark
./build/blocklab_benchmark --seconds 30
./build/blocklab_benchmark --steps 1000000
./build/blocklab_benchmark --initial-overrides 10000
./build/blocklab_benchmark --visualize
```

By default, the benchmark renders GPU observations offscreen without presenting frames to a window. Use `--visualize` to open a debug window that samples the current benchmark state without throttling simulation to real time.

## Interactive debug environment

```bash
./build/blocklab
```

### Controls

- `WASD` - move
- `Space` - jump
- Mouse or arrow keys - look around
- Left mouse button - attack
- Right mouse button - use selected hotbar slot
- `1`-`9` - select hotbar slot
- `Tab` - toggle mouse capture
- `~` - toggle the frame limiter
- `R` - reset
- `Esc` - quit
