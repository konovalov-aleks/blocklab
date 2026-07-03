# BlockLab RL Environment

BlockLab is a high-performance Minecraft-like reinforcement learning environment written in C++20. It combines a procedural voxel world, lightweight agent/NPC simulation, CUDA terrain mesh generation, and Vulkan rendering for GPU-backed observations.

The project is optimized for fast training loops: benchmarks render observations offscreen by default, avoid swapchain presentation unless explicitly requested, and keep render output on the GPU for downstream consumers.

## Features

At the current stage, BlockLab supports:

- Infinite procedural voxel world with sparse block overrides.
- Minecraft-like terrain made from grass, dirt, and stone blocks.
- Agent physics with collisions, jumping, yaw/pitch camera control, digging, and placing.
- Pig NPCs with simple movement behavior.
- CUDA terrain mesh generation.
- Vulkan mesh renderer with offscreen observation images.
- Optional GLFW/Vulkan debug window via `--visualize`.
- Benchmark mode focused on high steps-per-second throughput.

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

## Benchmark

```bash
./build/blocklab_benchmark
./build/blocklab_benchmark --seconds 30
./build/blocklab_benchmark --steps 1000000
./build/blocklab_benchmark --initial-overrides 10000
./build/blocklab_benchmark --visualize
```

By default, the benchmark renders GPU observations offscreen without presenting frames to a window. Use `--visualize` to open a debug window that samples the current benchmark state without throttling simulation to real time.

## Controls

- `WASD` - move
- `Space` - jump
- Mouse or arrow keys - look around
- `Q` - dig
- `E` - place
- `1` - select torch placement
- `2` - select dirt placement
- `3` - select stone placement
- `Tab` - toggle the frame limiter
- `R` - reset
- `Esc` - quit

## Architecture

- `blocklab::World` computes procedural base terrain and stores modified blocks as sparse overrides.
- `blocklab::Agent` handles movement, camera state, collisions, and block interaction.
- `blocklab::Environment` exposes `reset()` and `step(action)` for RL-style control loops.
- `blocklab::CudaTerrainMeshBuilder` builds visible terrain mesh data on CUDA.
- `blocklab::Renderer` owns Vulkan resources, offscreen observation images, entity instancing, and optional presentation.
