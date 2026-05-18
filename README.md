# BlockLab RL Environment

A minimal Minecraft-style voxel prototype written in C++20, CMake, and SDL3. The MVP is designed as a future gym-like reinforcement learning environment with `reset`, `step`, observation, reward, an agent, a deterministic procedural world, and a fixed simulation tick.

## MVP Features

- CPU simulation of an infinite procedural voxel world with sparse block overrides.
- Procedural terrain made from `Grass`, `Dirt`, and `Stone`.
- Agent physics with collisions, jumping, and yaw/pitch camera control.
- Actions for movement, jumping, digging blocks, and placing blocks.
- RL observation is modeled as a handle/view over a render target, without copying pixels through the CPU.
- SDL3 application for interactive environment inspection.

## Build

Requirements:

- CMake 3.24+
- Clang with C++20 support

SDL3, SDL_shadercross, and GLM are fetched by CMake by default:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++
cmake --build build
./build/blocklab
```

To build against an already installed SDL3 package instead, pass `-DUSE_SYSTEM_SDL3=ON`:

```bash
cmake -S . -B build -DCMAKE_CXX_COMPILER=clang++ -DUSE_SYSTEM_SDL3=ON
cmake --build build
```

`SDL_shadercross` is a separate SDL project, not part of the SDL3 core library. It is also fetched by default; pass `-DUSE_SYSTEM_SDLSHADERCROSS=ON` to use an installed package instead. Compute shaders are authored in HLSL under `shaders/` and compiled during the build with the `shadercross` target. The build emits MSL on Apple platforms, DXIL on Windows, and SPIR-V elsewhere.

By default, the window and render target are `320x180`. The resolution can be set from the command line:

```bash
./build/blocklab --resolution 640x360
./build/blocklab --resolution=1280x720
./build/blocklab --unlocked --visual-fps 120
```

## Benchmark

The `blocklab_benchmark` CLI executable runs a simple agent in the environment and prints simulation performance:

```bash
./build/blocklab_benchmark
./build/blocklab_benchmark --seconds 30
./build/blocklab_benchmark --steps 1000000 --report-interval 5
./build/blocklab_benchmark --action-steps 60:240
./build/blocklab_benchmark --initial-overrides 10000
```

By default, no window is opened, but the environment still renders observation into a hidden SDL3 GPU target and returns a texture handle. The benchmark also seeds the world with `1000` clustered block overrides after each reset, so long-running measurements include non-empty world-diff lookup costs. Use `--initial-overrides 0` for the old clean-world baseline. Add `--no-render` to compare against pure CPU simulation.

For a visual smoke check, enable the SDL3 GPU viewer:

```bash
./build/blocklab_benchmark --visualize --resolution 640x360
./build/blocklab_benchmark --visualize --visual-fps 30
```

The final report includes `steps_per_second`, average reward, episode count, accumulated dig/place counters, and the requested/applied/current override counts. The benchmark agent holds only movement and camera rotation for several steps to test sustained motion; `dig`, `place`, and `jump` remain one-tick actions.

If you opt into system SDL3 on macOS, Homebrew is usually enough:

```bash
brew install sdl3
```

## Controls

- `WASD` - move
- `Space` - jump
- mouse - rotate camera
- `Left/Right` - rotate camera with keyboard
- `Q` - dig the block in front of the agent
- `E` - place a block
- `R` - reset
- `Esc` - quit

The viewer prints `fps`, `sim_steps/s`, `total_steps`, and `observation_version` to the console. By default, simulation runs at a fixed 60 Hz and window presentation is limited by `--visual-fps`. The `--unlocked` flag removes fixed timestep pacing and runs `step()` as fast as the current render-observation path allows.

## Architecture

- `blocklab::World` computes base blocks procedurally from a seed and stores only modified blocks as overrides.
- `blocklab::Agent` owns physics and block interaction.
- `blocklab::Environment` exposes the gym-like API: `reset()` and `step(action)`.
- `blocklab::Renderer` visualizes state only and does not own environment logic.

Important: observation does not expose block coordinates, a local voxel cube, agent position, or velocity. Those values are available inside C++ for simulation, reward/debug viewer, and future `info`, but they are not part of the agent observation. This is intentional: the model is meant to learn from images. `Environment` can attach an `ObservationRenderer`; after that, `reset()` and `step()` update the environment observation and return a handle. In the current SDL backend, compute shaders generate the active procedural region on the GPU, apply sparse overrides, and build a compact mesh buffer. A graphics pass then renders that mesh into a persistent `SDL_GPUTexture` observation target through the SDL3 GPU API. For the production RL backend, the same contract should become a CUDA/DLPack/CUDA-array-interface view instead of a CPU `std::vector<uint8_t>`.

This separation is intended to support future headless runs, batched environment instances, and replacement of the internal world/physics backend with CUDA.

## API Example

```cpp
blocklab::Environment env(4);
// Without a renderer backend, observation is empty.
// With a renderer backend, reset/step return a GPU handle.
blocklab::Observation obs = env.reset();

blocklab::AgentAction action;
action.forward = 1.0f;
action.yawDelta = 0.01f;

blocklab::StepResult result = env.step(action);
// With the SDL backend, result.observation.device == ObservationDevice::SdlGpuTexture.
// Without a renderer backend, simulation-only mode returns ObservationDevice::None.
```

## Current Limitations

This is an MVP, not a full Minecraft implementation. The SDL backend now generates the active procedural region on the GPU, applies sparse overrides, builds a GPU-side mesh buffer, and rasterizes it into the observation texture. CPU simulation is still authoritative for physics and sparse overrides, and active overrides are uploaded to the GPU as a compact edit list. The next major step is to move physics and batched environment stepping toward a GPU-friendly layout as well.
