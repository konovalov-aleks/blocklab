#pragma once

#include <blocklab/utility/Error.h>

#include <cuda_runtime.h>

namespace blocklab {

#if defined(__CUDACC__)
#define BLOCKLAB_HOST_DEVICE __host__ __device__
#else
#define BLOCKLAB_HOST_DEVICE
#endif

inline void cudaCheck(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) [[unlikely]]
        fatalError(operation, " failed: ", cudaGetErrorString(result));
}

#if !defined(__CUDACC__) || defined(CUDA_CPU_FALLBACK_MODE)
inline float fast_sinf(float angle) { return __builtin_sinf(angle); }
inline float fast_cosf(float angle) { return __builtin_cosf(angle); }
#else
inline __device__ float fast_sinf(float angle) { return __sinf(angle); }
inline __device__ float fast_cosf(float angle) { return __cosf(angle); }
#endif

} // namespace blocklab
