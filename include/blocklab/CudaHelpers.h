#pragma once

#include "Error.h"

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

} // namespace blocklab
