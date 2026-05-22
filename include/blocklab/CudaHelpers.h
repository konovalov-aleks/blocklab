#pragma once

#include "Error.h"

#include <cuda_runtime.h>

inline void cudaCheck(cudaError_t result, const char* operation)
{
    if (result != cudaSuccess) [[unlikely]]
        fatalError(operation, "failed:", cudaGetErrorString(result));
}
