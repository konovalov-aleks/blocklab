#pragma once

#include <cuda_runtime.h>

namespace blocklab {

struct KernelLaunchArgs {
    dim3 gridDim;
    dim3 blockDim;
    unsigned sharedMemBytes = 0;
    cudaStream_t stream = nullptr;
};

} // namespace blocklab
