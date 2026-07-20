#pragma once

#include "KernelLaunchArgs.h"

#include <blocklab/gpu/cuda/CudaHelpers.h>

#include <cuda_runtime.h>

#include <utility>

#ifdef CUDA_CPU_FALLBACK_MODE
#   include <gpu/cuda/cpu_fallback/CpuKernelLauncher.h>
#endif // CUDA_CPU_FALLBACK_MODE

namespace blocklab {

#ifndef CUDA_CPU_FALLBACK_MODE

#define CUDA_KERNEL __global__ void
#define CUDA_RETURN return

template <typename FnT, typename... ArgsT>
void launchKernel(const char* name, FnT fn, KernelLaunchArgs launchArgs, ArgsT&&... args)
{
    fn<<<launchArgs.gridDim, launchArgs.blockDim, launchArgs.sharedMemBytes, launchArgs.stream>>>(
        std::forward<ArgsT>(args)...
    );
    cudaCheck(cudaGetLastError(), name);
}

#define CUDA_EXTERN_SHARED(type, name) extern __shared__ type name[]

#else

#define CUDA_KERNEL CpuKernelLauncher::Task
#define CUDA_RETURN co_return

template <typename FnT, typename... ArgsT>
void launchKernel(const char* /* name */, FnT fn, KernelLaunchArgs launchArgs, ArgsT&&... args)
{
    CpuKernelLauncher().launchKernel(fn, launchArgs, std::forward<ArgsT>(args)...);
}

#define CUDA_EXTERN_SHARED(type, name) type* name = reinterpret_cast<type*>(CpuKernelLauncher::dynamicSharedMemory());

#endif // CUDA_CPU_FALLBACK_MODE

} // namespace blocklab
