#include "blocklab/CudaObservation.h"

#include "blocklab/CudaHelpers.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace blocklab {
namespace {

    __global__ void rgba8ToFloatNchwKernel(
        const uchar4* rgba, float* nchw, uint32_t batchSize, uint32_t width, uint32_t height)
    {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        const uint32_t pixelsPerImage = width * height;
        const uint32_t totalPixels = batchSize * pixelsPerImage;
        if (index >= totalPixels)
            return;

        const uint32_t image = index / pixelsPerImage;
        const uint32_t pixel = index - image * pixelsPerImage;
        const uchar4 sample = rgba[index];
        const uint32_t planeBase = image * 3U * pixelsPerImage;
        constexpr float Inv255 = 1.0f / 255.0f;
        nchw[planeBase + pixel] = static_cast<float>(sample.x) * Inv255;
        nchw[planeBase + pixelsPerImage + pixel] = static_cast<float>(sample.y) * Inv255;
        nchw[planeBase + 2U * pixelsPerImage + pixel] = static_cast<float>(sample.z) * Inv255;
    }

} // namespace

void convertRgba8ToFloatNchw(
    const void* rgba8, float* nchw, uint32_t batchSize, uint32_t width, uint32_t height, uintptr_t streamHandle)
{
    const uint32_t totalPixels = batchSize * width * height;
    constexpr uint32_t ThreadCount = 256;
    auto stream = reinterpret_cast<cudaStream_t>(streamHandle);
    rgba8ToFloatNchwKernel<<<(totalPixels + ThreadCount - 1U) / ThreadCount, ThreadCount, 0, stream>>>(
        static_cast<const uchar4*>(rgba8), nchw, batchSize, width, height);
    cudaCheck(cudaGetLastError(), "rgba8ToFloatNchwKernel");
}

} // namespace blocklab
