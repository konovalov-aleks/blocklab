#include <blocklab/gpu/cuda/CudaHelpers.h>

#include <cuda_runtime.h>

#include <cassert>
#include <cstdint>

namespace blocklab {
namespace {

    __global__ void floatNchwToImageKernel(uchar4* rgba, const float* nchw, std::uint32_t batchSize,
        std::uint32_t width, std::uint32_t height, std::uint32_t imageGridWidth, std::uint32_t imageGridHeight,
        bool outputBGRA)
    {
        const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;

        const std::uint32_t pixelsPerRow = imageGridWidth * width;
        const std::uint32_t pixelsPerCol = imageGridHeight * height;
        const std::uint32_t totalPixels = pixelsPerRow * pixelsPerCol;
        if (index >= totalPixels)
            return;

        const std::uint32_t dstY = index / pixelsPerRow;
        const std::uint32_t dstX = index - dstY * pixelsPerRow;

        const std::uint32_t tileX = dstX / width;
        const std::uint32_t tileY = dstY / height;

        const std::uint32_t batchIdx = tileX + imageGridWidth * tileY;
        const std::uint32_t srcX = dstX % width;
        const std::uint32_t srcY = dstY % height;

        const std::uint32_t pixelsPerSrcChannel = width * height;
        const std::uint32_t pixelsPerSrcImage = pixelsPerSrcChannel * 3;
        const std::uint32_t srcBasePlaneOffset = batchIdx * pixelsPerSrcImage + srcY * width + srcX;

        const bool nonEmpty = batchIdx < batchSize;
        const std::uint8_t r = nonEmpty ? nchw[srcBasePlaneOffset + 0 * pixelsPerSrcChannel] * 255.0f : 91;
        const std::uint8_t g = nonEmpty ? nchw[srcBasePlaneOffset + 1 * pixelsPerSrcChannel] * 255.0f : 60;
        const std::uint8_t b = nonEmpty ? nchw[srcBasePlaneOffset + 2 * pixelsPerSrcChannel] * 255.0f : 57;

        const uchar4 color = outputBGRA ? uchar4 { b, g, r, 255 } : uchar4 { r, g, b, 255 };
        rgba[index] = color;
    }

} // namespace

void enqueueObservationConversionForDisplay(cudaStream_t stream, std::uint8_t* destination, float* source,
    std::uint32_t batchSize, std::uint32_t imageWidth, std::uint32_t imageHeight, std::uint32_t imageGridWidth,
    std::uint32_t imageGridHeight, bool outputBGRA /* true -> BGRA, false -> RGBA */)
{
    assert(batchSize <= imageGridWidth * imageGridHeight);

    const std::uint32_t totalPixels = imageGridWidth * imageWidth * imageGridHeight * imageHeight;
    constexpr std::uint32_t ThreadCount = 256;
    floatNchwToImageKernel<<<(totalPixels + ThreadCount - 1U) / ThreadCount, ThreadCount, 0, stream>>>(
        reinterpret_cast<uchar4*>(destination), source, batchSize, imageWidth, imageHeight, imageGridWidth,
        imageGridHeight, outputBGRA);
    cudaCheck(cudaGetLastError(), "floatNchwToImageKernel");
}

} // namespace blocklab
