#include <cuda_runtime.h>

#include <cstdint>

namespace blocklab {
namespace {
    /*
        __global__ void FloatNchwToImageKernel(const float* nchw, uchar4* rgba, std::uint32_t batchSize,
            std::uint32_t width, uint32_t height, std::uint32_t imageGridWidth)
        {
            const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t pixelsPerImage = width * height;
            const std::uint32_t totalPixels = batchSize * pixelsPerImage;
            if (index >= totalPixels)
                return;

            const std::uint32_t imageIdx = index / pixelsPerImage;
            const std::uint32_t imagePixelIdx = index - imageIdx * pixelsPerImage;

            std::uint32_t dstX = (imageIdx % imageGridWidth) * width + (imagePixelIdx % width);
            std::uint32_t dstY = (imageIdx / imageGridWidth) * height + (imagePixelIdx / width);

            uchar4* dst = &rgba[dstX + dstY * width];
            dst->x = nchw[index * 4] * 255.0f;
            dst->y = nchw[index * 4] * 255.0f;
            dst->z = nchw[index * 4] * 255.0f;
            dst->w = 127;
        }
    */
} // namespace

} // namespace blocklab
