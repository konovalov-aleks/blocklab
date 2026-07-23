#pragma once

#include <vulkan/vulkan.hpp>

#include <atomic>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

#define __device__
#define __host__
#define __shared__ thread_local
#define __syncthreads() co_await std::suspend_always();

namespace details {

    template <typename T>
    struct vec2 {
        T x;
        T y;

        T operator[](unsigned idx)
        {
            switch (idx) {
            case 0: return x;
            case 1: return y;
            };
            [[unlikely]]
            std::abort();
        }
    };

    template <typename T>
    struct vec3 {
        T x;
        T y;
        T z;

        T operator[](unsigned idx)
        {
            switch (idx) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            };
            [[unlikely]]
            std::abort();
        }
    };

    template <typename T>
    struct vec4 {
        T x;
        T y;
        T z;
        T w;

        T operator[](unsigned idx)
        {
            switch (idx) {
            case 0: return x;
            case 1: return y;
            case 2: return z;
            case 3: return w;
            };
            [[unlikely]]
            std::abort();
        }
    };

} // namespace details;

using uchar2 = details::vec2<std::uint8_t>;
using uchar3 = details::vec3<std::uint8_t>;
using uchar4 = details::vec4<std::uint8_t>;

using int2 = details::vec2<std::int32_t>;
using int3 = details::vec3<std::int32_t>;
using int4 = details::vec4<std::int32_t>;

using uint2 = details::vec2<std::uint32_t>;
using uint3 = details::vec3<std::uint32_t>;
using uint4 = details::vec4<std::uint32_t>;

struct dim3 {
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
        : x(x_)
        , y(y_)
        , z(z_)
    {}

    unsigned x = 1;
    unsigned y = 1;
    unsigned z = 1;
};

inline uchar3 make_uchar3(std::uint8_t x, std::uint8_t y, std::uint8_t z)
{
    return uchar3{ x, y, z };
}

inline uint3 make_uint3(std::uint32_t x, std::uint32_t y, std::uint32_t z)
{
    return uint3{ x, y, z };
}

enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
};

struct cudaExternalMemoryHandleDesc {
    vk::Device device;
    vk::DeviceMemory deviceMemory;
    unsigned long long size;
};

struct cudaExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
};

enum cudaExternalSemaphoreHandleType {
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
};

struct cudaExternalSemaphoreHandleDesc {
    cudaExternalSemaphoreHandleType type;
    vk::Device device;
    vk::Semaphore semaphore;
};

struct cudaExternalSemaphoreWaitParams {
    union {
        struct {
            unsigned long long value;
        } fence;
    } params;
};

struct cudaExternalSemaphoreSignalParams {
    union {
        struct {
            unsigned long long value;
        } fence;
    } params;
};

extern thread_local dim3 threadIdx;
extern thread_local dim3 blockIdx;

extern thread_local dim3 blockDim;
extern thread_local dim3 gridDim;

using cudaStream_t = void*;
using cudaEvent_t = void*;

struct cudaExternalSemaphoreStruct {
    vk::Device device;
    vk::Semaphore semaphore;
};

using cudaExternalSemaphore_t = cudaExternalSemaphoreStruct*;

struct cudaExternalMemoryStruct {
    vk::Device device;
    vk::DeviceMemory deviceMemory;
    cudaExternalMemoryBufferDesc desc;
    bool mapped = false;
    void* cpuMemory;
};
using cudaExternalMemory_t = cudaExternalMemoryStruct*;

enum cudaError {
    cudaSuccess = 0,
    cudaErrorMemoryAllocation = 2,
    cudaErrorNotYetImplemented = 31,
    cudaErrorNotReady = 600,
};
using cudaError_t = cudaError;

inline constexpr unsigned cudaEventDisableTiming = 0x02;

inline constexpr unsigned cudaStreamDefault = 0;
inline constexpr unsigned cudaStreamNonBlocking = 1;

inline cudaError_t cudaGetDevice(int*) { return cudaErrorNotYetImplemented; }

const char* cudaGetErrorString(cudaError_t);

cudaError_t cudaMalloc(void** devPtr, std::size_t size);
cudaError_t cudaFree(void*);

cudaError_t cudaMallocHost(void** ptr, std::size_t size);
cudaError_t cudaFreeHost(void*);

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind, cudaStream_t = nullptr);
cudaError_t cudaMemsetAsync(void* devPtr, int value, std::size_t count, cudaStream_t stream = nullptr);

cudaError_t cudaEventCreateWithFlags(cudaEvent_t*, unsigned flags);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);

cudaError_t cudaStreamCreateWithFlags(cudaStream_t*, unsigned flags);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned flags = 0);

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMemOut, const cudaExternalMemoryHandleDesc*);
cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t);
cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t, const cudaExternalMemoryBufferDesc*);

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSemOut, const cudaExternalSemaphoreHandleDesc*);
cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t);
cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray,
    const cudaExternalSemaphoreWaitParams* paramsArray, unsigned numExtSems, cudaStream_t stream = nullptr);
cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray,
    const cudaExternalSemaphoreSignalParams* paramsArray, unsigned numExtSems, cudaStream_t stream = nullptr);

template <std::integral T>
inline T min(T a, T b) { return a < b ? a : b; }

template <std::integral T, std::convertible_to<T> T2 = T>
T atomicAdd(T* address, T2 val)
{
    std::atomic_ref ref(*address);
    return ref.fetch_add(val, std::memory_order_relaxed);
}

template <std::integral T, std::convertible_to<T> T2 = T>
T atomicMin(T* address, T2 val)
{
    std::atomic_ref ref(*address);
    auto expected = *address;
    while (expected > val) {
        if (ref.compare_exchange_weak(expected, val, std::memory_order_relaxed))
            return expected;
    }
    return expected;
}

template <std::integral T, std::convertible_to<T> T2 = T>
T atomicMax(T* address, T2 val)
{
    std::atomic_ref ref(*address);
    auto expected = *address;
    while (expected < val) {
        if (ref.compare_exchange_weak(expected, val, std::memory_order_relaxed))
            return expected;
    }
    return expected;
}

template <std::integral T>
T atomicCAS(T* address, T expected, T newVal)
{
    std::atomic_ref ref(*address);
    return ref.compare_exchange_strong(expected, newVal, std::memory_order_relaxed);
}
