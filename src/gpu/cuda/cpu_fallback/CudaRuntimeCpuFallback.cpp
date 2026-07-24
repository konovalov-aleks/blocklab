#include "CudaRuntimeCpuFallback.h"

#include <blocklab/gpu/vulkan/Vulkan.h>

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_set>

thread_local dim3 threadIdx;
thread_local dim3 blockIdx;

thread_local dim3 blockDim;
thread_local dim3 gridDim;

static std::unordered_set<void*> s_mappedMemory;
std::mutex s_mappedMemoryMtx;

const char* cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case cudaSuccess: return "no error";
    case cudaErrorMemoryAllocation: return "out of memory";
    case cudaErrorNotYetImplemented: return "not implemented yet";
    case cudaErrorNotReady: return "device not ready";
    };
    return "";
}

cudaError_t cudaMalloc(void** devPtr, std::size_t size)
{
    return cudaMallocHost(devPtr, size);
}

cudaError_t cudaFree(void* devPtr)
{
    {
        std::unique_lock lock(s_mappedMemoryMtx);
        auto iter = s_mappedMemory.find(devPtr);
        if (iter != s_mappedMemory.end()) {
            s_mappedMemory.erase(iter);
            return cudaSuccess;
        }
    }
    return cudaFreeHost(devPtr);
}

cudaError_t cudaMallocHost(void** ptr, std::size_t size)
{
    assert(ptr);
    *ptr = std::malloc(size);
    return *ptr ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFreeHost(void* ptr)
{
    if (ptr)
        std::free(ptr);
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind)
{
    std::memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind, cudaStream_t)
{
    return cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaMemsetAsync(void* devPtr, int value, std::size_t count, cudaStream_t)
{
    std::memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t* eventOut, unsigned /* flags */)
{
    assert(eventOut);
    static unsigned id = 0;
    *eventOut = reinterpret_cast<cudaEvent_t>(++id);
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t)
{
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t)
{
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t)
{
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t* streamOut, unsigned /* flags */)
{
    assert(streamOut);
    static unsigned id = 0;
    *streamOut = reinterpret_cast<cudaStream_t>(++id);
    return cudaSuccess;
}

cudaError_t cudaStreamDestroy(cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned /* flags */)
{
    return cudaSuccess;
}

cudaError_t cudaImportExternalMemory(cudaExternalMemory_t* extMemOut, const cudaExternalMemoryHandleDesc* desc)
{
    cudaExternalMemory_t result = new cudaExternalMemoryStruct();
    result->device = desc->device;
    result->deviceMemory = desc->deviceMemory;
    result->mapped = false;
    *extMemOut = result;
    return cudaSuccess;
}

cudaError_t cudaDestroyExternalMemory(cudaExternalMemory_t mem)
{
    if (mem) {
        if (mem->mapped)
            mem->device.unmapMemory(mem->deviceMemory);
        delete mem;
    }
    return cudaSuccess;
}

cudaError_t cudaExternalMemoryGetMappedBuffer(void** devPtr, cudaExternalMemory_t mem, const cudaExternalMemoryBufferDesc* d)
{
    assert(devPtr);
    assert(mem);
    assert(d);
    if (!mem->mapped
      || d->offset < mem->desc.offset
      || d->offset + d->size > mem->desc.offset + mem->desc.size) {

        if (mem->mapped)
            mem->device.unmapMemory(mem->deviceMemory);

        mem->desc = *d;
        mem->mapped = true;

        mem->cpuMemory = blocklab::vkCheck(
            mem->device.mapMemory(mem->deviceMemory, d->offset, d->size), "VkDevice::mapMemory");

        std::unique_lock lock(s_mappedMemoryMtx);
        s_mappedMemory.insert(mem->cpuMemory);
    }
    *devPtr = mem->cpuMemory;
    return cudaSuccess;
}

cudaError_t cudaImportExternalSemaphore(cudaExternalSemaphore_t* extSemOut, const cudaExternalSemaphoreHandleDesc* desc)
{
    if (desc->type != cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd)
        return cudaErrorNotYetImplemented;

    assert(extSemOut);
    cudaExternalSemaphore_t res = new cudaExternalSemaphoreStruct();
    res->device = desc->device;
    res->semaphore = desc->semaphore;
    *extSemOut = res;
    return cudaSuccess;
}

cudaError_t cudaDestroyExternalSemaphore(cudaExternalSemaphore_t sem)
{
    delete sem;
    return cudaSuccess;
}

cudaError_t cudaWaitExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray,
    const cudaExternalSemaphoreWaitParams* paramsArray, unsigned numExtSems, cudaStream_t)
{
    if (!numExtSems)
        return cudaSuccess;

    // TODO replace with a class like absl::InlinedVector
    vk::Semaphore stackSemBuf[8];
    std::unique_ptr<vk::Semaphore[]> heapSemBuf;

    std::uint64_t stackValueBuf[std::size(stackSemBuf)];
    std::unique_ptr<std::uint64_t[]> heapValueBuf;

    vk::Semaphore* semaphores;
    std::uint64_t* values;

    vk::Device device = extSemArray[0]->device;

    if (numExtSems <= std::size(stackSemBuf)) {
        semaphores = stackSemBuf;
        values = stackValueBuf;
    } else {
        heapSemBuf = std::make_unique<vk::Semaphore[]>(numExtSems);
        semaphores = heapSemBuf.get();
        heapValueBuf = std::make_unique<std::uint64_t[]>(numExtSems);
        values = heapValueBuf.get();
    }

    for (unsigned i = 0; i < numExtSems; ++i) {
        semaphores[i] = extSemArray[i]->semaphore;
        values[i] = static_cast<std::uint64_t>(paramsArray[i].params.fence.value);
        assert(extSemArray[i]->device == device);
    }

    vk::SemaphoreWaitInfo waitInfo{
        .semaphoreCount = numExtSems,
        .pSemaphores = semaphores,
        .pValues = values,
    };
    blocklab::vkCheck(device.waitSemaphores(waitInfo, UINT64_MAX), "vk::Device::waitSemaphores");
    return cudaSuccess;
}

cudaError_t cudaSignalExternalSemaphoresAsync(const cudaExternalSemaphore_t* extSemArray,
    const cudaExternalSemaphoreSignalParams* paramsArray, unsigned numExtSems, cudaStream_t)
{
    // TODO
    (void)extSemArray;
    (void)paramsArray;
    (void)numExtSems;
    return cudaErrorNotYetImplemented;
}
