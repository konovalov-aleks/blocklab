#pragma once

#include "CudaRuntimeCpuFallback.h"

#include <gpu/cuda/KernelLaunchArgs.h>

#include <algorithm>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

namespace blocklab {

class CpuKernelLauncher {
public:
    // TODO use thread pool

    // TODO read hardware properties
    static constexpr std::uint32_t s_nThreads = 12;

    template <typename FnT, typename... ArgsT>
    void launchKernel(FnT fn, KernelLaunchArgs launchArgs, ArgsT&&... args)
    {
        const std::uint32_t nBlocks = launchArgs.gridDim.x * launchArgs.gridDim.y * launchArgs.gridDim.z;
        const std::uint32_t nThreadsToRun = std::min(s_nThreads, nBlocks);
        m_threads.reserve(nThreadsToRun);
        for (std::uint32_t i = 0; i < nThreadsToRun; ++i) {
            m_threads.emplace_back([=, this]() {
                threadFunc(i, nThreadsToRun, fn, launchArgs, args...);
            });
        }
        for (std::thread& t : m_threads)
            t.join();
    }

    static void* dynamicSharedMemory()
    {
        return m_dynamicSharedMemory.data();
    }

    struct TaskPromise;

    struct Task : public std::coroutine_handle<TaskPromise> {
        using promise_type = TaskPromise;
        dim3 threadIdx;
    };

    struct TaskPromise {
        Task get_return_object() { return { Task::from_promise(*this)}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() { }
        void unhandled_exception() { std::abort(); }
    };

private:

    template <typename FnT, typename... ArgsT>
    void threadFunc(std::uint32_t offset, std::uint32_t stride,
                    FnT fn, KernelLaunchArgs launchArgs, ArgsT&&... args)
    {
        m_dynamicSharedMemory.resize(launchArgs.sharedMemBytes);

        blockDim = launchArgs.blockDim;
        const std::uint32_t nBlocks = launchArgs.gridDim.x * launchArgs.gridDim.y * launchArgs.gridDim.z;

        std::vector<Task> suspendedTasks;
        std::vector<Task> newSuspendedTasks;

        for (std::uint32_t linearBlockIdx = offset; linearBlockIdx < nBlocks; linearBlockIdx += stride) {
            blockIdx.x = linearBlockIdx % launchArgs.gridDim.x;
            blockIdx.y = (linearBlockIdx / launchArgs.gridDim.x) % launchArgs.gridDim.y;
            blockIdx.z = linearBlockIdx / (launchArgs.gridDim.x * launchArgs.gridDim.y);

            suspendedTasks.clear();
            for (threadIdx.z = 0; threadIdx.z < launchArgs.blockDim.z; ++threadIdx.z) {
                for (threadIdx.y = 0; threadIdx.y < launchArgs.blockDim.y; ++threadIdx.y) {
                    for (threadIdx.x = 0; threadIdx.x < launchArgs.blockDim.x; ++threadIdx.x) {
                        Task task = fn(std::forward<ArgsT>(args)...);
                        if (!task.done()) {
                            task.threadIdx = threadIdx;
                            suspendedTasks.push_back(std::move(task));
                        } else
                            task.destroy();
                    }
                }
            }

            while (!suspendedTasks.empty()) {
                newSuspendedTasks.clear();
                for (Task& task : suspendedTasks) {
                    assert(!task.done());
                    threadIdx = task.threadIdx;
                    task.resume();
                    if (!task.done())
                        newSuspendedTasks.push_back(std::move(task));
                    else
                        task.destroy();
                }
                std::swap(newSuspendedTasks, suspendedTasks);
            }
        }
    }

    static thread_local std::vector<char> m_dynamicSharedMemory;
    std::vector<std::thread> m_threads;
};

} // namespace blocklab
