#pragma once

#include "CudaRuntimeCpuFallback.h"

#include <gpu/cuda/KernelLaunchArgs.h>

#include <algorithm>
#include <atomic>
#include <cassert>
#include <coroutine>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <thread>
#include <type_traits>
#include <vector>
#include <utility>

namespace blocklab {

// Not thread-safe. Concurrent kernel launches are not supported.
class CpuKernelLauncher {
public:
    static CpuKernelLauncher& instance() { return s_instance; }

    template <typename FnT, typename... ArgsT>
    void launchKernel(FnT fn, KernelLaunchArgs launchArgs, ArgsT&&... args)
    {
        static_assert((std::is_trivially_copyable_v<std::decay_t<ArgsT>> && ...),
            "CPU kernel launcher copies kernel arguments");

        const std::uint32_t nBlocks = launchArgs.gridDim.x * launchArgs.gridDim.y * launchArgs.gridDim.z;
        const std::uint32_t nThreadsToRun = std::min(maxConcurrency(), nBlocks);

        m_allTasksCompleted.store(false, std::memory_order_relaxed);
        m_activeThreads.store(nThreadsToRun - 1, std::memory_order_relaxed);
        for (std::uint32_t i = 1; i < nThreadsToRun; ++i) {
            m_threadPool[i - 1].task = [=, this]() {
                launchOnThread(i, nThreadsToRun, fn, launchArgs, args...);
            };
        }
        m_version.fetch_add(1, std::memory_order_release);
        m_version.notify_all();

        // use caller's thread for computations as well
        launchOnThread(0, nThreadsToRun, fn, launchArgs, args...);

        if (nThreadsToRun > 1)
            m_allTasksCompleted.wait(false, std::memory_order_acquire);
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
    CpuKernelLauncher();
    ~CpuKernelLauncher();

    CpuKernelLauncher(const CpuKernelLauncher&) = delete;
    CpuKernelLauncher& operator=(const CpuKernelLauncher&) = delete;

    std::uint32_t maxConcurrency() const
    {
        return static_cast<std::uint32_t>(m_threadPool.size() + 1);
    }

    template <typename FnT, typename... ArgsT>
    void launchOnThread(std::uint32_t offset, std::uint32_t stride,
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

    struct ThreadInfo {
        std::thread thread;
        std::function<void()> task;
    };

    void threadFunc(ThreadInfo&);

    static CpuKernelLauncher s_instance;
    static thread_local std::vector<char> m_dynamicSharedMemory;

    std::vector<ThreadInfo> m_threadPool;
    std::atomic<std::uint64_t> m_version = 0;
    std::atomic_int m_activeThreads = 0;
    std::atomic_bool m_allTasksCompleted = false;
    std::atomic_bool m_terminationRequested = false;
};

} // namespace blocklab
