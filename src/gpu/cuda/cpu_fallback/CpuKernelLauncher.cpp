#include "CpuKernelLauncher.h"

#include <cstddef>
#include <iostream>
#include <thread>

namespace blocklab {

CpuKernelLauncher CpuKernelLauncher::s_instance;
thread_local std::vector<char> CpuKernelLauncher::m_dynamicSharedMemory;

CpuKernelLauncher::CpuKernelLauncher()
{
    const unsigned nThreads = std::max(std::thread::hardware_concurrency(), 2U);
    std::cout << "CpuKernelLauncher concurrency = " << nThreads << std::endl;
    m_threadPool.resize(nThreads - 1); // caller's thread will be used as well to run a kernel
    for (std::size_t i = 0; i < nThreads - 1; ++i) {
        m_threadPool[i].thread = std::thread(
            [this, ti = &m_threadPool[i]]() {
                threadFunc(*ti);
            }
        );
    }
}

CpuKernelLauncher::~CpuKernelLauncher()
{
    m_terminationRequested.store(true, std::memory_order_relaxed);
    m_version.fetch_add(1, std::memory_order_release);
    m_version.notify_all();

    for (ThreadInfo& ti: m_threadPool)
        ti.thread.join();
}

void CpuKernelLauncher::threadFunc(ThreadInfo& ti)
{
    std::uint64_t lastVersion = 0;
    for (;;) {
        m_version.wait(lastVersion, std::memory_order_acquire);
        if (m_terminationRequested.load(std::memory_order_relaxed)) [[unlikely]]
            break;
        lastVersion = m_version.load(std::memory_order_relaxed);

        assert(ti.task);
        ti.task();

        if (m_activeThreads.fetch_sub(1, std::memory_order_relaxed) == 1) {
            m_allTasksCompleted.store(true, std::memory_order_release);
            m_allTasksCompleted.notify_one();
        }
    }
}

} // namespace blocklab
