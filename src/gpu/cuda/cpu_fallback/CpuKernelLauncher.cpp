#include "CpuKernelLauncher.h"

namespace blocklab {

thread_local std::vector<char> CpuKernelLauncher::m_dynamicSharedMemory;

} // namespace blocklab
