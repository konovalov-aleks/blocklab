#pragma once

#include "blocklab/CudaFuture.h"
#include "blocklab/WorldGenerator.h"

#include <memory>

namespace blocklab {

class CudaWorldGenerator final {
public:
    CudaWorldGenerator();
    ~CudaWorldGenerator();

    CudaWorldGenerator(const CudaWorldGenerator&) = delete;
    CudaWorldGenerator& operator=(const CudaWorldGenerator&) = delete;

    CudaFuture<WorldGenerationOutput> generate(const WorldGenerationInput&, WorldGenerationBuffers&&);

private:
    struct State;
    std::unique_ptr<State> m_state;
};

} // namespace blocklab
