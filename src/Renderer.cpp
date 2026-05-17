#include "blocklab/Renderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace blocklab {

namespace {

    constexpr float Pi = 3.14159265358979323846f;
    constexpr float EyeHeight = 1.62f;
    constexpr int32_t RegionSizeX = 64;
    constexpr int32_t RegionSizeY = Chunk::SizeY;
    constexpr int32_t RegionSizeZ = 64;
    constexpr int32_t RegionVolume = RegionSizeX * RegionSizeY * RegionSizeZ;
    constexpr uint32_t GenerateThreadCount = 256;
    constexpr uint32_t MaxActiveOverrides = 8192;
    constexpr uint32_t BytesPerBlock = sizeof(uint32_t);
    constexpr uint32_t VerticesPerBlock = 36;
    constexpr uint32_t MaxMeshVertices = RegionVolume * VerticesPerBlock;
    constexpr uint32_t AtlasTileSize = 16;
    constexpr uint32_t AtlasColumns = 4;
    constexpr uint32_t AtlasRows = 2;
    constexpr uint32_t AtlasWidth = AtlasTileSize * AtlasColumns;
    constexpr uint32_t AtlasHeight = AtlasTileSize * AtlasRows;

    struct MeshVertex {
        Vec4 position;
        Vec4 uvAndShade;
    };

    struct Pixel {
        uint8_t r = 0;
        uint8_t g = 0;
        uint8_t b = 0;
        uint8_t a = 255;
    };

    enum class AtlasTile : uint32_t {
        GrassTop = 0,
        Dirt = 1,
        Stone = 2,
        GrassSide = 3,
        Sand = 4,
        Wood = 5,
        Leaves = 6,
        Water = 7,
    };

    struct ShaderBlob {
        std::vector<Uint8> code;
        SDL_GPUShaderFormat format = SDL_GPU_SHADERFORMAT_INVALID;
    };

    SDL_GPUShaderFormat shaderFormat()
    {
#if defined(BLOCKLAB_SHADER_FORMAT_MSL)
        return SDL_GPU_SHADERFORMAT_MSL;
#elif defined(BLOCKLAB_SHADER_FORMAT_DXIL)
        return SDL_GPU_SHADERFORMAT_DXIL;
#elif defined(BLOCKLAB_SHADER_FORMAT_SPIRV)
        return SDL_GPU_SHADERFORMAT_SPIRV;
#else
        return SDL_GPU_SHADERFORMAT_INVALID;
#endif
    }

    const char* shaderDriverName()
    {
#if defined(BLOCKLAB_SHADER_FORMAT_MSL)
        return "metal";
#else
        return nullptr;
#endif
    }

    ShaderBlob loadShaderBlob(const char* name)
    {
        const std::string path = std::string(BLOCKLAB_SHADER_DIR) + "/" + name + "." + BLOCKLAB_SHADER_EXTENSION;
        std::ifstream input(path, std::ios::binary);
        if (!input) [[unlikely]]
            throw std::runtime_error("Failed to open shader: " + path);

        std::vector<Uint8> code((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        if (code.empty()) [[unlikely]]
            throw std::runtime_error("Shader is empty: " + path);

        return {
            .code = std::move(code),
            .format = shaderFormat(),
        };
    }

    Vec3 cameraForward(float yaw, float pitch)
    {
        const float pitchCos = std::cos(pitch);
        return glm::normalize(Vec3 {
            std::sin(yaw) * pitchCos,
            std::sin(pitch),
            std::cos(yaw) * pitchCos,
        });
    }

    uint32_t textureNoise(uint32_t tile, uint32_t x, uint32_t y)
    {
        uint32_t value = tile * 0x9e3779b9U ^ x * 0x85ebca6bU ^ y * 0xc2b2ae35U;
        value ^= value >> 16U;
        value *= 0x7feb352dU;
        value ^= value >> 15U;
        return value;
    }

    Pixel shade(Pixel base, int32_t delta)
    {
        const auto clampByte = [](int32_t value) { return static_cast<uint8_t>(std::clamp(value, 0, 255)); };
        return {
            .r = clampByte(static_cast<int32_t>(base.r) + delta),
            .g = clampByte(static_cast<int32_t>(base.g) + delta),
            .b = clampByte(static_cast<int32_t>(base.b) + delta),
            .a = base.a,
        };
    }

    inline uint32_t tileIndex(AtlasTile tile) { return static_cast<uint32_t>(tile); }

    void setAtlasPixel(std::vector<Pixel>& pixels, AtlasTile tile, uint32_t x, uint32_t y, Pixel color)
    {
        const uint32_t index = tileIndex(tile);
        const uint32_t tileX = index % AtlasColumns;
        const uint32_t tileY = index / AtlasColumns;
        const uint32_t atlasX = tileX * AtlasTileSize + x;
        const uint32_t atlasY = tileY * AtlasTileSize + y;
        pixels[atlasY * AtlasWidth + atlasX] = color;
    }

    void fillNoisyTile(std::vector<Pixel>& pixels, AtlasTile tile, Pixel base, int32_t amplitude)
    {
        for (uint32_t y = 0; y < AtlasTileSize; ++y) {
            for (uint32_t x = 0; x < AtlasTileSize; ++x) {
                const int32_t noise
                    = static_cast<int32_t>(textureNoise(tileIndex(tile), x, y) % (amplitude * 2 + 1)) - amplitude;
                setAtlasPixel(pixels, tile, x, y, shade(base, noise));
            }
        }
    }

    std::vector<Pixel> generateBlockAtlas()
    {
        std::vector<Pixel> pixels(AtlasWidth * AtlasHeight);

        fillNoisyTile(pixels, AtlasTile::GrassTop, { 83, 166, 74, 255 }, 22);
        fillNoisyTile(pixels, AtlasTile::Dirt, { 126, 86, 53, 255 }, 20);
        fillNoisyTile(pixels, AtlasTile::Stone, { 126, 130, 134, 255 }, 18);
        fillNoisyTile(pixels, AtlasTile::GrassSide, { 121, 83, 49, 255 }, 18);
        fillNoisyTile(pixels, AtlasTile::Sand, { 198, 178, 106, 255 }, 13);
        fillNoisyTile(pixels, AtlasTile::Wood, { 116, 76, 42, 255 }, 18);
        fillNoisyTile(pixels, AtlasTile::Leaves, { 65, 131, 67, 255 }, 24);
        fillNoisyTile(pixels, AtlasTile::Water, { 72, 132, 204, 210 }, 16);

        for (uint32_t y = 0; y < AtlasTileSize; ++y) {
            for (uint32_t x = 0; x < AtlasTileSize; ++x) {
                if (y < 4 || ((textureNoise(tileIndex(AtlasTile::GrassSide), x, y) & 7U) == 0U && y < 7)) {
                    const int32_t noise
                        = static_cast<int32_t>(textureNoise(tileIndex(AtlasTile::GrassTop), x, y) % 29U) - 14;
                    setAtlasPixel(pixels, AtlasTile::GrassSide, x, y, shade({ 75, 154, 69, 255 }, noise));
                }
            }
        }

        for (uint32_t y = 0; y < AtlasTileSize; ++y) {
            for (uint32_t x = 0; x < AtlasTileSize; ++x) {
                if ((textureNoise(tileIndex(AtlasTile::Wood), x / 2, 0) & 3U) == 0U) {
                    setAtlasPixel(
                        pixels, AtlasTile::Wood, x, y, shade({ 75, 47, 27, 255 }, static_cast<int32_t>(y % 3U) * 5));
                }
            }
        }

        return pixels;
    }

} // namespace

Renderer::Renderer(SDL_Window* window, RenderConfig config)
    : m_config(config)
    , m_window(window)
    , m_width(config.width)
    , m_height(config.height)
{
    m_gpuOverrides.reserve(MaxActiveOverrides);
    m_observation = {
        .width = m_config.width,
        .height = m_config.height,
        .channels = 4,
        .device = ObservationDevice::SdlGpuTexture,
        .format = ObservationFormat::RGBA8,
    };
    initializeGpuResources();
}

Renderer::~Renderer() { releaseGpuResources(); }

void Renderer::resize(int width, int height)
{
    m_width = std::max(1, width);
    m_height = std::max(1, height);
}

Observation Renderer::renderObservation(const World& world, const AgentState& agent)
{
    renderGpuFrame(world, agent);
    return m_observation;
}

Renderer::RenderParams Renderer::buildRenderParams(const AgentState& agent) const
{
    const Vec3 origin = agent.position + Vec3 { 0.0f, EyeHeight, 0.0f };
    const Vec3 forward = cameraForward(agent.yaw, agent.pitch);
    const Vec3 right = glm::normalize(Vec3 { std::cos(agent.yaw), 0.0f, -std::sin(agent.yaw) });
    const Vec3 up = glm::normalize(glm::cross(forward, right));
    const int32_t originX = floorToInt(agent.position.x) - RegionSizeX / 2;
    const int32_t originZ = floorToInt(agent.position.z) - RegionSizeZ / 2;

    return {
        .origin = { origin.x, origin.y, origin.z, 0.0f },
        .forward = { forward.x, forward.y, forward.z, 0.0f },
        .right = { right.x, right.y, right.z, 0.0f },
        .up = { up.x, up.y, up.z, 0.0f },
        .worldOriginAndWidth = { originX, 0, originZ, m_config.width },
        .regionAndHeight = { RegionSizeX, RegionSizeY, RegionSizeZ, m_config.height },
        .overrideInfo = { 0, 0, 0, 0 },
        .tuning = { 48.0f, Pi / 2.25f, 10.0f, 28.0f },
    };
}

bool Renderer::updateBlockBuffer(const World& world, const RenderParams& params, SDL_GPUCommandBuffer* commandBuffer)
{
    const int32_t originX = params.worldOriginAndWidth.x;
    const int32_t originZ = params.worldOriginAndWidth.z;
    const IVec3 regionOrigin { originX, 0, originZ };
    if (m_blockBufferValid && m_cachedRegionOrigin == regionOrigin && m_cachedWorldVersion == world.version())
        return false;

    const SDL_GPUStorageBufferReadWriteBinding blockBufferBinding {
        .buffer = m_blockBuffer,
        .cycle = false,
    };
    SDL_PushGPUComputeUniformData(commandBuffer, 0, &params, sizeof(params));
    SDL_GPUComputePass* generatePass = SDL_BeginGPUComputePass(commandBuffer, nullptr, 0, &blockBufferBinding, 1);
    SDL_BindGPUComputePipeline(generatePass, m_generatePipeline);
    SDL_DispatchGPUCompute(generatePass, (RegionVolume + GenerateThreadCount - 1) / GenerateThreadCount, 1, 1);
    SDL_EndGPUComputePass(generatePass);

    world.collectOverridesInRegion(regionOrigin, { RegionSizeX, RegionSizeY, RegionSizeZ }, m_activeOverrides);
    if (m_activeOverrides.size() > MaxActiveOverrides)
        m_activeOverrides.resize(MaxActiveOverrides);

    m_gpuOverrides.clear();
    m_gpuOverrides.reserve(m_activeOverrides.size());
    for (const BlockOverride& edit : m_activeOverrides) {
        m_gpuOverrides.push_back({
            .x = edit.coord.x,
            .y = edit.coord.y,
            .z = edit.coord.z,
            .block = static_cast<int32_t>(edit.block),
        });
    }

    if (!m_gpuOverrides.empty()) {
        void* mapped = SDL_MapGPUTransferBuffer(m_device, m_overrideTransferBuffer, true);
        if (mapped) {
            const uint32_t uploadSize = static_cast<uint32_t>(m_gpuOverrides.size() * sizeof(GpuBlockOverride));
            std::memcpy(mapped, m_gpuOverrides.data(), uploadSize);
            SDL_UnmapGPUTransferBuffer(m_device, m_overrideTransferBuffer);

            SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
            const SDL_GPUTransferBufferLocation source {
                .transfer_buffer = m_overrideTransferBuffer,
                .offset = 0,
            };
            const SDL_GPUBufferRegion destination {
                .buffer = m_overrideBuffer,
                .offset = 0,
                .size = uploadSize,
            };
            SDL_UploadToGPUBuffer(copyPass, &source, &destination, false);
            SDL_EndGPUCopyPass(copyPass);

            RenderParams overrideParams = params;
            overrideParams.overrideInfo.x = static_cast<int32_t>(m_gpuOverrides.size());
            SDL_PushGPUComputeUniformData(commandBuffer, 0, &overrideParams, sizeof(overrideParams));
            SDL_GPUComputePass* overridePass
                = SDL_BeginGPUComputePass(commandBuffer, nullptr, 0, &blockBufferBinding, 1);
            SDL_BindGPUComputePipeline(overridePass, m_applyOverridesPipeline);
            SDL_GPUBuffer* overrideBuffer = m_overrideBuffer;
            SDL_BindGPUComputeStorageBuffers(overridePass, 0, &overrideBuffer, 1);
            SDL_DispatchGPUCompute(overridePass,
                (static_cast<uint32_t>(m_gpuOverrides.size()) + GenerateThreadCount - 1) / GenerateThreadCount, 1, 1);
            SDL_EndGPUComputePass(overridePass);
        }
    }

    m_cachedRegionOrigin = regionOrigin;
    m_cachedWorldVersion = world.version();
    m_blockBufferValid = true;
    return true;
}

void Renderer::rebuildMeshBuffer(const RenderParams& params, SDL_GPUCommandBuffer* commandBuffer)
{
    const SDL_GPUStorageBufferReadWriteBinding resetBindings[] {
        {
            .buffer = m_meshDrawBuffer,
            .cycle = false,
        },
    };
    SDL_GPUComputePass* resetPass = SDL_BeginGPUComputePass(commandBuffer, nullptr, 0, resetBindings, 1);
    SDL_BindGPUComputePipeline(resetPass, m_resetMeshDrawPipeline);
    SDL_DispatchGPUCompute(resetPass, 1, 1, 1);
    SDL_EndGPUComputePass(resetPass);

    const SDL_GPUStorageBufferReadWriteBinding buildBindings[] {
        {
            .buffer = m_meshVertexBuffer,
            .cycle = false,
        },
        {
            .buffer = m_meshDrawBuffer,
            .cycle = false,
        },
    };
    SDL_PushGPUComputeUniformData(commandBuffer, 0, &params, sizeof(params));
    SDL_GPUComputePass* buildPass = SDL_BeginGPUComputePass(commandBuffer, nullptr, 0, buildBindings, 2);
    SDL_BindGPUComputePipeline(buildPass, m_buildMeshPipeline);
    SDL_GPUBuffer* blockBuffer = m_blockBuffer;
    SDL_BindGPUComputeStorageBuffers(buildPass, 0, &blockBuffer, 1);
    SDL_DispatchGPUCompute(buildPass, (RegionVolume + GenerateThreadCount - 1) / GenerateThreadCount, 1, 1);
    SDL_EndGPUComputePass(buildPass);
}

void Renderer::uploadBlockAtlas()
{
    const std::vector<Pixel> pixels = generateBlockAtlas();
    const uint32_t uploadSize = static_cast<uint32_t>(pixels.size() * sizeof(Pixel));
    const SDL_GPUTransferBufferCreateInfo transferBufferInfo {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = uploadSize,
    };
    SDL_GPUTransferBuffer* transferBuffer = SDL_CreateGPUTransferBuffer(m_device, &transferBufferInfo);
    if (!transferBuffer) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    void* mapped = SDL_MapGPUTransferBuffer(m_device, transferBuffer, true);
    if (!mapped) [[unlikely]] {
        SDL_ReleaseGPUTransferBuffer(m_device, transferBuffer);
        throw std::runtime_error(SDL_GetError());
    }
    std::memcpy(mapped, pixels.data(), uploadSize);
    SDL_UnmapGPUTransferBuffer(m_device, transferBuffer);

    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(m_device);
    if (!commandBuffer) [[unlikely]] {
        SDL_ReleaseGPUTransferBuffer(m_device, transferBuffer);
        throw std::runtime_error(SDL_GetError());
    }

    SDL_GPUCopyPass* copyPass = SDL_BeginGPUCopyPass(commandBuffer);
    const SDL_GPUTextureTransferInfo source {
        .transfer_buffer = transferBuffer,
        .offset = 0,
        .pixels_per_row = AtlasWidth,
        .rows_per_layer = AtlasHeight,
    };
    const SDL_GPUTextureRegion destination {
        .texture = m_blockAtlas,
        .mip_level = 0,
        .layer = 0,
        .x = 0,
        .y = 0,
        .z = 0,
        .w = AtlasWidth,
        .h = AtlasHeight,
        .d = 1,
    };
    SDL_UploadToGPUTexture(copyPass, &source, &destination, false);
    SDL_EndGPUCopyPass(copyPass);
    SDL_SubmitGPUCommandBuffer(commandBuffer);
    SDL_WaitForGPUIdle(m_device);
    SDL_ReleaseGPUTransferBuffer(m_device, transferBuffer);
}

void Renderer::initializeGpuResources()
{
    const SDL_GPUShaderFormat format = shaderFormat();
    m_device = SDL_CreateGPUDevice(format, true, shaderDriverName());
    if (!m_device) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    if (!SDL_ClaimWindowForGPUDevice(m_device, m_window)) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    SDL_SetGPUAllowedFramesInFlight(m_device, 2);

    const SDL_GPUTextureCreateInfo textureInfo {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER | SDL_GPU_TEXTUREUSAGE_COLOR_TARGET,
        .width = static_cast<Uint32>(m_config.width),
        .height = static_cast<Uint32>(m_config.height),
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
    };
    m_observationTarget = SDL_CreateGPUTexture(m_device, &textureInfo);
    if (!m_observationTarget) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    m_observation.handle = reinterpret_cast<uintptr_t>(m_observationTarget);

    const SDL_GPUTextureCreateInfo depthTextureInfo {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_D16_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_DEPTH_STENCIL_TARGET,
        .width = static_cast<Uint32>(m_config.width),
        .height = static_cast<Uint32>(m_config.height),
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
    };
    m_depthTarget = SDL_CreateGPUTexture(m_device, &depthTextureInfo);
    if (!m_depthTarget) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const SDL_GPUTextureCreateInfo blockAtlasInfo {
        .type = SDL_GPU_TEXTURETYPE_2D,
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
        .usage = SDL_GPU_TEXTUREUSAGE_SAMPLER,
        .width = AtlasWidth,
        .height = AtlasHeight,
        .layer_count_or_depth = 1,
        .num_levels = 1,
        .sample_count = SDL_GPU_SAMPLECOUNT_1,
    };
    m_blockAtlas = SDL_CreateGPUTexture(m_device, &blockAtlasInfo);
    if (!m_blockAtlas) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const SDL_GPUSamplerCreateInfo blockSamplerInfo {
        .min_filter = SDL_GPU_FILTER_NEAREST,
        .mag_filter = SDL_GPU_FILTER_NEAREST,
        .mipmap_mode = SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
        .address_mode_u = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_v = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        .address_mode_w = SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
    };
    m_blockSampler = SDL_CreateGPUSampler(m_device, &blockSamplerInfo);
    if (!m_blockSampler) [[unlikely]]
        throw std::runtime_error(SDL_GetError());
    uploadBlockAtlas();

    const SDL_GPUBufferCreateInfo blockBufferInfo {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = RegionVolume * BytesPerBlock,
    };
    m_blockBuffer = SDL_CreateGPUBuffer(m_device, &blockBufferInfo);
    if (!m_blockBuffer) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const SDL_GPUBufferCreateInfo overrideBufferInfo {
        .usage = SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_READ,
        .size = MaxActiveOverrides * static_cast<uint32_t>(sizeof(GpuBlockOverride)),
    };
    m_overrideBuffer = SDL_CreateGPUBuffer(m_device, &overrideBufferInfo);
    if (!m_overrideBuffer) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const SDL_GPUTransferBufferCreateInfo overrideTransferBufferInfo {
        .usage = SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD,
        .size = MaxActiveOverrides * static_cast<uint32_t>(sizeof(GpuBlockOverride)),
    };
    m_overrideTransferBuffer = SDL_CreateGPUTransferBuffer(m_device, &overrideTransferBufferInfo);
    if (!m_overrideTransferBuffer) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const SDL_GPUBufferCreateInfo meshVertexBufferInfo {
        .usage = SDL_GPU_BUFFERUSAGE_GRAPHICS_STORAGE_READ | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = MaxMeshVertices * static_cast<uint32_t>(sizeof(MeshVertex)),
    };
    m_meshVertexBuffer = SDL_CreateGPUBuffer(m_device, &meshVertexBufferInfo);
    if (!m_meshVertexBuffer) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const SDL_GPUBufferCreateInfo meshDrawBufferInfo {
        .usage = SDL_GPU_BUFFERUSAGE_INDIRECT | SDL_GPU_BUFFERUSAGE_COMPUTE_STORAGE_WRITE,
        .size = static_cast<uint32_t>(sizeof(SDL_GPUIndirectDrawCommand)),
    };
    m_meshDrawBuffer = SDL_CreateGPUBuffer(m_device, &meshDrawBufferInfo);
    if (!m_meshDrawBuffer) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const ShaderBlob generateShader = loadShaderBlob("generate_region");
    const SDL_GPUComputePipelineCreateInfo generateInfo {
        .code_size = generateShader.code.size(),
        .code = generateShader.code.data(),
        .entrypoint = "generateMain",
        .format = generateShader.format,
        .num_samplers = 0,
        .num_readonly_storage_textures = 0,
        .num_readonly_storage_buffers = 0,
        .num_readwrite_storage_textures = 0,
        .num_readwrite_storage_buffers = 1,
        .num_uniform_buffers = 1,
        .threadcount_x = GenerateThreadCount,
        .threadcount_y = 1,
        .threadcount_z = 1,
    };
    m_generatePipeline = SDL_CreateGPUComputePipeline(m_device, &generateInfo);
    if (!m_generatePipeline) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const ShaderBlob applyOverridesShader = loadShaderBlob("apply_overrides");
    const SDL_GPUComputePipelineCreateInfo applyOverridesInfo {
        .code_size = applyOverridesShader.code.size(),
        .code = applyOverridesShader.code.data(),
        .entrypoint = "applyOverridesMain",
        .format = applyOverridesShader.format,
        .num_samplers = 0,
        .num_readonly_storage_textures = 0,
        .num_readonly_storage_buffers = 1,
        .num_readwrite_storage_textures = 0,
        .num_readwrite_storage_buffers = 1,
        .num_uniform_buffers = 1,
        .threadcount_x = GenerateThreadCount,
        .threadcount_y = 1,
        .threadcount_z = 1,
    };
    m_applyOverridesPipeline = SDL_CreateGPUComputePipeline(m_device, &applyOverridesInfo);
    if (!m_applyOverridesPipeline) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const ShaderBlob resetMeshDrawShader = loadShaderBlob("reset_mesh_draw");
    const SDL_GPUComputePipelineCreateInfo resetMeshDrawInfo {
        .code_size = resetMeshDrawShader.code.size(),
        .code = resetMeshDrawShader.code.data(),
        .entrypoint = "resetMeshDrawMain",
        .format = resetMeshDrawShader.format,
        .num_samplers = 0,
        .num_readonly_storage_textures = 0,
        .num_readonly_storage_buffers = 0,
        .num_readwrite_storage_textures = 0,
        .num_readwrite_storage_buffers = 1,
        .num_uniform_buffers = 0,
        .threadcount_x = 1,
        .threadcount_y = 1,
        .threadcount_z = 1,
    };
    m_resetMeshDrawPipeline = SDL_CreateGPUComputePipeline(m_device, &resetMeshDrawInfo);
    if (!m_resetMeshDrawPipeline) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const ShaderBlob buildMeshShader = loadShaderBlob("build_mesh");
    const SDL_GPUComputePipelineCreateInfo buildMeshInfo {
        .code_size = buildMeshShader.code.size(),
        .code = buildMeshShader.code.data(),
        .entrypoint = "buildMeshMain",
        .format = buildMeshShader.format,
        .num_samplers = 0,
        .num_readonly_storage_textures = 0,
        .num_readonly_storage_buffers = 1,
        .num_readwrite_storage_textures = 0,
        .num_readwrite_storage_buffers = 2,
        .num_uniform_buffers = 1,
        .threadcount_x = GenerateThreadCount,
        .threadcount_y = 1,
        .threadcount_z = 1,
    };
    m_buildMeshPipeline = SDL_CreateGPUComputePipeline(m_device, &buildMeshInfo);
    if (!m_buildMeshPipeline) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const ShaderBlob meshVertexShaderBlob = loadShaderBlob("mesh_vertex");
    const SDL_GPUShaderCreateInfo meshVertexShaderInfo {
        .code_size = meshVertexShaderBlob.code.size(),
        .code = meshVertexShaderBlob.code.data(),
        .entrypoint = "meshVertexMain",
        .format = meshVertexShaderBlob.format,
        .stage = SDL_GPU_SHADERSTAGE_VERTEX,
        .num_samplers = 0,
        .num_storage_textures = 0,
        .num_storage_buffers = 1,
        .num_uniform_buffers = 1,
    };
    SDL_GPUShader* meshVertexShader = SDL_CreateGPUShader(m_device, &meshVertexShaderInfo);
    if (!meshVertexShader) [[unlikely]]
        throw std::runtime_error(SDL_GetError());

    const ShaderBlob meshFragmentShaderBlob = loadShaderBlob("mesh_fragment");
    const SDL_GPUShaderCreateInfo meshFragmentShaderInfo {
        .code_size = meshFragmentShaderBlob.code.size(),
        .code = meshFragmentShaderBlob.code.data(),
        .entrypoint = "meshFragmentMain",
        .format = meshFragmentShaderBlob.format,
        .stage = SDL_GPU_SHADERSTAGE_FRAGMENT,
        .num_samplers = 1,
        .num_storage_textures = 0,
        .num_storage_buffers = 0,
        .num_uniform_buffers = 0,
    };
    SDL_GPUShader* meshFragmentShader = SDL_CreateGPUShader(m_device, &meshFragmentShaderInfo);
    if (!meshFragmentShader) [[unlikely]] {
        SDL_ReleaseGPUShader(m_device, meshVertexShader);
        throw std::runtime_error(SDL_GetError());
    }

    const SDL_GPUColorTargetDescription colorTargetDescription {
        .format = SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM,
    };
    const SDL_GPUGraphicsPipelineCreateInfo meshPipelineInfo {
        .vertex_shader = meshVertexShader,
        .fragment_shader = meshFragmentShader,
        .vertex_input_state = {},
        .primitive_type = SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        .rasterizer_state = {
            .fill_mode = SDL_GPU_FILLMODE_FILL,
            .cull_mode = SDL_GPU_CULLMODE_NONE,
            .front_face = SDL_GPU_FRONTFACE_COUNTER_CLOCKWISE,
            .enable_depth_clip = true,
        },
        .multisample_state = {
            .sample_count = SDL_GPU_SAMPLECOUNT_1,
        },
        .depth_stencil_state = {
            .compare_op = SDL_GPU_COMPAREOP_LESS,
            .enable_depth_test = true,
            .enable_depth_write = true,
        },
        .target_info = {
            .color_target_descriptions = &colorTargetDescription,
            .num_color_targets = 1,
            .depth_stencil_format = SDL_GPU_TEXTUREFORMAT_D16_UNORM,
            .has_depth_stencil_target = true,
        },
    };
    m_meshPipeline = SDL_CreateGPUGraphicsPipeline(m_device, &meshPipelineInfo);
    SDL_ReleaseGPUShader(m_device, meshFragmentShader);
    SDL_ReleaseGPUShader(m_device, meshVertexShader);
    if (!m_meshPipeline) [[unlikely]]
        throw std::runtime_error(SDL_GetError());
}

void Renderer::releaseGpuResources()
{
    if (!m_device)
        return;

    if (m_meshPipeline)
        SDL_ReleaseGPUGraphicsPipeline(m_device, m_meshPipeline);
    if (m_buildMeshPipeline)
        SDL_ReleaseGPUComputePipeline(m_device, m_buildMeshPipeline);
    if (m_resetMeshDrawPipeline)
        SDL_ReleaseGPUComputePipeline(m_device, m_resetMeshDrawPipeline);
    if (m_applyOverridesPipeline)
        SDL_ReleaseGPUComputePipeline(m_device, m_applyOverridesPipeline);
    if (m_generatePipeline)
        SDL_ReleaseGPUComputePipeline(m_device, m_generatePipeline);
    if (m_overrideTransferBuffer)
        SDL_ReleaseGPUTransferBuffer(m_device, m_overrideTransferBuffer);
    if (m_overrideBuffer)
        SDL_ReleaseGPUBuffer(m_device, m_overrideBuffer);
    if (m_meshDrawBuffer)
        SDL_ReleaseGPUBuffer(m_device, m_meshDrawBuffer);
    if (m_meshVertexBuffer)
        SDL_ReleaseGPUBuffer(m_device, m_meshVertexBuffer);
    if (m_blockSampler)
        SDL_ReleaseGPUSampler(m_device, m_blockSampler);
    if (m_blockAtlas)
        SDL_ReleaseGPUTexture(m_device, m_blockAtlas);
    if (m_blockBuffer)
        SDL_ReleaseGPUBuffer(m_device, m_blockBuffer);
    if (m_depthTarget)
        SDL_ReleaseGPUTexture(m_device, m_depthTarget);
    if (m_observationTarget)
        SDL_ReleaseGPUTexture(m_device, m_observationTarget);

    SDL_ReleaseWindowFromGPUDevice(m_device, m_window);
    SDL_DestroyGPUDevice(m_device);
}

void Renderer::renderGpuFrame(const World& world, const AgentState& agent)
{
    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(m_device);
    if (!commandBuffer) [[unlikely]]
        return;

    RenderParams params = buildRenderParams(agent);
    params.overrideInfo.y = static_cast<int32_t>(world.seed());
    const bool worldMeshChanged = updateBlockBuffer(world, params, commandBuffer);
    if (worldMeshChanged)
        rebuildMeshBuffer(params, commandBuffer);
    SDL_PushGPUVertexUniformData(commandBuffer, 0, &params, sizeof(params));

    const SDL_GPUColorTargetInfo colorTargetInfo {
        .texture = m_observationTarget,
        .mip_level = 0,
        .layer_or_depth_plane = 0,
        .clear_color = { 0.42f, 0.64f, 0.86f, 1.0f },
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_STORE,
        .cycle = false,
    };
    const SDL_GPUDepthStencilTargetInfo depthTargetInfo {
        .texture = m_depthTarget,
        .clear_depth = 1.0f,
        .load_op = SDL_GPU_LOADOP_CLEAR,
        .store_op = SDL_GPU_STOREOP_DONT_CARE,
        .stencil_load_op = SDL_GPU_LOADOP_DONT_CARE,
        .stencil_store_op = SDL_GPU_STOREOP_DONT_CARE,
        .cycle = false,
    };
    SDL_GPURenderPass* renderPass = SDL_BeginGPURenderPass(commandBuffer, &colorTargetInfo, 1, &depthTargetInfo);
    SDL_BindGPUGraphicsPipeline(renderPass, m_meshPipeline);
    SDL_GPUBuffer* meshVertexBuffer = m_meshVertexBuffer;
    SDL_BindGPUVertexStorageBuffers(renderPass, 0, &meshVertexBuffer, 1);
    const SDL_GPUTextureSamplerBinding blockAtlasBinding {
        .texture = m_blockAtlas,
        .sampler = m_blockSampler,
    };
    SDL_BindGPUFragmentSamplers(renderPass, 0, &blockAtlasBinding, 1);
    SDL_DrawGPUPrimitivesIndirect(renderPass, m_meshDrawBuffer, 0, 1);
    SDL_EndGPURenderPass(renderPass);
    SDL_SubmitGPUCommandBuffer(commandBuffer);

    ++m_observation.version;
    if (m_config.presentToWindow)
        present();
}

void Renderer::present()
{
    SDL_GPUCommandBuffer* commandBuffer = SDL_AcquireGPUCommandBuffer(m_device);
    if (!commandBuffer) [[unlikely]]
        return;

    SDL_GPUTexture* swapchainTexture = nullptr;
    uint32_t swapchainWidth = 0;
    uint32_t swapchainHeight = 0;
    if (SDL_WaitAndAcquireGPUSwapchainTexture(
            commandBuffer, m_window, &swapchainTexture, &swapchainWidth, &swapchainHeight)
        && swapchainTexture) {
        const SDL_GPUBlitInfo blitInfo {
            .source = {
                .texture = m_observationTarget,
                .w = static_cast<Uint32>(m_config.width),
                .h = static_cast<Uint32>(m_config.height),
            },
            .destination = {
                .texture = swapchainTexture,
                .w = swapchainWidth,
                .h = swapchainHeight,
            },
            .load_op = SDL_GPU_LOADOP_DONT_CARE,
            .filter = SDL_GPU_FILTER_NEAREST,
        };
        SDL_BlitGPUTexture(commandBuffer, &blitInfo);
    }
    SDL_SubmitGPUCommandBuffer(commandBuffer);
}

} // namespace blocklab
