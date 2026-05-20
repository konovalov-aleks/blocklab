cmake_minimum_required(VERSION 3.24)

project(blocklab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

option(USE_SYSTEM_SDL3 "Use an installed SDL3 package instead of fetching SDL3" OFF)
option(USE_SYSTEM_SDLSHADERCROSS "Use an installed SDL3_shadercross package instead of fetching SDL_shadercross" OFF)

if(USE_SYSTEM_SDL3)
    find_package(SDL3 CONFIG REQUIRED)
else()
    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        SDL3
        GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
        GIT_TAG release-3.4.8
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(SDL3)
endif()

if(USE_SYSTEM_SDLSHADERCROSS)
    find_package(SDL3_shadercross CONFIG REQUIRED)
else()
    set(SDLSHADERCROSS_VENDORED ON CACHE BOOL "" FORCE)
    set(SDLSHADERCROSS_CLI ON CACHE BOOL "" FORCE)
    set(SDLSHADERCROSS_INSTALL OFF CACHE BOOL "" FORCE)
    set(SDLSHADERCROSS_TESTS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        SDL_shadercross
        GIT_REPOSITORY https://github.com/libsdl-org/SDL_shadercross.git
        GIT_TAG main
        GIT_SUBMODULES_RECURSE TRUE
        GIT_PROGRESS TRUE
    )
    FetchContent_GetProperties(SDL_shadercross)
    if(NOT sdl_shadercross_POPULATED)
        if(POLICY CMP0169)
            cmake_policy(PUSH)
            cmake_policy(SET CMP0169 OLD)
        endif()
        FetchContent_Populate(SDL_shadercross)
        if(POLICY CMP0169)
            cmake_policy(POP)
        endif()
        add_subdirectory(${sdl_shadercross_SOURCE_DIR} ${sdl_shadercross_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
endif()

FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.2
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(glm)

FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.7.1
    GIT_PROGRESS TRUE
)
FetchContent_MakeAvailable(Catch2)

if(APPLE)
    set(BLOCKLAB_SHADER_DEST MSL)
    set(BLOCKLAB_SHADER_EXTENSION msl)
    set(BLOCKLAB_SHADER_FORMAT_NAME MSL)
elseif(WIN32)
    set(BLOCKLAB_SHADER_DEST DXIL)
    set(BLOCKLAB_SHADER_EXTENSION dxil)
    set(BLOCKLAB_SHADER_FORMAT_NAME DXIL)
else()
    set(BLOCKLAB_SHADER_DEST SPIRV)
    set(BLOCKLAB_SHADER_EXTENSION spv)
    set(BLOCKLAB_SHADER_FORMAT_NAME SPIRV)
endif()

set(BLOCKLAB_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)

function(blocklab_compile_shader target_name source_file stage entrypoint output_name)
    set(output_file ${BLOCKLAB_SHADER_OUTPUT_DIR}/${output_name}.${BLOCKLAB_SHADER_EXTENSION})
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BLOCKLAB_SHADER_OUTPUT_DIR}
        COMMAND $<TARGET_FILE:shadercross>
            ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
            --source HLSL
            --dest ${BLOCKLAB_SHADER_DEST}
            --stage ${stage}
            --entrypoint ${entrypoint}
            --include ${CMAKE_CURRENT_SOURCE_DIR}/shaders
            --output ${output_file}
        DEPENDS
            ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/common.hlsl
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/graphics_common.hlsl
            shadercross
        VERBATIM
    )
    add_custom_target(${target_name} DEPENDS ${output_file})
endfunction()

blocklab_compile_shader(blocklab_generate_region_shader shaders/generate_region.comp.hlsl compute generateMain generate_region)
blocklab_compile_shader(blocklab_apply_overrides_shader shaders/apply_overrides.comp.hlsl compute applyOverridesMain apply_overrides)
blocklab_compile_shader(blocklab_reset_mesh_draw_shader shaders/reset_mesh_draw.comp.hlsl compute resetMeshDrawMain reset_mesh_draw)
blocklab_compile_shader(blocklab_build_mesh_shader shaders/build_mesh.comp.hlsl compute buildMeshMain build_mesh)
blocklab_compile_shader(blocklab_mesh_vertex_shader shaders/mesh.vert.hlsl vertex meshVertexMain mesh_vertex)
blocklab_compile_shader(blocklab_mesh_fragment_shader shaders/mesh.frag.hlsl fragment meshFragmentMain mesh_fragment)

add_library(blocklab_env
    src/Agent.cpp
    src/characters/Character.cpp
    src/characters/NPC.cpp
    src/characters/PigCharacter.cpp
    src/Environment.cpp
    src/World.cpp
)

target_include_directories(blocklab_env
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(blocklab_env PUBLIC cxx_std_20)
target_link_libraries(blocklab_env PUBLIC glm::glm)

target_compile_options(blocklab_env PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_executable(blocklab
    src/main.cpp
    src/Renderer.cpp
)

target_link_libraries(blocklab
    PRIVATE
        blocklab_env
        SDL3::SDL3
)
add_dependencies(blocklab
    blocklab_generate_region_shader
    blocklab_apply_overrides_shader
    blocklab_reset_mesh_draw_shader
    blocklab_build_mesh_shader
    blocklab_mesh_vertex_shader
    blocklab_mesh_fragment_shader
)

target_compile_definitions(blocklab PRIVATE
    BLOCKLAB_SHADER_DIR="${BLOCKLAB_SHADER_OUTPUT_DIR}"
    BLOCKLAB_SHADER_EXTENSION="${BLOCKLAB_SHADER_EXTENSION}"
    BLOCKLAB_SHADER_FORMAT_${BLOCKLAB_SHADER_FORMAT_NAME}=1
)

target_compile_options(blocklab PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_executable(blocklab_benchmark
    src/benchmark.cpp
    src/Renderer.cpp
)

target_link_libraries(blocklab_benchmark
    PRIVATE
        blocklab_env
        SDL3::SDL3
)
add_dependencies(blocklab_benchmark
    blocklab_generate_region_shader
    blocklab_apply_overrides_shader
    blocklab_reset_mesh_draw_shader
    blocklab_build_mesh_shader
    blocklab_mesh_vertex_shader
    blocklab_mesh_fragment_shader
)

target_compile_definitions(blocklab_benchmark PRIVATE
    BLOCKLAB_SHADER_DIR="${BLOCKLAB_SHADER_OUTPUT_DIR}"
    BLOCKLAB_SHADER_EXTENSION="${BLOCKLAB_SHADER_EXTENSION}"
    BLOCKLAB_SHADER_FORMAT_${BLOCKLAB_SHADER_FORMAT_NAME}=1
)

target_compile_options(blocklab_benchmark PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

enable_testing()

add_executable(blocklab_smoke_test
    tests/blocklab_smoke_test.cpp
)

target_link_libraries(blocklab_smoke_test PRIVATE blocklab_env)
target_link_libraries(blocklab_smoke_test PRIVATE Catch2::Catch2WithMain)

target_compile_options(blocklab_smoke_test PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_test(NAME blocklab_smoke_test COMMAND blocklab_smoke_test)

add_executable(blocklab_quad_tree_test
    tests/blocklab_quad_tree_test.cpp
)

target_link_libraries(blocklab_quad_tree_test PRIVATE blocklab_env)
target_link_libraries(blocklab_quad_tree_test PRIVATE Catch2::Catch2WithMain)

target_compile_options(blocklab_quad_tree_test PRIVATE
    $<$<CXX_COMPILER_ID:Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

add_test(NAME blocklab_quad_tree_test COMMAND blocklab_quad_tree_test)
