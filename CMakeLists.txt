cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/clang-14)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang-14)
endif()

project(blocklab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

include(FetchContent)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)

find_program(GLSLC_EXECUTABLE glslc REQUIRED)
find_program(CLANG_FORMAT_EXECUTABLE clang-format)

file(GLOB_RECURSE BLOCKLAB_FORMAT_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.h
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.h
)

if(CLANG_FORMAT_EXECUTABLE)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE} --sort-includes -i ${BLOCKLAB_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Format C++ and CUDA sources"
        VERBATIM
    )
else()
    message(STATUS "clang-format not found; format target will not be available")
endif()

set(BLOCKLAB_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(GLOB BLOCKLAB_SHADER_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.hlsl)
add_custom_target(blocklab_shaders)
function(blocklab_compile_shader target_name source_file stage entrypoint output_name)
    set(output_file ${BLOCKLAB_SHADER_OUTPUT_DIR}/${output_name}.spv)
    add_custom_command(
        OUTPUT ${output_file}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${BLOCKLAB_SHADER_OUTPUT_DIR}
        COMMAND ${GLSLC_EXECUTABLE}
            -x hlsl
            -fshader-stage=${stage}
            -fentry-point=${entrypoint}
            ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
            -o ${output_file}
        DEPENDS ${BLOCKLAB_SHADER_SOURCES}
        VERBATIM
    )
    add_custom_target(${target_name} DEPENDS ${output_file})
    add_dependencies(blocklab_shaders ${target_name})
endfunction()

blocklab_compile_shader(blocklab_voxel_vertex_shader shaders/voxel.vert.hlsl vertex voxelVertexMain voxel_vertex)
blocklab_compile_shader(blocklab_mesh_vertex_shader shaders/mesh.vert.hlsl vertex meshVertexMain mesh_vertex)
blocklab_compile_shader(blocklab_mesh_fragment_shader shaders/mesh.frag.hlsl fragment meshFragmentMain mesh_fragment)

function(blocklab_enable_strict_warnings target_name)
    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang,GNU>:-Wall -Wextra -Werror -Wpedantic -Wno-missing-field-initializers>
        $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--diag-suppress=20013>
        $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CXX_COMPILER_ID:Clang>>:-Xcompiler=-Wno-overlength-strings>
    )
endfunction()

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

if(NOT pybind11_FOUND)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.13.6
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

add_library(blocklab_env
    src/characters/AgentCharacter.cpp
    src/characters/Character.cpp
    src/characters/NPC.cpp
    src/characters/PigCharacter.cpp
    src/environment/Agent.cpp
    src/environment/Environment.cpp
    src/world/World.cpp
)

target_include_directories(blocklab_env
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(blocklab_env PUBLIC cxx_std_20)
set_target_properties(blocklab_env PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blocklab_env
    PUBLIC
        glm::glm
        CUDA::cudart
)

blocklab_enable_strict_warnings(blocklab_env)

add_library(blocklab_render
    src/characters/meshes/PigMesh.cpp
    src/environment/CudaObservation.cu
    src/gpu/interop/VulkanCudaInteropBuffer.cpp
    src/gpu/interop/VulkanCudaInteropSemaphore.cpp
    src/gpu/vulkan/GLFWInit.cpp
    src/gpu/vulkan/VulkanBuffer.cpp
    src/gpu/vulkan/Vulkan.cpp
    src/gpu/vulkan/VulkanSemaphore.cpp
    src/graphics/Display.cpp
    src/graphics/Display.cu
    src/graphics/Renderer.cpp
    src/world/WorldGenerator.cu
    src/world/WorldGenerator.cpp
)

target_include_directories(blocklab_render
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_features(blocklab_render PUBLIC cxx_std_20)
set_target_properties(blocklab_render PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blocklab_render
    PUBLIC
        blocklab_env
        glm::glm
    PRIVATE
        Vulkan::Vulkan
        CUDA::cudart
        glfw
)
add_dependencies(blocklab_render blocklab_shaders)
target_compile_definitions(blocklab_render PRIVATE BLOCKLAB_SHADER_DIR="${BLOCKLAB_SHADER_OUTPUT_DIR}")
target_compile_definitions(blocklab_render PUBLIC VULKAN_HPP_NO_EXCEPTIONS VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)

blocklab_enable_strict_warnings(blocklab_render)

add_executable(blocklab
    src/app/main.cpp
)

target_include_directories(blocklab PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(blocklab PRIVATE blocklab_render)

blocklab_enable_strict_warnings(blocklab)

add_executable(blocklab_benchmark
    src/app/benchmark.cpp
)

target_include_directories(blocklab_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(blocklab_benchmark PRIVATE blocklab_render)

blocklab_enable_strict_warnings(blocklab_benchmark)

pybind11_add_module(_native
    src/python/Bindings.cpp
)

target_link_libraries(_native PRIVATE blocklab_render)
blocklab_enable_strict_warnings(_native)
set_target_properties(_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/blocklab
)
target_include_directories(_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

enable_testing()

add_executable(blocklab_tests
    tests/CudaFuture.cpp
    tests/Hash.cpp
    tests/PageLockedVector.cpp
    tests/QuadTree.cpp
    tests/SmokeTest.cpp
    tests/WorldLighting.cpp
    tests/WorldTestUtils.cpp
)

target_include_directories(blocklab_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(blocklab_tests PRIVATE blocklab_render Catch2::Catch2WithMain)

blocklab_enable_strict_warnings(blocklab_tests)

add_test(NAME blocklab_tests COMMAND blocklab_tests)

set(BLOCKLAB_PYTHON_TEST_EXECUTABLE ${CMAKE_CURRENT_SOURCE_DIR}/.venv/bin/python)
set(BLOCKLAB_VULKAN_VALIDATION_RUNNER ${CMAKE_CURRENT_SOURCE_DIR}/scripts/run_vulkan_validation.py)

add_test(NAME blocklab_cpp_vulkan_validation
    COMMAND ${Python_EXECUTABLE} ${BLOCKLAB_VULKAN_VALIDATION_RUNNER} --
        $<TARGET_FILE:blocklab_benchmark>
        --seconds 2
        --batch-size 8
        --warmup-steps 16
)

add_test(NAME blocklab_python_vulkan_validation
    COMMAND ${Python_EXECUTABLE} ${BLOCKLAB_VULKAN_VALIDATION_RUNNER} --
        ${BLOCKLAB_PYTHON_TEST_EXECUTABLE}
        -m blocklab.examples.benchmark
        --seconds 2
        --num-envs 8
        --warmup-steps 16
        --materialize-observation
)
set_tests_properties(blocklab_python_vulkan_validation PROPERTIES
    ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/python"
)
