cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/clang-14)
    set(CMAKE_CUDA_HOST_COMPILER /usr/bin/clang-14)
endif()

project(blocklab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)

find_program(GLSLC_EXECUTABLE glslc REQUIRED)

set(BLOCKLAB_SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
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
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source_file}
        VERBATIM
    )
    add_custom_target(${target_name} DEPENDS ${output_file})
endfunction()

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
set_target_properties(blocklab_env PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(blocklab_env
    PUBLIC
        glm::glm
        CUDA::cudart
)

blocklab_enable_strict_warnings(blocklab_env)

add_library(blocklab_render
    src/MeshBuilderCuda.cu
    src/MeshBuilder.cpp
    src/Renderer.cpp
)

target_include_directories(blocklab_render
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
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
add_dependencies(blocklab_render blocklab_mesh_vertex_shader blocklab_mesh_fragment_shader)
target_compile_definitions(blocklab_render PRIVATE BLOCKLAB_SHADER_DIR="${BLOCKLAB_SHADER_OUTPUT_DIR}")

blocklab_enable_strict_warnings(blocklab_render)
set_source_files_properties(src/MeshBuilderCuda.cu PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)

add_executable(blocklab
    src/main.cpp
)

target_link_libraries(blocklab PRIVATE blocklab_render)

blocklab_enable_strict_warnings(blocklab)

add_executable(blocklab_benchmark
    src/benchmark.cpp
)

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

enable_testing()

add_executable(blocklab_smoke_test
    tests/blocklab_smoke_test.cpp
)

target_link_libraries(blocklab_smoke_test PRIVATE blocklab_env Catch2::Catch2WithMain)

blocklab_enable_strict_warnings(blocklab_smoke_test)

add_test(NAME blocklab_smoke_test COMMAND blocklab_smoke_test)

add_executable(blocklab_quad_tree_test
    tests/blocklab_quad_tree_test.cpp
)

target_link_libraries(blocklab_quad_tree_test PRIVATE blocklab_env Catch2::Catch2WithMain)

blocklab_enable_strict_warnings(blocklab_quad_tree_test)

add_test(NAME blocklab_quad_tree_test COMMAND blocklab_quad_tree_test)

add_executable(blocklab_page_locked_vector_test
    tests/page_locked_vector_test.cpp
)

target_link_libraries(blocklab_page_locked_vector_test PRIVATE blocklab_env Catch2::Catch2WithMain)

blocklab_enable_strict_warnings(blocklab_page_locked_vector_test)

add_test(NAME blocklab_page_locked_vector_test COMMAND blocklab_page_locked_vector_test)

add_executable(blocklab_hash_test
    tests/hash_test.cpp
)

target_link_libraries(blocklab_hash_test PRIVATE blocklab_env Catch2::Catch2WithMain)

blocklab_enable_strict_warnings(blocklab_hash_test)

add_test(NAME blocklab_hash_test COMMAND blocklab_hash_test)
