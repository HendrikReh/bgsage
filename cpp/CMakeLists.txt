cmake_minimum_required(VERSION 3.18)
project(bgbot LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BGBOT_BUILD_PYTHON "Build the pybind11 Python extension" ON)
# When this file is the top-level CMakeLists (cmake -S cpp), build the
# benchmark & profiler executables. The wheel build in ../CMakeLists.txt sets
# this to OFF to keep the pybind11 module as the only compiled target.
option(BGBOT_BUILD_APPS "Build benchmark/profiler executables" ON)

# Paths are written relative to this listfile's directory so the file can be
# either used as the top-level CMakeLists (cmake -S cpp) or include()'d from
# the pip build's root CMakeLists.txt and from mobile/CMakeLists.txt.
set(BGBOT_CPU_ROOT "${CMAKE_CURRENT_LIST_DIR}")

# Optimization flags. Apple Silicon (arm64) is the supported target; the
# x86-64 branch exists only for the Modal rollout image
# (scripts/modal_rollout.py), which builds this engine on Linux x86-64
# containers and uses the AVX2 kernels in neural_net.cpp.
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    set(BGBOT_TARGET_ARCH "${CMAKE_OSX_ARCHITECTURES}")
else()
    set(BGBOT_TARGET_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(BGBOT_TARGET_ARCH MATCHES "^(arm64|aarch64)$")
    set(BGBOT_OPT_FLAGS -O3 -funsafe-math-optimizations -fno-math-errno -funroll-loops)
else()
    # Modal only (Linux x86-64).
    set(BGBOT_OPT_FLAGS -O3 -ffast-math -funroll-loops -mavx2 -mfma)
endif()

# Core library (CPU only)
add_library(bgbot_lib STATIC
    ${BGBOT_CPU_ROOT}/src/board.cpp
    ${BGBOT_CPU_ROOT}/src/moves.cpp
    ${BGBOT_CPU_ROOT}/src/strategy.cpp
    ${BGBOT_CPU_ROOT}/src/pubeval.cpp
    ${BGBOT_CPU_ROOT}/src/game.cpp
    ${BGBOT_CPU_ROOT}/src/benchmark.cpp
    ${BGBOT_CPU_ROOT}/src/encoding.cpp
    ${BGBOT_CPU_ROOT}/src/neural_net.cpp
    ${BGBOT_CPU_ROOT}/src/training.cpp
    ${BGBOT_CPU_ROOT}/src/multipy.cpp
    ${BGBOT_CPU_ROOT}/src/rollout.cpp
    ${BGBOT_CPU_ROOT}/src/cube.cpp
    ${BGBOT_CPU_ROOT}/src/cube_eval.cpp
    ${BGBOT_CPU_ROOT}/src/match_equity.cpp
    ${BGBOT_CPU_ROOT}/src/bearoff.cpp
)
target_include_directories(bgbot_lib PUBLIC ${BGBOT_CPU_ROOT}/include)
target_compile_options(bgbot_lib PRIVATE ${BGBOT_OPT_FLAGS})

# Threading support
find_package(Threads REQUIRED)
target_link_libraries(bgbot_lib PUBLIC Threads::Threads)

# Apple Silicon: batched NN evaluation through Accelerate's BLAS, which runs on
# the CPU's matrix unit (SME on M4, AMX on M1-M3) — see
# NeuralNetwork::forward_group. -DBGBOT_USE_ACCELERATE=OFF restores the
# NEON delta kernels.
if(APPLE AND BGBOT_TARGET_ARCH MATCHES "^(arm64|aarch64)$")
    option(BGBOT_USE_ACCELERATE "Batched NN evaluation via Apple Accelerate" ON)
    if(BGBOT_USE_ACCELERATE)
        find_library(BGBOT_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
        target_link_libraries(bgbot_lib PUBLIC ${BGBOT_ACCELERATE_FRAMEWORK})
        target_compile_definitions(bgbot_lib PRIVATE BGBOT_USE_ACCELERATE=1)
        # The new CBLAS interface needs macOS 13.3+ at runtime; with an older
        # deployment target keep the classic one (it emits deprecation warnings).
        if(NOT CMAKE_OSX_DEPLOYMENT_TARGET OR
           CMAKE_OSX_DEPLOYMENT_TARGET VERSION_GREATER_EQUAL "13.3")
            target_compile_definitions(bgbot_lib PRIVATE ACCELERATE_NEW_LAPACK)
        else()
            target_compile_options(bgbot_lib PRIVATE -Wno-deprecated-declarations)
        endif()
        message(STATUS "bgbot: Accelerate batched NN evaluation ON")
    endif()
endif()

if(BGBOT_BUILD_APPS)
    # 4-ply cube profiler
    add_executable(bgbot_profile_4ply ${BGBOT_CPU_ROOT}/src/profile_cube_4ply.cpp)
    target_link_libraries(bgbot_profile_4ply PRIVATE bgbot_lib)

    # Truncated rollout benchmark/profilers
    add_executable(benchmark_truncated_rollout ${BGBOT_CPU_ROOT}/src/benchmark_truncated_rollout.cpp)
    target_link_libraries(benchmark_truncated_rollout PRIVATE bgbot_lib)

    add_executable(benchmark_3t ${BGBOT_CPU_ROOT}/src/benchmark_3t.cpp)
    target_link_libraries(benchmark_3t PRIVATE bgbot_lib)

    add_executable(benchmark_4ply ${BGBOT_CPU_ROOT}/src/benchmark_4ply.cpp)
    target_link_libraries(benchmark_4ply PRIVATE bgbot_lib)

    add_executable(benchmark_4ply_opt ${BGBOT_CPU_ROOT}/src/benchmark_4ply_opt.cpp)
    target_link_libraries(benchmark_4ply_opt PRIVATE bgbot_lib)

    add_executable(test_4ply_accuracy ${BGBOT_CPU_ROOT}/src/test_4ply_accuracy.cpp)
    target_link_libraries(test_4ply_accuracy PRIVATE bgbot_lib)

    add_executable(profile_3t ${BGBOT_CPU_ROOT}/src/profile_3t.cpp)
    target_link_libraries(profile_3t PRIVATE bgbot_lib)

    add_executable(profile_3t_detailed ${BGBOT_CPU_ROOT}/src/profile_3t_detailed.cpp)
    target_link_libraries(profile_3t_detailed PRIVATE bgbot_lib)

    add_executable(profile_cube_benchmark ${BGBOT_CPU_ROOT}/src/profile_cube_benchmark.cpp)
    target_link_libraries(profile_cube_benchmark PRIVATE bgbot_lib)
endif()

if(BGBOT_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(bgbot_cpp ${BGBOT_CPU_ROOT}/pybind/bindings.cpp)
    target_link_libraries(bgbot_cpp PRIVATE bgbot_lib)
endif()
