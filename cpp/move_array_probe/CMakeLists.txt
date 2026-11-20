# Private move-generator-only validation build; never replaces bgbot_cpp.
cmake_minimum_required(VERSION 3.18)
project(neongammon_move_array_probe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_neongammon_move_array_probe NO_EXTRAS
    bindings.cpp ../src/moves.cpp ../src/board.cpp)
target_include_directories(_neongammon_move_array_probe PRIVATE ../include)
target_compile_options(_neongammon_move_array_probe PRIVATE -O3)
