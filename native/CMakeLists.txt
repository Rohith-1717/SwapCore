cmake_minimum_required(VERSION 3.16)
project(swapcore_native)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

pybind11_add_module(swapcore_native
    FaultHandler.cpp
    MemRegion.cpp
    PageTable.cpp
    MemoryManager.cpp
    SwapManager.cpp
    EvictionManager.cpp
    runtime_gru.cpp
    Bindings.cpp
)

target_include_directories(swapcore_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
