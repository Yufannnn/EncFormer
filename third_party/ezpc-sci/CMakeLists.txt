cmake_minimum_required(VERSION 3.18)
project(ezpc_sci_binding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(EZPC_SCI_PYTHON_BINDING "Build Python pybind11 module" ON)
option(EZPC_STANDALONE_TEST    "Build standalone test binary"  OFF)

# ---------------------------------------------------------------------------
# Find EzPC/SCI build tree.
# Set SCI_BUILD_DIR to the SCI/build directory (contains lib/ and headers).
# ---------------------------------------------------------------------------
set(SCI_BUILD_DIR "" CACHE PATH "Path to SCI build directory (contains lib/libSCI-*.a)")

# Auto-detect sibling SCI build
if(NOT SCI_BUILD_DIR OR SCI_BUILD_DIR STREQUAL "")
    set(_DEFAULT_SCI "${CMAKE_CURRENT_SOURCE_DIR}/../ezpc/SCI/build")
    if(EXISTS "${_DEFAULT_SCI}/lib/libSCI-OT.a" OR EXISTS "${_DEFAULT_SCI}/lib/libSCI-FloatingPoint.a")
        set(SCI_BUILD_DIR "${_DEFAULT_SCI}")
        message(STATUS "Auto-detected SCI build: ${SCI_BUILD_DIR}")
    endif()
endif()

# Check if SCI is available
set(EZPC_HAS_SCI FALSE)
if(SCI_BUILD_DIR AND EXISTS "${SCI_BUILD_DIR}/lib")
    set(EZPC_HAS_SCI TRUE)
    message(STATUS "SCI libraries found at: ${SCI_BUILD_DIR}/lib")
else()
    message(STATUS "SCI not found — building emulated-only mode")
endif()

# ---------------------------------------------------------------------------
# SCI source and dependency paths
# ---------------------------------------------------------------------------
if(EZPC_HAS_SCI)
    # SCI source root (for headers)
    get_filename_component(SCI_ROOT "${SCI_BUILD_DIR}/.." ABSOLUTE)
    set(SCI_SRC_DIR "${SCI_ROOT}/src")

    # Eigen (built by SCI)
    set(EIGEN_DIR "${SCI_ROOT}/extern/eigen")

    # SEAL headers (installed by SCI build)
    set(SEAL_INCLUDE_DIR "${SCI_BUILD_DIR}/include")
endif()

# ---------------------------------------------------------------------------
# Core C++ interface library
# ---------------------------------------------------------------------------
add_library(ezpc_sci_core INTERFACE)

target_include_directories(ezpc_sci_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(EZPC_HAS_SCI)
    target_include_directories(ezpc_sci_core INTERFACE
        ${SCI_SRC_DIR}
        ${EIGEN_DIR}
        ${SEAL_INCLUDE_DIR}
    )
    target_compile_definitions(ezpc_sci_core INTERFACE
        EZPC_HAS_SCI=1
        SCI_OT=1
    )
    # SCI requires AES-NI, SIMD intrinsics, and rdseed (same flags as SCI's own build)
    target_compile_options(ezpc_sci_core INTERFACE
        -maes -msse4.1 -mavx -mavx2 -mrdseed
        -Wno-deprecated-declarations
        -Wno-macro-redefined
    )
    target_link_directories(ezpc_sci_core INTERFACE ${SCI_BUILD_DIR}/lib)
    # Link order matters for static archives
    target_link_libraries(ezpc_sci_core INTERFACE
        SCI-FloatingPoint
        SCI-BuildingBlocks
        SCI-Math
        SCI-OT
        SCI-LinearOT
        SCI-GC
        seal
        ssl
        crypto
        gmp
        gmpxx
        pthread
    )
endif()

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ezpc_sci_core INTERFACE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# Python binding via pybind11
# ---------------------------------------------------------------------------
if(EZPC_SCI_PYTHON_BINDING)
    find_package(pybind11 REQUIRED)

    pybind11_add_module(ezpc_sci src/binding.cpp)
    target_link_libraries(ezpc_sci PRIVATE ezpc_sci_core)
    target_compile_definitions(ezpc_sci PRIVATE EZPC_BUILDING_PYTHON=1)

    install(TARGETS ezpc_sci LIBRARY DESTINATION .)
endif()

# ---------------------------------------------------------------------------
# Standalone test binary
# ---------------------------------------------------------------------------
if(EZPC_STANDALONE_TEST)
    add_executable(ezpc_sci_test src/test_main.cpp)
    target_link_libraries(ezpc_sci_test PRIVATE ezpc_sci_core)
endif()
