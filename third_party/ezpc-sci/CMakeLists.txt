cmake_minimum_required(VERSION 3.18)
project(ezpc_sci_binding LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EZPC_SCI_PYTHON_BINDING "Build Python pybind11 module" ON)

set(SCI_BUILD_DIR "" CACHE PATH "Path to SCI build directory (contains lib/libSCI-*.a)")

if(NOT SCI_BUILD_DIR OR SCI_BUILD_DIR STREQUAL "")
    set(_DEFAULT_SCI "${CMAKE_CURRENT_SOURCE_DIR}/../ezpc/SCI/build")
    if(EXISTS "${_DEFAULT_SCI}/lib/libSCI-OT.a" OR EXISTS "${_DEFAULT_SCI}/lib/libSCI-FloatingPoint.a")
        set(SCI_BUILD_DIR "${_DEFAULT_SCI}")
        message(STATUS "Auto-detected SCI build: ${SCI_BUILD_DIR}")
    endif()
endif()

set(EZPC_HAS_SCI FALSE)
if(SCI_BUILD_DIR AND EXISTS "${SCI_BUILD_DIR}/lib")
    set(EZPC_HAS_SCI TRUE)
    message(STATUS "SCI libraries found at: ${SCI_BUILD_DIR}/lib")
else()
    message(STATUS "SCI not found — building emulated-only mode")
endif()

if(EZPC_HAS_SCI)
    get_filename_component(SCI_ROOT "${SCI_BUILD_DIR}/.." ABSOLUTE)
    set(SCI_SRC_DIR "${SCI_ROOT}/src")

    set(EIGEN_DIR "${SCI_ROOT}/extern/eigen")

    set(SEAL_INCLUDE_DIR "${SCI_BUILD_DIR}/include")

    file(READ "${SCI_SRC_DIR}/utils/net_io_channel.h" _NETIO_H)
    string(FIND "${_NETIO_H}" "recv_counter" _HAS_RECV)
    if(_HAS_RECV GREATER -1)
        set(EZPC_SCI_HAS_RECV_TAP TRUE)
        message(STATUS "SCI NetIO recv counter patch: present")
    else()
        set(EZPC_SCI_HAS_RECV_TAP FALSE)
        message(STATUS "SCI NetIO recv counter patch: absent (bytes_recv = 0, no leak tap)")
    endif()

    string(FIND "${_NETIO_H}" "ext_bytes" _HAS_EXT)
    if(_HAS_EXT GREATER -1)
        set(EZPC_SCI_HAS_EXT_COUNTER TRUE)
        message(STATUS "SCI OT-extension counter patch: present")
    else()
        set(EZPC_SCI_HAS_EXT_COUNTER FALSE)
    endif()

    file(READ "${SCI_SRC_DIR}/OT/split-iknp.h" _IKNP_H)
    string(FIND "${_IKNP_H}" "ef_pool_on" _HAS_POOL)
    if(_HAS_POOL GREATER -1)
        set(EZPC_SCI_HAS_OT_POOL TRUE)
        message(STATUS "SCI OT pool patch: present")
    else()
        set(EZPC_SCI_HAS_OT_POOL FALSE)
    endif()

    execute_process(COMMAND git -C "${SCI_ROOT}" rev-parse HEAD
                    OUTPUT_VARIABLE EZPC_SCI_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET RESULT_VARIABLE _GIT_RES)
    if(NOT _GIT_RES EQUAL 0)
        set(EZPC_SCI_COMMIT "unknown")
    endif()
    message(STATUS "EzPC commit: ${EZPC_SCI_COMMIT}")
endif()

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
        EZPC_SCI_COMMIT="${EZPC_SCI_COMMIT}"
        EMP_USE_RANDOM_DEVICE=1
    )
    if(EZPC_SCI_HAS_RECV_TAP)
        target_compile_definitions(ezpc_sci_core INTERFACE EZPC_SCI_HAS_RECV_TAP=1)
    endif()
    if(EZPC_SCI_HAS_OT_POOL)
        target_compile_definitions(ezpc_sci_core INTERFACE EZPC_SCI_HAS_OT_POOL=1)
    endif()
    if(EZPC_SCI_HAS_EXT_COUNTER)
        target_compile_definitions(ezpc_sci_core INTERFACE EZPC_SCI_HAS_EXT_COUNTER=1)
    endif()
    target_compile_options(ezpc_sci_core INTERFACE
        -maes -msse4.1 -mavx -mavx2
        -Wno-deprecated-declarations
        -Wno-macro-redefined
    )
    target_link_directories(ezpc_sci_core INTERFACE ${SCI_BUILD_DIR}/lib)
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

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(ezpc_sci_core INTERFACE OpenMP::OpenMP_CXX)
endif()

if(EZPC_SCI_PYTHON_BINDING)
    find_package(pybind11 QUIET)
    if(NOT pybind11_FOUND)
        include(FetchContent)
        FetchContent_Declare(pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.12.0
        )
        FetchContent_MakeAvailable(pybind11)
    endif()

    pybind11_add_module(ezpc_sci src/binding.cpp)
    target_link_libraries(ezpc_sci PRIVATE ezpc_sci_core)
    target_compile_definitions(ezpc_sci PRIVATE EZPC_BUILDING_PYTHON=1)

    install(TARGETS ezpc_sci LIBRARY DESTINATION .)
endif()

