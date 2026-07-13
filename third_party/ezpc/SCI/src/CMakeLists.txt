# set(CMAKE_FIND_DEBUG_MODE 1)
include_directories(${CMAKE_CURRENT_LIST_DIR})

add_library(Eigen3::Eigen INTERFACE IMPORTED)
set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/extern/eigen")

add_subdirectory(utils)
add_subdirectory(OT)
add_subdirectory(GC)
add_subdirectory(Millionaire)
add_subdirectory(BuildingBlocks)
add_subdirectory(LinearOT)
add_subdirectory(LinearHE)
add_subdirectory(NonLinear)
add_subdirectory(Math)
add_subdirectory(FloatingPoint)

add_library(SCI-common INTERFACE)

find_package(Threads REQUIRED)

target_link_libraries(SCI-common
    INTERFACE
    SCI-NonLinear
    Threads::Threads
)

target_include_directories(SCI-common
    INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:./>
)

add_library(SCI-OT library_fixed_uniform.cpp library_fixed.cpp
    globals.cpp cleartext_library_fixed.cpp)
add_library(SCI-HE library_fixed_uniform.cpp globals.cpp)
add_library(SCI-SecfloatML library_float_common.cpp library_float_secfloat.cpp globals_float.cpp cleartext_library_float.cpp)
add_library(SCI-Beacon library_float_common.cpp library_float_beacon.cpp globals_float.cpp cleartext_library_float.cpp)

target_link_libraries(SCI-OT PUBLIC SCI-common SCI-LinearOT SCI-GC
    SCI-Math SCI-BuildingBlocks SCI-FloatingPoint Eigen3::Eigen)
target_link_libraries(SCI-HE PUBLIC SCI-common SCI-LinearHE)
target_link_libraries(SCI-SecfloatML PUBLIC SCI-FloatingPoint) 
target_link_libraries(SCI-Beacon PUBLIC SCI-FloatingPoint) 

target_compile_definitions(SCI-OT PUBLIC SCI_OT=1)
target_compile_definitions(SCI-HE PUBLIC SCI_HE=1)

install(TARGETS
        SCI-OT
        SCI-HE
        SCI-common
        SCI-FloatingPoint
	    SCI-SecfloatML
	    SCI-Beacon
        SCI-BuildingBlocks
        SCI-LinearOT
        SCI-LinearHE
        SCI-NonLinear
        SCI-Math
        SCI-Millionaire
        SCI-GC
        SCI-OTPrimitive
        SCI-utils
        EXPORT SCITargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(EXPORT SCITargets
        FILE SCITargets.cmake
        NAMESPACE SCI::
        DESTINATION lib/cmake/SCI)

install(DIRECTORY
        utils
        OT
        GC
        Millionaire
        NonLinear
        BuildingBlocks
        LinearOT
        LinearHE
        Math
        FloatingPoint
        DESTINATION include
        FILES_MATCHING
        PATTERN "*.h"
        PATTERN "*.hpp")

install(FILES
        "defines.h"
        "defines_uniform.h"
        "defines_float.h"
        "globals.h"
        "globals_float.h"
        "library_fixed.h"
        "library_fixed_uniform.h"
        "library_float.h"
        "cleartext_library_fixed.h"
        "cleartext_library_fixed_uniform.h"
        "cleartext_library_float.h"
        DESTINATION include)

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/cmake/FindGMP.cmake"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils/cmake/source_of_randomness.cmake"
        DESTINATION lib/cmake/SCI)
