# project settings
cmake_minimum_required(VERSION 3.18)

project(Beatnik LANGUAGES C CXX VERSION 0.1.0)

# Default to C++17 so we can use if constexpr and [[maybe_unused]]
# and so GTest/GMock can build
if (NOT BLT_CXX_STD)
    set(BLT_CXX_STD "c++17" CACHE STRING "")
endif()

include(GNUInstallDirs)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

# Enable debugging for all builds
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)

##---------------------------------------------------------------------------##
## Print the revision number to stdout
##---------------------------------------------------------------------------##
FIND_PACKAGE(Git)
IF(GIT_FOUND AND IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/.git)
  EXECUTE_PROCESS(
    COMMAND           ${GIT_EXECUTABLE} log --pretty=format:%H -n 1
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE   Beatnik_GIT_COMMIT_HASH
    )
ELSE()
  SET(Beatnik_GIT_COMMIT_HASH "Not a git repository")
ENDIF()
MESSAGE(STATUS "Beatnik Revision = '${Beatnik_GIT_COMMIT_HASH}'")

##---------------------------------------------------------------------------##
## Bring up the BLT tools
## We use LLNL BLT to help with building, linking, and testing on various HPC 
## systems. Before bringing in BLT, we set a variety of flags whch tell it the 
## featurs we're interested in.
##---------------------------------------------------------------------------##
set(ENABLE_TESTS ON)
if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    # Use internal BLT if no BLT_SOURCE_DIR is given
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/thirdparty/blt" CACHE PATH "")
    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR
            "The BLT git submodule is not present. "
            "Either run the following two commands in your git repository: \n"
            "    git submodule init\n"
            "    git submodule update\n"
            "Or add -DBLT_SOURCE_DIR=/path/to/blt to your CMake command." )
    endif()
endif()

# Disable google test pthreads in BLT to support to ensure we compile in 
# some environments, as some compilers (e.g. craycc) and BLT aren't communicating
# on linking against pthreads.
set(gtest_disable_pthreads ON CACHE BOOL "")
include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

##---------------------------------------------------------------------------##
## Get the dependent packages we need: Cabana, Silo, VTK, and ClangFormat    ##
##---------------------------------------------------------------------------##
find_package(Cabana REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()
find_package(SILO REQUIRED)

find_package(VTK REQUIRED) 

# Add VTK include directories
include_directories(${VTK_INCLUDE_DIRS})

# Get Clang Format to use for making sure the resulting code is
# properly formatted
find_package(CLANG_FORMAT 10)

# Tell CMAKE to find MPIAdvance
# find_package(MPI_Advance REQUIRED)
# if (MPI_Advance_FOUND)
#   include_directories(${MPI_Advance_INCLUDE_DIR})
# else()
#   message(FATAL_ERROR "MPI_Advance not found!")
# endif()

# Tell CMAKE to find Tessera
find_package(Tessera REQUIRED)
if (Tessera_FOUND)
  include_directories(${Tessera_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Tessera not found!")
endif()

# library
add_subdirectory(src)

# examples
add_subdirectory(examples)

# tests
add_subdirectory(tests)

# Add a target for formatting the code using Clang
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp tests/*.hpp tests/*.cpp examples/*.cpp examples/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
