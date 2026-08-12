#------------------------------------------------------------------------------#
# Project settings
#
# This cmake organization, macros, and setup is copied from the
# Cabana library: https://github.com/ECP-copa/Cabana/
#------------------------------------------------------------------------------#
cmake_minimum_required(VERSION 3.18)

project(Beatnik LANGUAGES C CXX VERSION 0.1.0)

# If the user doesn't provide a build type default to release
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
  #release comes with -O3 by default
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)

# Disable verbose makefiles
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose Makefiles" OFF)

# use gnu standard install directories
include(GNUInstallDirs)
set(Beatnik_INSTALL_PACKAGEDIR "${CMAKE_INSTALL_DATADIR}/cmake/Beatnik" CACHE PATH "Install location of CMake target files")

include(FeatureSummary)

# add local cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# find kokkos
find_package(Kokkos 5 REQUIRED)

# set supported kokkos devices
set(BEATNIK_SUPPORTED_DEVICES SERIAL THREADS OPENMP CUDA HIP SYCL OPENMPTARGET)

# check user required kokkos device types
foreach(_device ${BEATNIK_SUPPORTED_DEVICES})
  option(Beatnik_REQUIRE_${_device} "Build Beatnik with required Kokkos ${device} support" OFF)
  if(Beatnik_REQUIRE_${_device})
    kokkos_check( DEVICES ${_device} )
  endif()
endforeach()

# ensure that we can use lambdas
if(Kokkos_ENABLE_CUDA)
  kokkos_check(OPTIONS CUDA_LAMBDA)
endif()

# find cabana
find_package(Cabana REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()

# find MPI
find_package(MPI COMPONENTS CXX )

# find Tessera. REQUIRED, not optional: Tessera is the distributed unstructured
# surface Beatnik is built on (Beatnik_MeshInterface.hpp / Beatnik_IOInterface.hpp),
# not an accelerator that can be compiled out the way Canopy can.
find_package(Tessera REQUIRED)

# standard dependency macro
macro(Beatnik_add_dependency)
  cmake_parse_arguments(BEATNIK_DEPENDENCY "" "PACKAGE;VERSION" "COMPONENTS" ${ARGN})
  find_package( ${BEATNIK_DEPENDENCY_PACKAGE} ${BEATNIK_DEPENDENCY_VERSION} QUIET COMPONENTS ${BEATNIK_DEPENDENCY_COMPONENTS} )
  string(TOUPPER "${BEATNIK_DEPENDENCY_PACKAGE}" BEATNIK_DEPENDENCY_OPTION )
  option(
    Beatnik_REQUIRE_${BEATNIK_DEPENDENCY_OPTION}
    "Require Beatnik to build with ${BEATNIK_DEPENDENCY_PACKAGE} support" ${BEATNIK_DEPENDENCY_PACKAGE}_FOUND)
  if(Beatnik_REQUIRE_${BEATNIK_DEPENDENCY_OPTION})
    find_package( ${BEATNIK_DEPENDENCY_PACKAGE} ${BEATNIK_DEPENDENCY_VERSION} REQUIRED COMPONENTS ${BEATNIK_DEPENDENCY_COMPONENTS} )
  endif()
  set(Beatnik_ENABLE_${BEATNIK_DEPENDENCY_OPTION} ${${BEATNIK_DEPENDENCY_PACKAGE}_FOUND})
endmacro()

# Optional dependency: Canopy
Beatnik_add_dependency(PACKAGE Canopy)
set_package_properties(Canopy PROPERTIES
  TYPE OPTIONAL
  PURPOSE "Fast multipole solver for far-field forces"
)

#------------------------------------------------------------------------------#
# Architecture
#------------------------------------------------------------------------------#
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(Beatnik_BUILD_MARCH "" CACHE STRING "Arch to use with -march= (if empty CMake will try to use 'native') in release build and only release build")

  # Try -march first. On platforms that don't support it, GCC will issue
  # a hard error, so we'll know not to use it.
  if(Beatnik_BUILD_MARCH)
    set(INTERNAL_Beatnik_BUILD_MARCH ${Beatnik_BUILD_MARCH})
  else()
    set(INTERNAL_Beatnik_BUILD_MARCH "native")
  endif()

  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-march=${INTERNAL_Beatnik_BUILD_MARCH}" COMPILER_SUPPORTS_MARCH)
  if(COMPILER_SUPPORTS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=${INTERNAL_Beatnik_BUILD_MARCH}")
  elseif(Beatnik_BUILD_MARCH)
    message(FATAL_ERROR "The flag -march=${INTERNAL_Beatnik_BUILD_MARCH} is not supported by the compiler")
  else()
    unset(INTERNAL_Beatnik_BUILD_MARCH)
  endif()
endif()

##---------------------------------------------------------------------------##
## Code coverage testing
##---------------------------------------------------------------------------##
option(Beatnik_ENABLE_COVERAGE_BUILD "Do a coverage build" OFF)
if(Beatnik_ENABLE_COVERAGE_BUILD)
  message(STATUS "Enabling coverage build")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -O0")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} --coverage")
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  option(WITH_ASAN "Build with address sanitizer" OFF)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT APPLE)
  option(WITH_MSAN "Build with memory sanitizer (experimental; requires a memory-sanitized Python interpreter)" OFF)
endif()

if(WITH_ASAN AND WITH_MSAN)
  message( FATAL_ERROR "Address sanitizer and memory sanitizer cannot be enabled simultaneously")
endif()
if(WITH_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=address")
endif()

if(WITH_MSAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory -fno-omit-frame-pointer")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -fsanitize=memory")
endif()

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
## Diagnostics / profiling instrumentation
##---------------------------------------------------------------------------##
# Mirrors Canopy_ENABLE_PROFILING / Canopy_PROFILING_LEVEL semantics:
#
#   Beatnik_ENABLE_PROFILING (OFF by default) — kill switch. OFF forces
#     the resolved level to 0 regardless of any Beatnik_PROFILING_LEVEL.
#   Beatnik_PROFILING_LEVEL (cache string "") — detail level:
#     0=off, 1=basic (e.g. FmmBRSolver auto_maintain action log),
#     2=detailed, 3=verbose (both reserved for future per-phase timers).
#     If left "" and ENABLE_PROFILING is ON, defaults to 1.
#
# When the resolved level is > 0, BEATNIK_ENABLE_PROFILING and
# BEATNIK_PROFILING_LEVEL=<level> are added as compile definitions both
# at the top level and on the Beatnik INTERFACE target so consumers see
# them too.
option(Beatnik_ENABLE_PROFILING "Profiling" OFF)
set(Beatnik_PROFILING_LEVEL "" CACHE STRING
    "Profiling detail level: 0=off, 1=basic, 2=detailed, 3=verbose (reserved)")
set_property(CACHE Beatnik_PROFILING_LEVEL PROPERTY STRINGS "" 0 1 2 3)

if(NOT Beatnik_ENABLE_PROFILING AND NOT Beatnik_PROFILING_LEVEL STREQUAL "")
  set(_beatnik_profiling_level 0)
elseif(Beatnik_PROFILING_LEVEL STREQUAL "")
  if(Beatnik_ENABLE_PROFILING)
    set(_beatnik_profiling_level 1)
  else()
    set(_beatnik_profiling_level 0)
  endif()
else()
  set(_beatnik_profiling_level ${Beatnik_PROFILING_LEVEL})
endif()

if(_beatnik_profiling_level GREATER 0)
  set(Beatnik_ENABLE_PROFILING ON CACHE BOOL "Profiling" FORCE)
  add_compile_definitions(BEATNIK_ENABLE_PROFILING)
  add_compile_definitions(BEATNIK_PROFILING_LEVEL=${_beatnik_profiling_level})
endif()

add_feature_info(Profiling Beatnik_ENABLE_PROFILING
    "Enable Beatnik diagnostics (level=${_beatnik_profiling_level})")

#------------------------------------------------------------------------------#
# Tests and Documentation
#------------------------------------------------------------------------------#
# enable unit tests
option(Beatnik_ENABLE_TESTING "Build tests" OFF)
add_feature_info(Tests Beatnik_ENABLE_TESTING "Build unit tests (requires GTest)")

# Set installing tests to off by default
option(Beatnik_INSTALL_TEST_EXECUTABLES "Install Beatnik test executables" OFF)

if(Beatnik_ENABLE_TESTING)
  find_package(GTest 1.10 REQUIRED)
  # Workaround for FindGTest module in CMake older than 3.20
  if(TARGET GTest::gtest)
    set(gtest_target GTest::gtest)
  elseif(TARGET GTest::GTest)
    set(gtest_target GTest::GTest)
  else()
    message(FATAL_ERROR "bug in GTest find module workaround")
  endif()
  set(TEST_HARNESS_DIR ${CMAKE_SOURCE_DIR}/cmake/test_harness)
  include(cmake/test_harness/test_harness.cmake)
  add_subdirectory(tests)
  enable_testing()
endif()

# enable doxygen
option(Beatnik_ENABLE_DOXYGEN "Build documentation" OFF)
add_feature_info(Documentation Beatnik_ENABLE_DOXYGEN "Build documentation (requires Doxygen)")
if(Beatnik_ENABLE_DOXYGEN)
  find_package(Doxygen REQUIRED)
  doxygen_add_docs(doxygen src src)
endif()

##---------------------------------------------------------------------------##
## Libraries and Examples
##---------------------------------------------------------------------------##

if(MPI_FOUND)
  add_subdirectory(src)
else()
  message(FATAL_ERROR "Beatnik requires MPI")
endif()

option(Beatnik_ENABLE_EXAMPLES "Build examples" OFF)
# Set installing examples to off by default
option(Beatnik_INSTALL_EXAMPLES "Install Beatnik example executables" OFF)
add_feature_info(Examples Beatnik_ENABLE_EXAMPLES "Build examples")
if(Beatnik_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()

##---------------------------------------------------------------------------##
## Package Configuration
##---------------------------------------------------------------------------##
write_basic_package_version_file("BeatnikConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BeatnikConfig.cmakein
  ${CMAKE_CURRENT_BINARY_DIR}/BeatnikConfig.cmake @ONLY)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/BeatnikConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/BeatnikConfigVersion.cmake"
  DESTINATION ${Beatnik_INSTALL_PACKAGEDIR})

feature_summary(INCLUDE_QUIET_PACKAGES WHAT ALL)
