# project settings
cmake_minimum_required(VERSION 3.18)

project(Beatnik LANGUAGES C CXX VERSION 0.1.0)

include(GNUInstallDirs)

# find dependencies
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH} ${caliper_DIR})

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
## Get the dependent packages we need: Cabana, Silo, and ClangFormat         ##
##---------------------------------------------------------------------------##
find_package(Cabana REQUIRED COMPONENTS Cabana::Grid Cabana::Core)
if( NOT Cabana_ENABLE_MPI )
  message( FATAL_ERROR "Cabana must be compiled with MPI" )
endif()
if( NOT Cabana_ENABLE_GRID )
  message( FATAL_ERROR "Cabana must be compiled with Grid support" )
endif()
find_package(SILO REQUIRED)

# Get Clang Format to use for making sure the resulting code is
# properly formatted
find_package(CLANG_FORMAT 10)

# Find Caliper
find_package(caliper REQUIRED)

# library
add_subdirectory(src)

# examples
add_subdirectory(examples)

# tests - currently disabled
# add_subdirectory(tests)

# Add a target for formatting the code using Clang
if(CLANG_FORMAT_FOUND)
  file(GLOB_RECURSE FORMAT_SOURCES src/*.cpp src/*.hpp tests/*.hpp tests/*.cpp examples/*.cpp examples/*.hpp)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXECUTABLE} -i -style=file ${FORMAT_SOURCES}
    DEPENDS ${FORMAT_SOURCES})
endif()
