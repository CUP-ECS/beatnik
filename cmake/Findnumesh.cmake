# Findnumesh.cmake
# Tries to locate the Numesh library and headers
if(NOT DEFINED NUMESH_DIR)
    execute_process(
        COMMAND spack location -i numesh
        OUTPUT_VARIABLE NUMESH_install
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(NUMESH_DIR ${NUMESH_install})
endif()

find_path(
    NUMESH_INCLUDE_DIR
    NAMES NuMesh_Core.hpp
    PATHS ${NUMESH_DIR}/include
          ENV CPATH
    NO_DEFAULT_PATH
)

# There are no library files
# find_library(
    # NUMESH_LIBRARY
    # NAMES numesh
    # PATHS ${NUMESH_DIR}/lib
    # NO_DEFAULT_PATH
# )

# Check and handle errors properly
# if (NUMESH_LIBRARY)
    # message(STATUS "Found numesh library: ${NUMESH_LIBRARY}")
# else()
    # message(FATAL_ERROR "Could not find numesh library!")
# endif()

if (NUMESH_INCLUDE_DIR)
    message(STATUS "Found numesh INCLUDE_DIR: ${NUMESH_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Could not find numesh INCLUDE_DIR!")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    numesh
    REQUIRED_VARS NUMESH_INCLUDE_DIR
)

if(numesh_FOUND)
    # set(NUMESH_LIBRARIES ${NUMESH_LIBRARY})
    set(NUMESH_INCLUDE_DIRS ${NUMESH_INCLUDE_DIR})
    mark_as_advanced(NUMESH_INCLUDE_DIR)
endif()
