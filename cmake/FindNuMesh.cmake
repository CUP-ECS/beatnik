# search prefix path
set(NuMesh_PREFIX "${CMAKE_INSTALL_PREFIX}" CACHE STRING "Help cmake to find NuMesh")

# check include
find_path(NuMesh_INCLUDE_DIR NAMES NuMesh_Core.hpp HINTS ${NuMesh_PREFIX}/include)

# check lib
# find_library(MPI_Advance_LIBRARY NAMES mpi_advance
# 	HINTS ${MPI_Advance_PREFIX}/lib)

# setup found
# if (MPI_Advance_INCLUDE_DIR AND MPI_Advance_LIBRARY)
# 	set(MPI_Advance_FOUND ON)
# endif()
if (NuMesh_INCLUDE_DIR)
  set(NuMesh_FOUND ON)
  set(NuMesh_INCLUDE_DIRS ${NuMesh_INCLUDE_DIR})
endif()

# handle QUIET/REQUIRED
include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NuMesh_FOUND to TRUE
# if all listed variables are TRUE
# find_package_handle_standard_args(Numesh DEFAULT_MSG MPI_Advance_INCLUDE_DIR MPI_Advance_LIBRARY)
find_package_handle_standard_args(NuMesh DEFAULT_MSG NuMesh_INCLUDE_DIR)

# Hide internal variables
# mark_as_advanced(NuMesh_INCLUDE_DIR MPI_Advance_FOUND MPI_Advance_LIBRARY MPI_Advance_PREFIX)
mark_as_advanced(NuMesh_INCLUDE_DIR MPI_Advance_FOUND MPI_Advance_PREFIX)