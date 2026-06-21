# Expose the uppercase BEATNIK_ENABLE_CANOPY for #cmakedefine in
# Beatnik_Config.hpp.in. The Beatnik_add_dependency macro sets the
# PascalCase Beatnik_ENABLE_CANOPY cmake variable; cmakedefine is
# case-sensitive on the cmake-variable lookup, so without this alias
# the C macro stays #undef'd even when the build is +canopy.
set(BEATNIK_ENABLE_CANOPY ${Beatnik_ENABLE_CANOPY})
configure_file(Beatnik_Config.hpp.in Beatnik_Config.hpp)

set(HEADERS_PUBLIC
  # General problem/mesh/time integration files
  SurfaceMesh.hpp
  Operators.hpp
  ProblemManager.hpp
  Solver.hpp
  SiloWriter.hpp
  Profiling.hpp

  # Routines to support the general Z-Model Solution Approach
  TimeIntegrator.hpp
  BoundaryCondition.hpp

  # ZModel details here.
  ZModel.hpp

  # BR Solver base classes
  BRSolverBase.hpp
  CreateBRSolver.hpp

  # Exact BR Solver
  ExactBRSolver.hpp

  # Cutoff BR Solver
  CutoffBRSolver.hpp
  SpatialMesh.hpp
  HaloComm.hpp
  )

if(Beatnik_ENABLE_CANOPY)
  list(APPEND HEADERS_PUBLIC FmmBRSolver.hpp)
endif()

#set(SOURCES
#  Mesh.cpp
#  ProblemManager.cpp
#  )

set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  SILO::silo
  MPI::MPI_CXX
  )

if(Beatnik_ENABLE_CANOPY)
  list(APPEND DEPENDS_ON Canopy::Canopy)
endif()
 
set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
       
add_library( Beatnik INTERFACE )

# Require minimum of C++17
set_target_properties(Beatnik PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_17)

# Propagate the resolved profiling level to consumers of Beatnik
# (rocketrig, tests, external projects) so the same #ifdef
# BEATNIK_ENABLE_PROFILING / BEATNIK_PROFILING_LEVEL checks fire in
# their TUs. _beatnik_profiling_level is computed at the top level.
if(_beatnik_profiling_level GREATER 0)
  target_compile_definitions(Beatnik INTERFACE BEATNIK_ENABLE_PROFILING)
  target_compile_definitions(Beatnik INTERFACE
    BEATNIK_PROFILING_LEVEL=${_beatnik_profiling_level})
endif()

# Optional: per-step FMM particle-snapshot dumps in FmmBRSolver.hpp, feeding the
# offline FMM-vs-exact replay harness (canopy/examples/04_nan_replay). OFF by
# default; enable with -DBeatnik_ENABLE_FMM_SNAPSHOT=ON. Requires profiling for
# the step number; a no-op outside the snapshot step window (see FmmBRSolver.hpp).
option(Beatnik_ENABLE_FMM_SNAPSHOT
  "Dump per-step FMM particle snapshots for the offline replay harness" OFF)
if(Beatnik_ENABLE_FMM_SNAPSHOT)
  target_compile_definitions(Beatnik INTERFACE BEATNIK_FMM_SNAPSHOT_DEBUG)
endif()

target_link_libraries( Beatnik INTERFACE ${DEPENDS_ON} )

target_include_directories( Beatnik INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

install(TARGETS Beatnik
  EXPORT Beatnik_Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  

install(EXPORT Beatnik_Targets
  FILE Beatnik_Targets.cmake
  NAMESPACE Beatnik::
  DESTINATION ${Beatnik_INSTALL_PACKAGEDIR})

install(FILES ${HEADERS_PUBLIC} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# install(FILES ${HEADERS_IMPL} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/impl/)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/Beatnik_Config.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)