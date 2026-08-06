# Expose the uppercase BEATNIK_ENABLE_CANOPY for #cmakedefine in
# Beatnik_Config.hpp.in. The Beatnik_add_dependency macro sets the
# PascalCase Beatnik_ENABLE_CANOPY cmake variable; cmakedefine is
# case-sensitive on the cmake-variable lookup, so without this alias
# the C macro stays #undef'd even when the build is +canopy.
set(BEATNIK_ENABLE_CANOPY ${Beatnik_ENABLE_CANOPY})
configure_file(Beatnik_Config.hpp.in Beatnik_Config.hpp)

# Beatnik is HEADER-ONLY: every public header is `Beatnik_`-prefixed, lives in
# this directory, and is installed as-is. There are no .cpp files and no
# compiled library — `Beatnik` is a CMake INTERFACE target.
#
# The list below is the whole library. Keep it alphabetical within each group
# and add new headers here in the same change that creates them; an installed
# tree missing a header fails only at the consumer's compile, which is a long
# way from the mistake.
set(HEADERS_PUBLIC
  # Foundations: scalars, the stub macro, the CLI enums, parameter structs.
  Beatnik_Types.hpp
  Beatnik_Params.hpp
  Profiling.hpp

  # Thin adapters for the external libraries. NOTHING else in Beatnik may name
  # a Tessera or Canopy type — see the contract comment at the top of each.
  Beatnik_MeshInterface.hpp      # Tessera: unstructured surface + topology
  Beatnik_IOInterface.hpp        # Tessera HDF5: checkpoint read/write
  Beatnik_FarFieldInterface.hpp  # Canopy: FMM far-field kernel sums

  # Distributed-memory communication points.
  Beatnik_Communication.hpp

  # Discrete differential geometry and the solution state.
  Beatnik_MeshGeometry.hpp
  Beatnik_SurfaceState.hpp
  Beatnik_VolumeProjection.hpp

  # Birkhoff-Rott: quadrature, the solver interface, its implementations,
  # and the factory.
  Beatnik_SourceQuadrature.hpp
  Beatnik_BRSolverBase.hpp
  Beatnik_BRSolverDirect.hpp
  Beatnik_BRSolverFMM.hpp
  Beatnik_CreateBRSolver.hpp

  # The z-model itself and its time integration.
  Beatnik_ZModelSolver.hpp
  Beatnik_TimeIntegrator.hpp

  # Adaptivity.
  Beatnik_AdaptiveMesh.hpp
  Beatnik_DynamicRemesh.hpp
  Beatnik_MeshQuality.hpp

  # Setup, restart, diagnostics, and the top-level driver.
  Beatnik_InitialCondition.hpp
  Beatnik_Restart.hpp
  Beatnik_Diagnostics.hpp
  Beatnik_Solver.hpp
  )

# Beatnik_BRSolverFMM.hpp is installed unconditionally: it names no Canopy type
# (that is Beatnik_FarFieldInterface.hpp's job) and Beatnik_CreateBRSolver.hpp
# guards the construction on BEATNIK_ENABLE_CANOPY, so a ~canopy build compiles
# it fine and refuses `--br-approximation fmm` at run time with a clear message.

set(DEPENDS_ON
  Cabana::Core
  Cabana::Grid
  MPI::MPI_CXX
  )

# Silo was the pre-redesign I/O path and is gone: checkpoint output is HDF5 via
# Tessera (Beatnik_IOInterface.hpp). Re-add SILO::silo here only if a Silo
# writer comes back, and restore the find_package(SILO) in the top-level
# CMakeLists.txt in the same change.

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