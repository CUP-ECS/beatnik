configure_file(Beatnik_Config.hpp.in Beatnik_Config.hpp)

set(HEADERS_PUBLIC
  # General problem/mesh/time integration files
  SurfaceMesh.hpp
  Operators.hpp
  ProblemManager.hpp
  Solver.hpp
  SiloWriter.hpp

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