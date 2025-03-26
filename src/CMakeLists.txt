
configure_file(Beatnik_Config.hpp.in Beatnik_Config.hpp)

set(HEADERS
  # General problem/time integration files
  Beatnik_ArrayUtils.hpp
  Beatnik_Types.hpp
  Operators.hpp
  ProblemManager.hpp
  Solver.hpp
  SiloWriter.hpp
  VTKWriter.hpp

  # Mesh files
  CreateMesh.hpp
  MeshBase.hpp
  StructuredMesh.hpp
  UnstructuredMesh.hpp

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
 
set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
       
add_library( beatnik INTERFACE ${SOURCES} )
target_link_libraries( beatnik INTERFACE ${DEPENDS_ON} )
# target_link_libraries( beatnik INTERFACE ${MPI_ADVANCE_LIBRARY} )
target_link_libraries( beatnik INTERFACE ${VTK_LIBRARIES})
# Must include these libraries to link with VTK correctly
target_link_libraries( beatnik INTERFACE ${SEACASIoss_LIBRARIES})
# target_link_libraries( beatnik INTERFACE OpenGL::GL) 
target_include_directories( beatnik INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
	  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
	  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}> )
# target_include_directories(beatnik INTERFACE ${MPI_ADVANCE_INCLUDE_DIRS})
target_include_directories(beatnik INTERFACE ${NUMESH_INCLUDE_DIRS})

install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(TARGETS beatnik LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
