
configure_file(Beatnik_Config.hpp.in Beatnik_Config.hpp)

set(HEADERS
  # General problem/mesh/time integration files
  Mesh.hpp
  Operators.hpp
  ProblemManager.hpp
  Solver.hpp
  SiloWriter.hpp

  # Routines to support the general Z-MOdel Solutio Approach
  TimeIntegrator.hpp
  BoundaryCondition.hpp

  # ZModel details here.
  ZModel.hpp
  ExactBRSolver.hpp
#  PvfmmBRSolver.hpp
#  VerletBRVelocity.hpp
  )

#set(SOURCES
#  Mesh.cpp
#  ProblemManager.cpp
#  )

set(DEPENDS_ON
  Cabana::cabanacore
  Cabana::Cajita
  Silo::silo 
#  pvfmm
  mpi
  )
 
if(Beatnik_ENABLE_PVFMM)
  list(APPEND HEADERS
    PvfmmBRSolver.hpp
    )
  list(APPEND DEPENDS_ON
    pvfmm
  )
endif(Beatnik_ENABLE_PVFMM)

set(INCLUDES
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  )
       
blt_add_library( NAME beatnik
                 HEADERS ${HEADERS}
                 SOURCES ${SOURCES}
                 INCLUDES ${INCLUDES}
                 DEPENDS_ON ${DEPENDS_ON} 
  )

# Need to set up a proper install target XXX
