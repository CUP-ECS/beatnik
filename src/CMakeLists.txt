set(HEADERS
  # General problem/mesh/time integration files
  Mesh.hpp
  ProblemManager.hpp
  Solver.hpp
  SiloWriter.hpp

  # Routines to support the general Z-MOdel Solutio Approach
  TimeIntegrator.hpp
  BoundaryCondition.hpp
  ArtificialViscosity.hpp
#  Interpolation.hpp

  # ZModel details here.
  ZModel.hpp
  # DirectBRVelocity.hpp
  # FMMBRVelocity.hpp
  # VerletBRVelocity.hpp
  )

set(SOURCES
  Mesh.cpp
  ProblemManager.cpp
  )

set(DEPENDS_ON
  Cabana::cabanacore
  Cabana::Cajita
  Silo::silo 
  mpi
  )
 
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
