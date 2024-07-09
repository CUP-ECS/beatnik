if (ENABLE_TESTS)

    # blt_add_executable( NAME     testComputeInterfaceVelocityPeriodicBC
    #                     SOURCES tstExactBRSolver.cpp
    #                     DEPENDS_ON beatnik gtest
    #                     OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
                        
    # blt_add_test(NAME testComputeInterfaceVelocityPeriodicBC
    #             NUM_MPI_TASKS 4
    #             COMMAND testComputeInterfaceVelocityPeriodicBC)

    blt_add_executable(NAME tstMesh
                       SOURCES tstMesh.cpp 
                       HEADERS tstMesh.hpp TestingBase.hpp 
                       DEPENDS_ON beatnik gtest
                       OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME MeshTests
                 COMMAND tstMesh)
    
    blt_add_executable( NAME     tstBoundaryCondition
                 SOURCES tstBoundaryCondition.cpp
                 HEADERS tstBoundaryCondition.hpp TestingBase.hpp
                 DEPENDS_ON beatnik gtest
                 OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})

    blt_add_test(NAME BoundaryConditionTests
                 COMMAND tstBoundaryCondition)
    
    blt_add_executable(NAME tstCutoffSolver
                      SOURCES tstCutoffSolver.cpp 
                      HEADERS tstCutoffSolver.hpp TestingBase.hpp
                      DEPENDS_ON beatnik gtest
                      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME CutoffSolverTests
                 COMMAND tstCutoffSolver)

    # blt_add_executable(NAME tstProblemManager
    #                   SOURCES tstProblemManager.cpp 
    #                   HEADERS tstProblemManager.hpp tstMesh.hpp 
    #                   DEPENDS_ON beatnik gtest)
    # blt_add_test(NAME ProblemManagerTests
    #              COMMAND tstProblemManager)

endif()
