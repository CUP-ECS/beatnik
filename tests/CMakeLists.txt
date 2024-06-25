if (ENABLE_TESTS)

    blt_add_executable( NAME     testComputeInterfaceVelocityPeriodicBC
                        SOURCES tstExactBRSolver.cpp
                        DEPENDS_ON beatnik gtest
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
                        
    blt_add_test(NAME testComputeInterfaceVelocityPeriodicBC
                NUM_MPI_TASKS 4
                COMMAND testComputeInterfaceVelocityPeriodicBC)

    # blt_add_executable(NAME tstMesh
    #                    SOURCES tstMesh.cpp 
    #                    INCLUDES tstMesh.hpp 
    #                    DEPENDS_ON beatnik gtest)
    # blt_add_test(NAME MeshTests
    #              COMMAND tstMesh)

    # blt_add_executable(NAME tstProblemManager
    #                   SOURCES tstProblemManager.cpp 
    #                   INCLUDES tstProblemManager.hpp tstMesh.hpp 
    #                   DEPENDS_ON beatnik gtest)
    # blt_add_test(NAME ProblemManagerTests
    #             COMMAND tstProblemManager)

endif()
