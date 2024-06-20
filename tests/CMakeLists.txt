blt_add_executable( NAME     multTest
                    SOURCES tstExactBRSolver.hpp
                    DEPENDS_ON beatnik gtest
                    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
                    
blt_add_test(NAME multTest
            COMMAND multTest)

# blt_add_executable(NAME tstMesh
#                    SOURCES tstMesh.cpp 
#                    INCLUDES tstMesh.hpp 
#                    DEPENDS_ON beatnik gtest)
# blt_add_test(NAME MeshTests
#              COMMAND tstMesh)

#blt_add_executable(NAME tstProblemManager
#                   SOURCES tstProblemManager.cpp 
#                   INCLUDES tstProblemManager.hpp tstMesh.hpp 
#                   DEPENDS_ON beatnik gtest)
#blt_add_test(NAME ProblemManagerTests
#             COMMAND tstProblemManager)
