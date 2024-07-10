if (ENABLE_TESTS)

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
    

    blt_add_executable(NAME tstCutoffBRSolver
                      SOURCES tstCutoffBRSolver.cpp 
                      HEADERS tstCutoffBRSolver.hpp TestingBase.hpp
                      DEPENDS_ON beatnik gtest
                      OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME CutoffBRSolverTests
                 COMMAND tstCutoffBRSolver)
    

    blt_add_executable( NAME     tstExactBRSolver
                        SOURCES tstExactBRSolver.cpp
                        HEADERS tstExactBRSolver.hpp TestingBase.hpp
                        DEPENDS_ON beatnik gtest
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})             
    blt_add_test(NAME ExactBRSolverTests
                 COMMAND tstExactBRSolver)
    

    blt_add_executable( NAME     tstZModel
                        SOURCES  tstZModel.cpp
                        HEADERS  tstZModel.hpp TestingBase.hpp
                        DEPENDS_ON beatnik gtest
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY})
    blt_add_test(NAME ZModelTests
                 COMMAND tstZModel)

    # blt_add_executable(NAME tstProblemManager
    #                   SOURCES tstProblemManager.cpp 
    #                   HEADERS tstProblemManager.hpp tstMesh.hpp 
    #                   DEPENDS_ON beatnik gtest)
    # blt_add_test(NAME ProblemManagerTests
    #              COMMAND tstProblemManager)

endif()
