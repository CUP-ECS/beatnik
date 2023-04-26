include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

blt_add_executable( NAME       rocketrig
                    SOURCES    rocketrig.cpp 
                    DEPENDS_ON beatnik)

blt_add_executable( NAME       meshtest
                    SOURCES    meshtest.cpp 
                    DEPENDS_ON beatnik)
