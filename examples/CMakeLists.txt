include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Rocketrig
blt_add_executable( NAME rocketrig
                    SOURCES rocketrig.cpp 
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY})
target_link_libraries( rocketrig PRIVATE beatnik )
install(TARGETS rocketrig DESTINATION ${CMAKE_INSTALL_BINDIR})

# Rising bubble
blt_add_executable( NAME risingbubble
                    SOURCES risingbubble.cpp 
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY})
target_link_libraries( risingbubble PRIVATE beatnik )
install(TARGETS risingbubble DESTINATION ${CMAKE_INSTALL_BINDIR})
