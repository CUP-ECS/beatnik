include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

blt_add_executable( NAME rocketrig
                    SOURCES rocketrig.cpp 
                    OUTPUT_DIR ${EXAMPLE_OUTPUT_DIRECTORY})

target_link_libraries( rocketrig PRIVATE beatnik )

install(TARGETS rocketrig DESTINATION ${CMAKE_INSTALL_BINDIR})
