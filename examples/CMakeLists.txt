include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable( rocketrig rocketrig.cpp )
target_link_libraries( rocketrig PRIVATE beatnik )

install(TARGETS rocketrig DESTINATION ${CMAKE_INSTALL_BINDIR})
