include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_executable( test_zdot test_zdot.cpp )
target_link_libraries( test_zdot PRIVATE beatnik )

install(TARGETS test_zdot DESTINATION ${CMAKE_INSTALL_BINDIR})
