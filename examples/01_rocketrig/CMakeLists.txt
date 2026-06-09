############################################################################
# Copyright (c) 2025 by the Beatnik authors                                #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Beatnik library. Beatnik is distributed under a #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################

# the next for lines are just, so that we can use this as
# a standalone CMakeLists.txt for testing of the Target export
cmake_minimum_required(VERSION 3.16)
project(Beatnik_Rocketrig)
if(NOT TARGET Beatnik)
  find_package(Beatnik)
endif()

add_executable(rocketrig rocketrig.cpp)
target_link_libraries(rocketrig Beatnik)
add_test(NAME Beatnik_Example_01_rocketrig COMMAND ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG}
${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:rocketrig> ${MPIEXEC_POSTFLAGS})
set_tests_properties(Beatnik_Example_01_rocketrig PROPERTIES PROCESSORS ${MPIEXEC_MAX_NUMPROCS})
if(Beatnik_INSTALL_EXAMPLES)
      install(TARGETS rocketrig
              RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
