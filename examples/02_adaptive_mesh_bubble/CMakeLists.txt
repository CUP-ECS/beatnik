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
#
# Example 02 — adaptive-mesh rising bubble.
#
# C++/Kokkos/MPI port of zmodel3d-amr's run_adaptive_mesh_bubble.py. Unlike
# example 01, this one targets the CURRENT solver API and builds unconditionally
# under Beatnik_ENABLE_EXAMPLES=ON.
#
# It BUILDS and RUNS. The solver integrates: task T2d landed the RHS, the volume
# projection and the TVD-RK3 integrator, and T4a/T4b landed indicator-driven
# refinement and split-only dynamic remeshing. What is still stubbed are the
# follow-on mesh-quality passes (T4d, T4e) and the circulation field filter
# (T5c); `Solver::requireSupportedConfiguration` rejects a configuration that
# needs one of those before the first step, naming the option to change and the
# task that owns it. The DEFAULT configuration is currently one of those, because
# `--isotropic-cleanup` defaults on — so a real invocation either integrates to
# completion or exits with that message, never dies in an unnamed stub.
#
# The end-to-end regression cases are NOT registered here. They live in
# tests/CMakeLists.txt as standalone self-validating binaries under
# tests/regression_tests/, each with its own gold set (e.g.
# tests/regression_tests/direct-solve-10-steps/gold/), so that the gate runs one
# binary per case across the backend x rank sweep rather than shelling out to
# this driver. Those cases are in the `regression` tier, which is the ship gate,
# so adding one requires the user's explicit confirmation (CLAUDE.md "Minimum
# test set").
#
# The accepted arguments are documented in the top-level README.md and in
# InputFile.hpp::printSchema, and all three must be updated together.
############################################################################

# The next four lines let this directory be configured standalone, for testing
# the exported Beatnik target.
cmake_minimum_required(VERSION 3.16)
project(Beatnik_AdaptiveMeshBubble)
if(NOT TARGET Beatnik)
  find_package(Beatnik)
endif()

add_executable(adaptive_mesh_bubble adaptive_mesh_bubble.cpp)
target_link_libraries(adaptive_mesh_bubble Beatnik)

# InputFile.hpp is included as "InputFile.hpp" from the driver, so the source
# directory must be on the include path. Named explicitly rather than relying on
# the compiler's quoted-include search, which differs between compilers.
target_include_directories(adaptive_mesh_bubble
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# A cheap, always-passing smoke check: `--help` must exit 0 and print the
# schema. It exercises option-table construction (every setter binds a real
# field) without touching a stub. Tier `unit` — diagnostic, not gating.
add_test(NAME Beatnik_Example_02_adaptive_mesh_bubble_help
  COMMAND adaptive_mesh_bubble --help)
set_tests_properties(Beatnik_Example_02_adaptive_mesh_bubble_help PROPERTIES
  LABELS unit
  PASS_REGULAR_EXPRESSION "Usage: adaptive_mesh_bubble")

if(Beatnik_INSTALL_EXAMPLES)
  install(TARGETS adaptive_mesh_bubble
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
