# List of potential Beatnik projects

## Small-to-Midsized projects suitable for students and student group class projects
1. Implmenent unit test cases for core Beatnik abstractions, including the mesh setup, 
   time integration, Z-Model derivative calculation, BR solver, and I/O modules. Prior matlab
   implementations of the z-model can be used to generate gold standard testcases to compare 
   against.
 
1. Instrument and benchmark Beatnik low order weak scaling on modern GPU systems, including 
   examining how changing HeFFTe parameters (reordering, etc.) impacts application performance.

1. Instrument Beatnik with LogGOPSim to collect communication traces, and examine how changes in
   network performance characteristics impact the perfomrance of different Beatnik solvers.

1. Instument and optimize the high-order velocity calculation currently used in the Exact BR
   Solver (and that will also be used in other high-order solves). The current implementation
   relies on Kokkos atomic views to calculate reductions, while switching to proper Kokkos 
   reductions and multi-level parallelism would improve both performance and perfomrance portability.
   GPU Gems 3 has a discussion of a high-level algorithm that could be used for this as well.

1. Implement and initial conditions for new tests for example gaussian variation of the initial 
   surface and using this to estimate fluid mixing rates along an interface.

1. Implement and test file input capability to initialize ProblemManager state and other test case
   state, and demonstrate its usage for checkpoint/restart capability.

1. Implement a python/pykokkos interface to the solver that allows problems to be initialized
   (and saved if checkpointing support is added) and solved from a simple python interface, 
   providing more general, easier examples than the current C++ driver-based approach. 

1. Add support for additional boundary conditions, for example spherical boundary conditions for
   a simple latitude/longitude sphere configuration, or specifying boundary conditions from 
   pykokkos code (if python support was implemented).

1. Compare perfomrance and accuracy of low, medium, and high order solves under different initial
   conditions and parameters.

## Larger projects (appropriate for an MS thesis or as a portion of a Ph.D. dissertation)

1. Rearchitect the Z-Model into separate velocity and vorticity solves that can have different
   decompositions are explicitly coupled together to migrate data between the two solves.

1. Implement a velocity solver that uses a 3D mesh to track points in the velocity solve including
   finding nearby points. This would allow for spatial decomposition of points and reduced high-order 
   solution order approaches.

1. Add support for multiple interface surfaces in the same solution, building on the description of this
   in the original Z-Model paper

1. Combine a Z-model example (e.g. the fluid mixing rate example) with a fluid solver benchmark, 
   including representative data movement between the two solvers, to provide a coupled code 
   communication benchmark //even if migrated data// isn't actually used by either solver.

1. Add remeshing support to regenerate the interface surface and node distribution along that surface
   as the simulation runs, including adding additional points and/or moving points to areas of 
   high vorticity (where interface rollup happens).
