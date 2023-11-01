# List of potential Beatnik projects

1. Implmenent unit test cases for core Beatnik abstractions, including the mesh setup, 
   time integration, Z-Model derivative calculation, BR solver, and I/O modules. Prior matlab
   implementations of the z-model can be used to generate gold standard testcases to compare 
   against.
 
1. Instrument and benchmark Beatnik low order weak scaling on modern GPU systems, including 
   examining how changing HeFFTe parameters (reordering, etc.) impacts application performance.
   This effort would lead to larger examininations and optimizaitons of FFT performance, 
   for example around communication strategies in FFT implementations. 

1. Instument and optimize the high-order velocity calculation currently used in the Exact BR
   Solver (and that will also be used in other high-order solves). The current implementation
   relies on Kokkos atomic views to calculate reductions, while switching to proper Kokkos 
   reductions and multi-level parallelism would improve both performance and perfomrance portability.
   GPU Gems 3 has a discussion of a high-level algorithm that could be used for this as well.

1. Implement and test more initial conditions, including gaussian variation of the initial surface

1. Implement and test file input capability to initialize ProblemManager state and other test case
   state, and demonstrate its usage for checkpoint/restart capability

1. Implement a structured input deck format for interface problems, either as a straight text file
   or as a python/pykokkos script that initializes solver state and runs test cases more generally
   than the current C++-based driver approach.
