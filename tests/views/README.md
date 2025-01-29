These subdirectories contain the values in the position and vorticity views after timestep 5.

The values in these views are taken for correct as of Beatnik release 1.1. 

`tstSolver.hpp`
and `tstSolver.cpp` use these files to test the correctness of the fluid interface solver.

All simulations used to generate these files used the default run settings defined in `rocketrig.cpp`,
with the following exceptions:
  - A mesh size of 64x64 was used (`-n 64`)
  - For the cutoff solver, a cutoff distance of 0.25 was used (`-c 0.25`)

The generalized run command is:
`examples/rocketrig -O {order} -b {periodic/free} -n 64 -t 5 -S {solver type} -c {0.25, if cutoff}`
 
 The naming convention for the files is as follows:
 `{position/vorticity}_{mesh_size}_r{rank}.{comm_size}.view`