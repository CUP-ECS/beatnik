# Beatnik - A Prototype High Performance Parallel Interface Benchmark

## Description

Beatnik is a benchmark for global communication based on Pandya and Shkoller's 3D fluid interace "Z-Model" in the Cabana mesh framework [1]. The goals of Beatnik are to:
  1. Provide an interesting and meaningful benchmark for numerical methods that require global communication, for example for far-field force calculations. This includes fast fourier transforms, distance sort cutoff-based methods, and (eventually) fast multi-pole methods.
  1. Understand the performance characteristics of different parallel decompositions of the Z-Model based on both a 2D decomposition based on logical mesh location location and a space-filling curve mesh decomposition.
  1. Provide a working prototype parallel implementation of the fluid interface model that other codes can use to create multi-scale models and codes.

Beatnik uses a simple mesh-based representation of the surface manifold as a Cabana grid 2D mesh in I/J space and a regular block 2D decomposition of this manifold. The physical position of each element in the mesh is stored as a separate vector in the nodes of the mesh. This design results in simple and efficient computation and communication strategies for surface normals, artificial viscosity, and Fourier transforms elements. However, it complicates methods where the data decomposition and communication is based on the spatial location of manifold points, requiring them to either maintain a separate spatial decomposition of the surface or to continually construct a spatial decomposition. A surface mesh that decomposed the mesh by spatial location would be an interesting alternative but would have the opposite issue - communication for surface calculations would be more complex but the (expensive) far force methods that rely on spatial decompositions (e.g. distance sort and spatial tree methods like the fast multi-pole method) would be less expensive.

## Building Beatnik

Beatnik relies on multiple external packages to build, including:
  * ECP CoPA's Cabana/Grid particle and mesh framework [2]
  * UT-Knoxville's HeFFTe fast fourier transform library [3]
  * A high-performance GPU-aware MPI implementation such as OpenMPI, MPICH, or MVAPICH

To ease building Beatnik, the configs/ directory includes Spack configuration files for building in spack environments on multiple systems and test case run scripts for a variety of systems. In addition, the latest version of Spack includes a package description for directly building Beatnik. More information on building Beatnik can be found in the README.md file in the configs/ directory.

## Running Beatnik

By default, Beatnik solves a multi-mode rocket rig problem sized for a single serial CPU core with about 4GB of memory. All run-time parameters — mesh size, initial condition, physics constants, solver choice, I/O frequency, weak-scaling factor, and the FMM tunables — are read from a single input file passed to `rocketrig` as its only argument. Setting these parameters accurately generally requires expertise in fluid interface models, so we ship two examples drawn from the Z-Model papers as starting points.

The node-level parallelism / accelerator backend is selected at compile time based on which Kokkos backends are enabled in the build (priority: CUDA/HIP/SYCL > OpenMP > Threads > Serial). The selected backend is printed in the banner when `rocketrig` runs.

### Invocation

```
rocketrig <input_file>          # run with the supplied input file
rocketrig --help                # print the full input-file schema
```

To run under MPI, prepend the usual launcher (`mpirun -n N`, `flux run`, `srun`, …) — `rocketrig` itself still takes exactly one positional argument.

### Input file format

Plain text, `key = value` per line. `#` starts a comment to end-of-line. Blank lines are OK. Missing keys keep their built-in defaults, so a near-empty file is valid — override only what you care about. Unknown keys, malformed lines, and bad enum values all error with the file path, line number, key, offending value, and (for enums) the full list of accepted values, e.g. `rocketrig.in:21: invalid value for 'br_solver': 'magic' (expected one of: exact, cutoff, fmm)`.

Run `rocketrig --help` for the full schema. The key groups are:

| Group | Keys |
| --- | --- |
| Mesh / domain | `nodes`, `bounding_box`, `weak_scale` |
| Time integration | `timesteps`, `delta_t`, `write_frequency` |
| Initial condition | `initial_condition` (`cos`/`sech2`/`gaussian`/`random`), `magnitude`, `variation`, `period`, `tilt` |
| Physics / boundary | `boundary` (`periodic`/`free`), `gravity` (Gs), `atwood` |
| Solver | `solver_order` (`low`/`medium`/`high`), `br_solver` (`exact`/`cutoff`/`fmm`), `cutoff_distance`, `heffte_configuration`, `mu`, `epsilon` |
| FMM tunables (when `br_solver = fmm`) | `fmm_ncrit`, `fmm_max_depth`, `fmm_mac_theta`, `fmm_replication_depth`, `fmm_imbalance_tol`, `fmm_ncrit_tol`, `fmm_{x,y,z}{min,max}_tol` |

`br_solver = fmm` requires Beatnik to be built with Canopy support (`Beatnik_ENABLE_CANOPY=ON`).

### Example 1: periodic multi-mode rocket rig

The default test case is a cosine-distributed initial interface, periodic boundaries, low-order Z-model with the exact BR solver. The shipped [examples/01_rocketrig/rocketrig.in](examples/01_rocketrig/rocketrig.in) reproduces this configuration:

```
mpirun -n 4 bin/rocketrig examples/01_rocketrig/rocketrig.in
```

To explore variations, copy `rocketrig.in` and edit the keys you care about — e.g. set `nodes = 512` for a larger mesh, or `weak_scale = 16` with `write_frequency = 0` to scale up by 16× and skip I/O.

### Example 2: non-periodic single-mode Gaussian rollup

A single-mode `sech2` rollup that recreates the Gaussian perturbation results in Pandya and Shkoller's paper (sections 2.3 and 2.4). High-order Z-model on free boundaries; works best on a GPU accelerator since the exact high-order BR solver is compute-intensive. An input file for this case (n=64, sech2, free, atwood 0.15, μ=2, ε=2, magnitude 0.1, period 9.0):

```
nodes              = 64
initial_condition  = sech2
magnitude          = 0.1
period             = 9.0
atwood             = 0.15
boundary           = free
solver_order       = high
mu                 = 2.0
epsilon            = 2.0
```

The shipped [examples/01_rocketrig/fmm.in](examples/01_rocketrig/fmm.in) is the same problem driven by the Canopy FMM BR solver instead of the exact solver, with several FMM tunables nudged off their defaults — useful as a starting point when configuring an FMM run.

## Planned Development Steps

Beatnik is being implemented in multiple distinct steps, with associated planned releases:

  * Version 1.0 Features

    1. A low-order model implementation that relies on Cabana Grid/HeFFTe Fourier transforms for estimating velocity interface at mesh points.
    1. A high-order model implementation based on brute-force exact computation of long-range forces
    1. A medium-order model that uses the Fourier transform for estimating interface velocity and the far-field force solver for estimating how the vorticity changes at each interface point. 
    1. Support for periodic boundary conditions and free boundary conditions
    1. Simple benchmark examples including a single-mode Gaussian roll-up test and the multi-mode rocket rig experiment.
    1. Direct support for weak scaling of benchmarks through command line arguments
   
  * Version 1.1 Features

    1. Support for exact or cutoff-based BR solvers
    2. Support for Gaussian and randomized initial particle positions in the z-direction
    3. Added tests using the [BLT](https://github.com/LLNL/blt) framework

  * Version 1.X Planned Features

    1. Improved timestep, desingularization, and artificial viscosity parameter handling. The goal of this is to provide good defaults when other input parameters are changed.
    1. File-based interface initialization (also useful for checkpointing)
    1. Support for coupling with other applications through either I/O (e.g. ADIOS) or Communication (e.g. Portage) 
    1. Additional test case definitions

  * Potential later (e.g. >=2.0) features

    1. Direct fast multi-pole or P3M solver for scalable, high precision high-order model solves.
    1. Support for multiple interface manifolds in a single simulation.
    2. Support for unstructured and adaptive meshes

## Acknowledgment, Contributors, and Copyright Information

Beatnik is primarily available as open source under a 3-Clause BSD License. It is being developed at the University of New Mexico, Tennessee Tech University, and the University of Alabama under funding the U.S. Department of Energy's Predictive Science Academic Alliance Partnership III (PSAAP-III) program. Contributors to Beatnik development include:

  * Patrick G. Bridges (patrickb@unm.edu)
  * Thomas Hines (tmhines3@ua.edu)
  * Jered Dominguez-Trujillo (jereddt@unm.edu)
  * Jacob McCullough (jmccullough12@unm.edu)
  * Jason Stewart (jastewart@unm.edu)

The general structure of Beatnik and the rocketrig examples were taken from the ExaMPM proxy application (https://github.com/ECP-copa/ExaMPM) developed by the ECP Center for Particle Applications (CoPA), which was also available under a 3-Clause BSD License when used for creating application structure. 

## References

1. Gavin Pandya and Steve Shkoller. "3d Interface Models for Raleigh-Taylor Instability." Published as arxiv.org preprint https://arxiv.org/abs/2201.04538, 2022.

1. https://github.com/ECP-copa/Cabana/

1. Innovative Computing Laboratory. "heFFTe." URL: https://icl.utk.edu/fft/
