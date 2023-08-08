# Beatnik - A  prototype High Performance Parallel Interface Benchmark

## Description

Beatnik is a benchmark for global communication based on Pandya and Shkoller's 3D fluid interace "Z-Model" in the Cabana/Cajita mesh framework [1]. The goals 
of Beatnik are to:
  1. Provide an interesting and meaningful benchmark for numerical methods that require global communication, particularly fast fourier transforms and fast multipole methods.
  1. Understand the performance characteristics of different parallel decompositions of the Z-Model based on both a 2D decomposition based on logical mesh location location and a space-filling curve mesh decomposition.
  1. Provide a working prototype parallel implementation of the fluid interface model that other implementations can use to understand the implementation.  

The initial Beatnik implementation of the Z-Model uses a simple mesh-based representation of the surface manifold with a regular decomposition as a Cajita 2D mesh in I/J space and the physical position of each element in the mesh stored as a separate vector in the nodes of the mesh. This is efficient for the low-order z-model, as the computation of surface normals, artificial viscosity, and Fourier transforms for estimating interface velocities are straightforward in this representation. 

Because Beatnik does not yet include a spatial mesh decomposition, its support for scalable far-field solvers in the higher-order interface solution models (e.g. the fast multipole method, P3M, or distance-sorting cutoff-based methods) is limited. In particular, Beatnik 1.0 currently only supports either O(N^2) brute-force calculation of far-field forces or the use of an external far-field force solver that re-sorts the mesh at each derivative calculation.

In the future, we plan to add the ability to decompose the Beatnik mesh spatially by adding a ParticleMesh abstraction that the implements the portions of Cajita meshes Beatnik requires using Cabana particle abstractions. This will in turn enable the direct implementation of scalable far-field force methods. This work is planned for Beatnik 2.0.
 
## Building Beatnik

Beatnik relies on multiple external packages to build, including:
  * LLNL's build, link, test (BLT) library [2]
  * ECP CoPA's Cabana/Cajita particle and mesh framework [3]
  * UT-Knoxville's HeFFTe fast fourier transform library [4]
  * A high-performance MPI implementation such as OpenMPI, MPICH, or MVAPICH

To ease building Beatnik, the configs/ directory includes Spack configuration files for building in spack environments on multiple systems and test case run scripts for those systems, as well as a spack package description for directly building Beatnik. This spack package will be contributed back to the mainline Spack repository following the first public Beatnik release.

### Building Beatnik in a Spack Environment

Assuming that you have Spack already installed on your HPC systems (as described in https://spack.readthedocs.io), you can use spack to create an environment for building and developing spack as follows:

  1. If not checked out from git recursively, checkout all needed Beatnik submodules, e.g. `git submodule init && git submodule update --recursive`
  1. Create a build directory for housing the Spack environment and housing the out-of-source build, e.g. `mkdir build-hopper` on the UNM hopper compute cluster.
  1. Copy the appropriate spack.yaml file from configs/[systemname]/ to spack.yaml in the newly-created build directory, e.g. `cp configs/unm-hopper/spack.yaml build-hopper/`
  1. Perform any compiler setup needed using the system module system, as spack environments do not necessarily configure the compiler. This could include installing a fortran compiler on MacOS systems (e.g. using homebrew), or loading the proper compiler module on an HPC system. This compiler should be compatible with one used in the spack.yaml file chosen, and ideally described in a README.md file in the associated configs/ directory
  1. Change directory to the created build directory and create a spack environment in which to build Beatnik in that directory, e.g. `cd build-hopper; spack env create -d . spack.yaml`
  1. Activate, concretize, and install the resulting environment, e.g. `spack env activate -d . && spack concretize && spack install`
  1. Run cmake and make to create the appropriate Makefiles and build using them, e.g. `cmake .. && make`.
  
### Building Beatnik directly using a Spack package description

XXX 

### Developing Beatnik using a Spack package description

XXX 

### Beatnik Build-Time Configuration Options

## Running Beatnik

By default, Beatnik solves a simple multi-mode rocket rig problem sized for a 
single serial CPU core with approximately 4GB of memory. It also includes 
command line options to change initial problem state, I/O frequency, and to 
weak-scale scale up the initial problem to larger number of processes.

### General command line parameters

  * `-x [cuda|threads|serial]` - The node-level parallelism/accelerator backend to use
  * `-F [write-frequency]` - Interval between timesteps when I/O is written
  * `-O [solution order]` - Order of solver to use ('high', 'medium', or 'low'). 'low' is the default.
  * `-w [weak scaling factor] - Scale up the problem specification, including the x/y bounding box, to be N times larger

### Problem-specific command line parameters

  * `-n [i/j mesh dimension ]` - Number of points on the interface manifold in the I and J dimensions
  * `-t [timesteps] - number of timesteps to simulate
  * `-I [interface initialization]` - Function to use for interface initial condition. Currently only 'cos' and 'sech2' are supported.
  * `-m [magnitude]` - The maximum magnitude of the initialization function. 
  * `-p [period]` - The number of periods of the interface in the initial bounding box
  * `-a [atwood]` - Atwood's constant for the difference in pressure between the two fluids 
  * `-g [gravity]` - Gravitational acceleration in the -Z direction
  * `-a [atwood]` -  Atwood's constant for the difference in pressure between the two fluids 
  * `-M [mu]` - Mu, the artificial viscosity constant used in the Z-Model
  * `-e [epsilon]` - Epsilon, the desingularization constant used in the Z-Model expressed as a fraction of the distance between interface mesh points
  
### Example 1: Periodic Multi-mode Rocket Rig
The simplest test case and the one to which the rocketrig example program defaults is an initial interface distributed according to a cosine function. Simple usage examples:
  1. Serial execution: `bin/rocketrig -x serial`
  1. Cuda execution (on systems with GPUs) with a 512x512 mesh: `bin/rocketrig -x cuda -n 512`
  1. Cuda execution with a 1024x1024 problem scaled up to be sixteen times as large in terms of bounding box and number of total points with no I/O: bin/rocketrig -x cuda -n 1024 -F 0 -w 16`

### Example 2: Non-periodic Single-mode Gaussian Rollup
Another test case is a single-mode rollup test where the intitial interface is set according to a hyperbolic secant function. This testcase recreates the the gaussian perturbation results in Panda and Shkoller's paper from sections 2.3 and 2.4.  To run this testcase with a high-order model, use the following command line parameters. Note that this works best with a GPU accelerator, as the exact high-order far field force solver is very compute intensive and is generally impractical for non-trivial mesh sizes without GPU acceleration:
`bin/rocketrig -x cuda -O high -n 64 -I sech2 -m 0.1 -p 9.0 -b free -a 0.15 -M 2 -e 2`

## Planned Development Steps

Beatnik is being implemented in multiple distinct steps, with associated planned releases:

  * Version 1.0 Features

    1. A low-order model implementation that relies on Cajita/HeFFTe Fourier transforms for estimating velocity interface at mesh points.
    1. A high-order model implementation based on brute-force exact computation of long-range forces
    1. A medium-order model that uses the Fourier transform for estimating interface velocity and the far-field force solver for estimating how the vorticity changes at each interface point. 
    1. Support for periodic boundary conditions and free boundary conditions
    1. A few simple benchmark examples, including a single-mode gaussian roll-up test and the multi-mode rocket rig experiment.
    1. Direct support for weak scaling of benchmarks through command line arguments

  * Version 1.X Planned Features

    1. A cutoff-based approach for calculating far-field forces using the Cabana particle framework that accelerates far-field force calculations.
    1. Improved timestep, desingularization, and artificial viscosity handling to provide good defaults for the input parameters given
    1. Additional interface initialization options, including gaussian random and file-based interface initialization (also useful for checkpointing)
    1. Support for coupling with other applications through either I/O (e.g. ADIOS) or Communication (e.g. Portage) approaches
    1. Additional test cases definitions

  * Version 2.0 Planned Features

    1. Spatial partitioning of the mesh using a space-filling curve to better optimize the high-order model
    1. Direct fast multipole or P3M solver for scalable, high precision high-order model solves.

## Acknowledgement, Contributors, and Copyright Information

Beatnik is primarily availble as open source under a 3-Clause BSD License. It is being developed at the University of New Mexico, University of Tennessee at Chatanooga, and the University of Alabama under funding the U.S. Department of Energy's Predictive Science Academic Alliance Partnership III (PSAAP-III) program. Contributors to Beatnik development include the following:

  * Patrick G. Bridges (patrickb@unm.edu)
  * Thomas Hines (thomas-hines-01@utc.edu)
  * Jered Dominguez-Trujillo (jereddt@unm.edu)
  * Jacob McCullough (jmccullough12@unm.edu)
  * Jason Stewart (jastewart@unm.edu)

The general structure of Beatnik and the rocketrig examples were taken from the ExaMPM proxy application (https://github.com/ECP-copa/ExaMPM) developed by the ECP Center for Particle Applications (CoPA), which was also available under a 3-Clause BSD License when used for creating application structure. 

## References

1. Gavin Pandya and Steve Shkoller. "3d Interface Models for Raleigh-Taylor Instability." Published as arxiv.org preprint https://arxiv.org/abs/2201.04538, 2022.

1. https://github.com/LLNL/blt

1. https://github.com/ECP-copa/Cabana/

1. Innovative Computing Laboratory. "heFFTe." URL: https://icl.utk.edu/fft/
