# Beatnik - A Prototype High Performance Parallel Interface Benchmark

## Description

Beatnik is a benchmark for global communication based on Pandya and Shkoller's 3D fluid interace "Z-Model" in the Cabana framework [1]. The goals of Beatnik are to:
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

By default, Beatnik solves a simple multi-mode rocket rig problem sized for a single serial CPU core with approximately 4GB of memory. It also includes command line options to change initial problem state, I/O frequency, and to weak-scale scale up the initial problem to larger number of processes. It also includes problem-specific command line parameters; setting these parameters accurately generally requires expertise in fluid interface models. However, we provide several useful examples drawn from the ZModel papers that recreate the results in those papers.

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
Another test case is a single-mode rollup test where the intitial interface is set according to a hyperbolic secant function. This testcase recreates the the Gaussian perturbation results in Panda and Shkoller's paper from sections 2.3 and 2.4.  To run this testcase with a high-order model, use the following command line parameters. Note that this works best with a GPU accelerator, as the exact high-order far field force solver is very compute intensive and is generally impractical for non-trivial mesh sizes without GPU acceleration:
`bin/rocketrig -x cuda -O high -n 64 -I sech2 -m 0.1 -p 9.0 -b free -a 0.15 -M 2 -e 2`

## Planned Development Steps

Beatnik is being implemented in multiple distinct steps, with associated planned releases:

  * Version 1.0 Features

    1. A low-order model implementation that relies on Cabana Grid/HeFFTe Fourier transforms for estimating velocity interface at mesh points.
    1. A high-order model implementation based on brute-force exact computation of long-range forces
    1. A medium-order model that uses the Fourier transform for estimating interface velocity and the far-field force solver for estimating how the vorticity changes at each interface point. 
    1. Support for periodic boundary conditions and free boundary conditions
    1. Simple benchmark examples including a single-mode Gaussian roll-up test and the multi-mode rocket rig experiment.
    1. Direct support for weak scaling of benchmarks through command line arguments

  * Version 1.X Planned Features

    1. Rearchitecting of the z-model solve into explicitly-coupled surface mesh and spatial mesh solvers
    1. A spatial mesh cutoff-based approach for calculating far-field forces using the Cabana particle framework. The goal of this work is to understand the accuracy/performance tradeoffs in the Z-Model, particularly in the medium-order
    1. Improved timestep, desingularization, and artificial viscosity parameter handling. The goal of this is to provide good defaults when other input parameters are changed.
    1. Additional interface initialization options, including Gaussian random and file-based interface initialization (also useful for checkpointing)
    1. Support for coupling with other applications through either I/O (e.g. ADIOS) or Communication (e.g. Portage) 
    1. Additional test case definitions

  * Potential later (e.g. >=2.0) features

    1. Direct fast multi-pole or P3M solver for scalable, high precision high-order model solves.
    1. Support for multiple interface manifolds in a single simulation.

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
