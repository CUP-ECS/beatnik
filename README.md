# Beatnik - A  prototype High Performance Parallel Interface Benchmark

## Description

Beatnik is a benchmark for global communication based on Pandya and Shkoller's 
3D fluid interace "Z-Model" in the Cabana/Cajita mesh framework. The goals of 
Beatnik are to:
  1. Provide an interesting and meaningful benchmark for numerical methods 
     that require global communication, particularly fast fourier transforms 
     and fast multipole methods.
  1. Understand the performance characteristics of different parallel decompositions
     of the Z-Model based on both a 2D decomposition based on logical mesh location 
     location and a space-filling curve mesh decomposition.

The initial Beatnik implementation of the Z-Model uses a simple mesh-based 
representation of the surface manifold with a regular decomposition 
as a Cajita 2D mesh in I/J space and the physical position of each element in 
the mesh stored as a separate vector in the nodes of the mesh. This is efficient 
for the low-order z-model, as the computation of surface normals, artificial viscosity, 
and Fourier transforms for estimating interface velocities are straightforward in 
this representation. Higher-order models, howqever, require more accurate calculation
of far-field forces and the initial representation may dramatically increase communication 
requirements. 

In the future, we plan to use Beatnik to directly implement a fast multipole method in
Beatnik built on top of a mesh decomposition based on space-filling curves. This will allow
the mesh to be decomposed based on spatial location but still be indexed logically for 
surface-local calculations, reducing fast multipole communicaiton overheads.
 
## Building Beatnik

TBA

## Running Beatnik

The simplest test case is a simple rocket-rig experiment with an initial interface 
distributed according to a cosine function:
  1. Serial execution: `bin/rocketrig -x serial`
  1. Cuda execution (on systems with GPUs) with a 512x512 mesh: `bin/rocketrig -x cuda -n 512`

Another test case is a single-mode rollup test where the intitial interface is 
set according to a hyperbolic secant function. This testcase recreates the 
XXX experiment and the results in Raag and Shkoller's paper from section XXX.  To run this
testcase with a high-order model, use the following command line parameters. Note that we 
assume a GPU accelerator, as the exact high-order far field force solver is very compute intensive:

`bin/rocketrig -x cuda -O high -n 64 -I sech2 -m 0.1 -p 9.0 -F 1 -a 0.15 -M 2 -e 2`

## Planned Development Steps

Beatnik is being implemented in multiple distinct steps, with associated planned releases:

  * Version 1.0 Features

    1. A low-order model implementation that relies on Cajita/HeFFTe Fourier transforms for estimating velocity interface at mesh points.

    2. A high-order model implementation based on either exact or PVFMM for computing long-range forces

    3. A medium-order model that uses the Fourier transform for estimating interface velocity and the fast multipole method for estimating how the vorticity changes at each interface point. 

    4. Support for periodic boundary conditions and free boundary conditions

    5. Multiple Benchmark examples, including a single-mode gaussian roll-up test, the rocket rig experiment, and a rising bubble test

    6. Direct support for weak scaling of benchmarks through command line arguments

  * Version 1.1 Expected Features

    1. A cutoff-based approach for calculating far-field forces using the Cabana particle framework that accelerates far-field force calculations by avoiding the complex hierarchical communications and calculations in the fast multipole solver.

    2. Improved timestep handling to make sure that the timestep is always appropriate for the parameters given

    3. Support for coupling with other applications through either I/O (e.g. ADIOS) or Communication (e.g. Portage) approaches

  * Version 2.0 Expected Features
  
    1. Direct fast multipole implementation and a mesh decomposition based on space-filling curves to optimize high-order model 
       performance
