* Beatnik - An early High Performance Parallel Interface prototype

Beatnik is an initial prototype of Pandya and Shkoller's 3D fluid interace
"z-model" in the Cabana/Cajita mesh framework. The goals of Beatnik are to:
  1. Provide an interesting and meaningful benchmark for numerical methods 
     that require global communication, particularly fast fourier transforms 
     and fast multipole methods.
  1. Understand the performance characteristics of a parallel decomposition of
     the Z-Model based on the logical mesh location as a potential precursor 
     to later High performance parallel interface explorations. Also note that
     Beatniks are, in some sense, precursors to Hippies.

As mentioned above, this implementation of the Z-Model uses a simple 
mesh-based representation of the surface manifold with a regular decomposition 
as a Cajita 2D mesh in I/J space and the physical position of each element in 
the mesh stored as a separate vector in the nodes of the mesh. This is efficient 
for the low-order z-model, as both surface normals, artificial viscosity, and 
Fourier transforms for estimating interface velocities are straightforward in 
this representation. For higher-order models that require more accurate calculation
of far-field forces, this representation may dramatically increase communication 
requirements. A different design and partitioning strategy for the problem, not 
provided by Beatnik, would better optimize the high-order model.
 
Beatnik is being implemented in multiple distinct steps:

  1. A low-order model implementation that relies on Cajita/HeFFTe Fourier transforms for estimating velocity interface at mesh points.

  2. A high-order model implementation that relies on the external PVFMM fast multipole library for accurately calculating far-field forces on mesh points at the cost of significant communication between the native logically-partitionted mesh and the multipole solver's internal physical partitioning of points. Note that this method requires a fairly small timestep to maintain stability.

  3. A medium-order model that uses the Fourier transform for estimating interface velocity and the fast multipole method for estimating how the vorticity changes at each interface point. While this implementation requires more computational methods and steps at each time step than the high-order model, it is also does net require as small of timesteps as the high-order model.

  4. (Potentially) A cutoff-based approach for calculating far-field forces using the Cabana particle framework that accelerates far-field force calculations by avoiding the complex hierarchical communications and calculations in the fast multipole solver.
