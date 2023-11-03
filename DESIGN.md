# Overall Design of the Beatnik Z-Model Implmentation

## Overview



## ZModel Deriviate Calculation Strategy

Initial assumption - Surface-based information on Z and W is correct. Z and W in the spatial 
domain may be incorrect. The redistribution object stores the owner process of particle in the 
spatial mesh based on their last location. 

1. Calculate vorticity measure on the surface.

1. If using spatial velocity solve, update information in the spatial distribution of the particle cloud
   The initial location of particles in the particle cloud is based on their position during the *previous*
   derivative calculation, which may be incorrect. This step both provides the data needed for velocity
   calculation and updates their position and vorticity appropriately.
   1. Send updated postition, vorticity, and vorticity measure to the spatial distribution based on 
     previous position.
   1. Update owner process in redistributor class based on the new position
   1. Redistrubute particles that have moved between owning processes
   1. At this point, the surface and spatial views are consistent

1. Do a velocity solve, which may require a spatial solve or a surface solve depending on the
   solver order and method used.

1. If using a spatial velocity solve, update information in the surface distribution, which 
   requires collecting zdot from spatial distribution back to the original surface distribution

1. Do a surface-based vorticity derivative solve, including surface normals, magnitude of the
   velocity vector, and finite differences.
   1. Compute V on each surface point (which requires the magnitude of the velocity)
   1. Halo V so we can compute finite differences and laplacians on it
   1. Calculate wdot at each surface point
 
1. Return zdot and wdot to the time integrator
