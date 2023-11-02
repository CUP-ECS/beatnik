# Overall Design of the Beatnik Z-Model Implmentation

## Overview



## ZModel Deriviate Calculation Strategy

Initial assumption - Surface-based information on Z and W is correct. Z and W in the spatial 
domain may be incorrect. The redistribution object stores the owner process of particle in the 
spatial mesh based on their last location. 

1. Calculate vorticity measure on the surface.

1. If using spatial velocity solve, update information in the spatial distribution of the particle cloud
   The initial location of particles in the particle cloud is based on their position during hte *previous*
   derivative calculationm, which may be incorrect. This step both provides the data needed for velocity
   calculation and updates their position and vorticity appropriately.
   - Send updated postition, vorticity, and vorticity measure to the spatial distribution based on 
     previous position.
   - Update owner process in redistributor class based on the new position
   - Redistrubute particles that have moved between owning processes
   - At this point, the surface and spatial views are consistent

1. Do a velocity solve

1. If using a spatial velocity solve, update information in the surface distribution
   - Collect zdot from spatial distributiona

1. Do a surface-based vorticity derivative solve, including surface normals, normal velocity,
   and finite differences 
 
1. Return zdot and wdot to the time integrator
