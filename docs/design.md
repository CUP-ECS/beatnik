# Beatnik design notes

Detailed descriptions of the algorithms and design decisions used in this
repository: data structures, parallel decomposition, numerical choices, and
**why** each was picked.

`README.md` covers what Beatnik is, how to build it, and how to run it, and
links here rather than duplicating this material. The split is: *how to use it*
stays in the README, *how and why it works* lives here. Machine-specific build
and run facts belong in `systems/<system>/claude.md` and in neither of these.

> **STATUS: SCAFFOLD.** Commit `89ec015` removed the previous solver
> (`src/{Solver,ZModel,ProblemManager,SurfaceMesh,SpatialMesh,TimeIntegrator,BoundaryCondition,Operators,HaloComm,SiloWriter}.hpp`
> and the three BR solvers) to build a new one from scratch on the
> `rising-bubble-redesign` branch. The sections below are the headings that fit
> this project; most of them describe code that does not exist yet, so they are
> deliberately left empty rather than filled with prose the code does not
> support. **Fill each section as the corresponding component lands.**
>
> The pre-redesign design write-up — a step-by-step account of the Z-Model
> derivative-calculation strategy, including the surface/spatial consistency
> ordering — is preserved in git history at `89ec015^:DESIGN.md`. Read it for
> background, but do not assume it describes current code. The pre-redesign
> project idea list is likewise at `89ec015^:PROJECTS.md`.

## Model and governing equations

*To fill in:* the interface model being solved, the state carried per interface
point, and the form of the evolution equations actually implemented.

Background: Beatnik implements Pandya and Shkoller's 3D fluid-interface
"Z-Model" (see README reference 1) on top of the Cabana mesh framework.

## Mesh representation

*To fill in:* how the surface manifold is represented, and what is stored where.

Durable design decision carried across the redesign, and the central trade-off
in the benchmark: the surface manifold is a Cabana 2D grid in logical I/J space
with a regular 2D block decomposition, and each node's **physical position is
stored as a separate vector on the mesh node** rather than being implied by its
index. That makes surface-local work (normals, artificial viscosity, Fourier
transforms) simple and efficient, because neighbors in logical space are
neighbors in memory and in the decomposition. It makes anything keyed on
*spatial* proximity harder, because spatially near points can live on any rank —
so spatially-decomposed far-field methods must either maintain a second spatial
decomposition or rebuild one continually. A spatially-decomposed surface mesh
would invert both properties; that alternative is deliberately not taken.

## Parallel decomposition

*To fill in:* rank topology, halo/ghost regions and their depth, which
operations are surface-local vs global, and where collectives appear.

Note for the redesign: any second (spatial) decomposition and the migration
between it and the surface decomposition needs its consistency ordering written
down explicitly — which view is authoritative at each step. Getting that
ordering wrong was a recurring source of subtle bugs in the previous
implementation, and `89ec015^:DESIGN.md` documents the ordering that was used.

## Far-field / velocity solve

*To fill in:* the velocity-evaluation strategies provided, their cost and
accuracy characteristics, and how one is selected.

Context: computing interface velocity is the expensive, communication-bound part
and the reason Beatnik exists as a benchmark for global communication. The
previous implementation offered exact all-pairs, distance-cutoff, and
fast-multipole (via the external Canopy library) evaluators behind a common
interface, selected at runtime. Whether the redesign keeps that structure is an
open decision — record it here once made.

## Time integration

*To fill in:* the integrator, its order, and how the timestep is chosen or
constrained.

Any CFL-style stability constraint and its scaling with mesh resolution and
domain size belongs here, with the derivation. Practical consequences for
choosing run parameters belong in the README.

## Numerical choices

*To fill in:* desingularization / softening, artificial viscosity, finite
difference and quadrature stencils, and the tolerances that go with them. State
the default for each and what it trades off — these are the knobs that decide
whether a run is stable, and an undocumented default is a trap.

## I/O

*To fill in:* output format, what is written, write frequency, and the
parallel-write strategy.

Silo is a required dependency (`find_package(SILO REQUIRED)` in
`CMakeLists.txt`), and the previous implementation wrote Silo files.

## Diagnostics and profiling

Implemented and current. `Beatnik_ENABLE_PROFILING` plus
`Beatnik_PROFILING_LEVEL` (0=off, 1=basic, 2=detailed, 3=verbose; levels 2 and 3
are reserved) resolve to the `BEATNIK_ENABLE_PROFILING` and
`BEATNIK_PROFILING_LEVEL` compile definitions, applied both at the top level and
on the `Beatnik` INTERFACE target so consumers see them too. The semantics
mirror Canopy's `Canopy_ENABLE_PROFILING` / `Canopy_PROFILING_LEVEL`
deliberately, so the two libraries can be reasoned about together. See
[src/Profiling.hpp](../src/Profiling.hpp) and the resolution logic in
[CMakeLists.txt](../CMakeLists.txt).

## Optional dependencies

`Canopy` is an optional dependency providing a fast multipole solver for
far-field forces. It is discovered with the `Beatnik_add_dependency` macro,
which sets `Beatnik_ENABLE_CANOPY`; `src/CMakeLists.txt` aliases that to the
uppercase `BEATNIK_ENABLE_CANOPY` so `#cmakedefine` in `Beatnik_Config.hpp.in`
resolves (cmakedefine's variable lookup is case-sensitive — without the alias
the C macro stays undefined even in a `+canopy` build).
