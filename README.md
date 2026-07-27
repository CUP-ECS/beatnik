# Beatnik - A Prototype High Performance Parallel Interface Benchmark

## Description

Beatnik is a benchmark for global communication based on Pandya and Shkoller's 3D fluid interace "Z-Model" in the Cabana mesh framework [1]. The goals of Beatnik are to:
  1. Provide an interesting and meaningful benchmark for numerical methods that require global communication, for example for far-field force calculations. This includes fast fourier transforms, distance sort cutoff-based methods, and (eventually) fast multi-pole methods.
  1. Understand the performance characteristics of different parallel decompositions of the Z-Model based on both a 2D decomposition based on logical mesh location location and a space-filling curve mesh decomposition.
  1. Provide a working prototype parallel implementation of the fluid interface model that other codes can use to create multi-scale models and codes.

> **Note on the current state of the repository.** The `rising-bubble-redesign`
> branch is rebuilding the solver from scratch: commit `89ec015` removed the
> previous `src/` solver headers, its only end-to-end test, and the run scripts
> that drove them. The build/run/test framework, the dependency set, and the
> documentation below are current; the solver internals they describe are being
> rewritten. Sections that document code that does not presently compile say so
> explicitly. See [Known Issues](#known-issues).

## Documentation and design reference

Detailed descriptions of the algorithms and design decisions used in this
repository — data structures, parallel decomposition, numerical choices, and why
each was picked — live in **[docs/design.md](docs/design.md)**.

This README covers what Beatnik is, how to build it, and how to run it. Anything
about *how and why the implementation works* belongs in `docs/design.md`, so the
two do not drift into duplicates. Machine-specific build and run instructions are
in `systems/<system>/claude.md`, and the build/run/test workflow itself is in
[CLAUDE.md](CLAUDE.md).

## Usage

Beatnik is a **header-only CMake INTERFACE library** (target `Beatnik::Beatnik`,
installed with a `BeatnikConfig.cmake`), plus example driver programs under
`examples/` that use it. Consuming it from another CMake project:

```cmake
find_package(Beatnik REQUIRED)
target_link_libraries(my_app PRIVATE Beatnik::Beatnik)
```

`Beatnik_Config.hpp` is generated at configure time and reports which optional
features the build has — most importantly `BEATNIK_ENABLE_CANOPY`. Diagnostics
are controlled by `BEATNIK_ENABLE_PROFILING` / `BEATNIK_PROFILING_LEVEL`; see
[src/Profiling.hpp](src/Profiling.hpp).

The public entry points and key types are in flux while the solver is rewritten.
This section will document them (the solver object, problem manager, mesh types,
and the minimal end-to-end call sequence) as they land; it deliberately does not
describe the removed pre-redesign API. The previous API surface is in git history
at `89ec015^:src/`.

The node-level parallelism / accelerator backend is selected at compile time from
the Kokkos backends the build enables (priority: CUDA/HIP/SYCL > OpenMP > Threads
> Serial). Which backends are available is a per-system fact — see
`systems/<system>/claude.md`.

### Minimal end-to-end example

The intended shape of a run: build with examples enabled, then launch an example
driver under MPI with an input deck.

```
# build (spack mode; see systems/<system>/claude.md for your machine)
spack env activate ~/spack_envs/tuolumne_beatnik
spack install

# run
mpirun -n 4 rocketrig examples/01_rising_bubble/multi_mode.in
```

On a scheduled machine, replace `mpirun -n 4` with that system's launcher — on
tuolumne, `flux run --ntasks=4 --nodes=1 ...`. Do not hand-roll the launch: use
the run command in `systems/<system>/claude.md` §4, or a batch script from
[scripts/tuolumne/test_template.flux](scripts/tuolumne/test_template.flux), which
sources the resolver so the required runtime environment is set for you.

### Examples and their accepted arguments

Examples are built only with `Beatnik_ENABLE_EXAMPLES=ON` (spack: `+examples`).
One example per ordered `examples/NN_name/` directory; shared helpers go in
[examples/common/](examples/common/).

#### `01_rising_bubble` — `rocketrig`

> **Does not currently build.** `rocketrig.cpp` and `InputFile.hpp` target the
> removed pre-redesign solver API. They are kept as the starting material for the
> rising-bubble driver, behind an explicit opt-in
> (`-DBeatnik_EXAMPLES_ALLOW_BROKEN=ON`). The argument documentation below
> describes what the retained parser accepts and is carried forward with it.

All run-time parameters — mesh size, initial condition, physics constants, solver
choice, I/O frequency, weak-scaling factor, and the FMM tunables — are read from a
single input file passed as the only positional argument.

```
rocketrig <input_file>          # run with the supplied input file
rocketrig --help                # print the full input-file schema
```

To run under MPI, prepend the usual launcher (`mpirun -n N`, `flux run`, `srun`,
…) — `rocketrig` itself still takes exactly one positional argument.

**Input file format.** Plain text, `key = value` per line. `#` starts a comment to
end-of-line. Blank lines are OK. Missing keys keep their built-in defaults, so a
near-empty file is valid — override only what you care about. Unknown keys,
malformed lines, and bad enum values all error with the file path, line number,
key, offending value, and (for enums) the full list of accepted values, e.g.
`rocketrig.in:21: invalid value for 'br_solver': 'magic' (expected one of: exact, cutoff, fmm)`.

Run `rocketrig --help` for the full schema. The key groups are:

| Group | Keys |
| --- | --- |
| Mesh / domain | `nodes`, `bounding_box`, `weak_scale` |
| Time integration | `timesteps`, `delta_t`, `write_frequency` |
| Initial condition | `initial_condition` (`cos`/`sech2`/`gaussian`/`random`), `magnitude`, `variation`, `period`, `tilt` |
| Physics / boundary | `boundary` (`periodic`/`free`), `gravity` (Gs), `atwood` |
| Solver | `solver_order` (`low`/`medium`/`high`), `br_solver` (`exact`/`cutoff`/`fmm`), `cutoff_distance`, `heffte_configuration`, `mu`, `epsilon` |
| FMM tunables (when `br_solver = fmm`) | `fmm_ncrit`, `fmm_max_depth`, `fmm_mac_theta`, `fmm_replication_depth`, `fmm_imbalance_tol`, `fmm_ncrit_tol`, `fmm_{x,y,z}{min,max}_tol`, `fmm_near_softening_factor` |

`br_solver = fmm` requires Beatnik to be built with Canopy support
(`Beatnik_ENABLE_CANOPY=ON`).

`fmm_near_softening_factor` (default `4.0`) guards the FMM far field against the
Plummer softening (`eps = sqrt(epsilon)`): pairs closer than
`fmm_near_softening_factor · eps` are evaluated by the softened near-field (P2P)
rather than the unsoftened multipole far-field (M2L). This prevents a blow-up at
full roll-up, where the rolled-up core shrinks below `eps` and the unsoftened far
field would otherwise produce a spurious, far-too-large velocity. Larger values
are more accurate but push more pairs into the costlier P2P path; `0` disables the
floor. See Canopy's `near_softening_factor` for details.

The shipped deck [examples/01_rising_bubble/multi_mode.in](examples/01_rising_bubble/multi_mode.in)
is a periodic multi-mode rocket rig: a cosine-distributed initial interface,
periodic boundaries, low-order Z-model with the exact BR solver. To explore
variations, copy it and edit the keys you care about — e.g. `nodes = 512` for a
larger mesh, or `weak_scale = 16` with `write_frequency = 0` to scale up 16× and
skip I/O.

A second shipped case, a non-periodic single-mode `sech2` Gaussian rollup
recreating sections 2.3–2.4 of Pandya and Shkoller (high-order Z-model, free
boundaries, `atwood 0.15`, `mu = 2`, `epsilon = 2`, `magnitude 0.1`,
`period 9.0`, `nodes 64`), was removed along with its pre-scaled 1024/4000/16000
variants in `89ec015`; the decks are recoverable from
`89ec015^:examples/01_rocketrig/`. It is compute-intensive and works best on a GPU
accelerator. Its scaling rules are documented under
[Scaling to large meshes](#design-limitation-scaling-to-large-meshes) below.

## Dependencies and Build Notes

Beatnik depends on the following packages in all configurations:

  1. ECP CoPA's Cabana/Grid particle and mesh framework [2], version 0.7.0 or newer, built **with MPI and Grid support** (both are hard-enforced at configure time)
  1. Kokkos 4.0 or newer
  1. A HeFFTe version compatible with Cabana [3] (tested with 2.4.0)
  1. LLNL Silo 4.11.1 or newer, configured with MPI support
  1. A high-performance **GPU-aware** MPI implementation such as OpenMPI, MPICH, or MVAPICH
  1. GTest 1.10+ when `Beatnik_ENABLE_TESTING=ON`
  1. Optionally **Canopy**, which provides the fast multipole far-field solver. `Beatnik_ENABLE_CANOPY` follows from whether it is found.

**Build with spack.** The beatnik spack package enforces these requirements, so
use it for both installation (`spack install` / `spack env create`) and
development (`spack develop` in an environment) — that also lets you develop
Beatnik's *dependencies* the same way (`spack develop cabana && spack concretize -f`).
See the [Spack developer-workflow documentation](https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html).
On systems with a site spack install, upstreaming to it is advisable.

Per-system environments, build commands and their gotchas are **not** in this
file. Run `hostname` and read the matching `systems/<system>/claude.md`; the
committed spack environment snapshots live beside it. The build/run profile
mechanism (build modes, the resolver, and the gate) is documented in
[CLAUDE.md](CLAUDE.md). The pre-redesign `configs/` directory that used to hold
per-system spack files for UNM Hopper and LLNL Tioga was removed in `89ec015`;
those files are at `89ec015^:configs/` if either machine is revived.

Non-obvious build constraints:

- **Header-only rebuilds can be missed.** `Beatnik` is a CMake INTERFACE library
  that does not track `HEADERS_PUBLIC` as dependencies, so a header-only change
  can produce a sub-second no-op `spack install`. Workaround: `touch` a consumer
  `.cpp` first. See [Future Optimizations](#future-optimizations).
- **`-march`** defaults to `native` in Release builds; override with
  `Beatnik_BUILD_MARCH` on a platform where that fails.
- **`Beatnik_PROFILING_LEVEL` is gated by `Beatnik_ENABLE_PROFILING`.** With the
  kill switch OFF, any level you set resolves to 0.
- **ASan and MSan are mutually exclusive** (`WITH_ASAN` / `WITH_MSAN`); enabling
  both is a hard configure error.

### Design limitation: mesh representation trade-off

Beatnik uses a simple mesh-based representation of the surface manifold as a Cabana grid 2D mesh in I/J space and a regular block 2D decomposition of this manifold. The physical position of each element in the mesh is stored as a separate vector in the nodes of the mesh. This design results in simple and efficient computation and communication strategies for surface normals, artificial viscosity, and Fourier transforms elements. However, it complicates methods where the data decomposition and communication is based on the spatial location of manifold points, requiring them to either maintain a separate spatial decomposition of the surface or to continually construct a spatial decomposition. A surface mesh that decomposed the mesh by spatial location would be an interesting alternative but would have the opposite issue - communication for surface calculations would be more complex but the (expensive) far force methods that rely on spatial decompositions (e.g. distance sort and spatial tree methods like the fast multi-pole method) would be less expensive.

This is a deliberate, long-standing design choice rather than a defect. See
[docs/design.md](docs/design.md) for the full discussion.

### Design limitation: scaling to large meshes

Running the single-mode rollup at large `nodes` is not just a matter of raising the mesh resolution. The intended way to grow the problem is to scale `bounding_box` **linearly** with `nodes` (e.g. `bounding_box = nodes / 64`, taking the 64-node `bounding_box = 1.0` case as the baseline). This keeps the physical grid spacing `dx = 2·bounding_box/(nodes-1)` roughly constant, which is what stops interface points from getting pathologically close together — the regime where the BR/FMM velocity evaluation blows up.

When you scale the domain this way, two things must change with it, and a third becomes an unavoidable cost:

1. **Initial-condition geometry must be re-scaled.** For the `sech2` IC the height is `magnitude · sech²(period · r²)` with `r` the physical distance from the domain center, so:
   - `magnitude` (the vertical amplitude) scales **linearly** with `bounding_box`.
   - `period` scales as the **inverse square** of `bounding_box`. `period` sets an inverse width (larger `period` ⇒ *narrower* bump), so leaving it unscaled — or worse, scaling it up — collapses the bump to a sub-cell spike and you get a flat, unevolving surface. (This inverse-square rule is specific to `sech2`; the `cos` IC's `period` is a wavelength and scales *linearly*.)

2. **`delta_t` must be kept small — do not inflate it to "evolve faster".** A geometrically larger domain has *higher* characteristic interface velocities (`v ~ sqrt(A·g·λ)` with the feature wavelength `λ ∝ bounding_box`, so `v ∝ √bounding_box`). With `dx` held fixed, the CFL-stable timestep therefore *shrinks* as `~1/√bounding_box`, roughly `delta_t_auto / √(bounding_box)` at peak velocity (where `delta_t_auto = tau/50` for the high-order solver and `tau = 1/sqrt(atwood·gravity)`). Because the initial vorticity is zero, the velocity starts near zero and grows as the instability develops — so an oversized `delta_t` will appear stable for the first several steps and then suddenly blow up *while the surface is still nearly flat*. That sudden blowup is numerical (CFL), not a physical rollup instability; the cure is a smaller `delta_t`, not more smoothing.

3. **The step count to reach a rollup grows ~linearly with `bounding_box`.** Reaching a rollup needs physical time `∝ √bounding_box` (the feature timescale), integrated at a stable step `∝ 1/√bounding_box`, so the number of timesteps for a comparable rollup scales as roughly `bounding_box` relative to the small-domain case. A large-mesh run with the same modest step count as a small case will only show early, small-amplitude motion — not a developed rollup. Note that `gravity` and `atwood` do **not** provide a shortcut: with the auto `delta_t` they cancel out of the achieved evolution.

Scaling table from the 64-node baseline (`bounding_box = 1.0`, `magnitude = 0.1`, `period = 9.0`, `atwood = 0.15`), with `B = bounding_box`:

| nodes | B = bbox | magnitude | period      |
| ----- | -------- | --------- | ----------- |
| 64    | 1        | 0.1       | 9.0         |
| 256   | 4        | 0.4       | 0.5625      |
| 512   | 8        | 0.8       | 0.140625    |
| 1024  | 16       | 1.6       | 0.03515625  |
| 4000  | 62.5     | 6.25      | 0.0023040   |
| 16000 | 250      | 25.0      | 0.000144    |

In short: scale `bounding_box` with `nodes`, re-scale `magnitude` (∝ B) and `period` (∝ 1/B²) to match, keep `delta_t` at or below `tau/(50·√B)`, and budget timesteps that grow with `B`. When in doubt, verify the physics on a moderate mesh (e.g. the 1024 case) where a full rollup is cheap, then trust the scaling for the production meshes.

### Design limitation: FMM tuning is coupled to run scale

The FMM (Canopy) far-field path has tunables whose good values depend on the run,
not just on correctness. The configuration validated on a full 256² roll-up was
`fmm_mac_theta = 0.4`, `fmm_max_depth = 19`, `fmm_imbalance_tol = 0.20`,
`epsilon = 2`, with `fmm_near_softening_factor` at its default `4.0` or higher.
Two constraints worth knowing before tuning:

- **`fmm_max_depth = 19` is a hard maximum** — Canopy's `TreeBuilder` throws above
  19, because of Morton-key storage.
- **`fmm_ncrit = 64` beat `128` at scale on GPU.** On a 1536² (B=24) full-rollup at
  256 ranks / 64 nodes, calibration measured 3.80 s/step at `ncrit=64` versus
  4.11 s/step at `ncrit=128` over the first 50 steps; the larger-leaf GPU
  throughput win did not materialize.
- **FMM cost grows through roll-up.** Canopy `solve()` dominates per-call cost, and
  the softening floor widens the near-field P2P as the core densifies, so cost
  climbs as the run develops. `fmm_near_softening_factor` trades far-field
  accuracy against that cost.

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

A longer list of self-contained project ideas, sized from student projects to
thesis work, was removed in `89ec015` and is preserved at `89ec015^:PROJECTS.md`.

## Future Optimizations

Opportunities noticed but deliberately not taken. These are *improvements*, not
defects — defects belong in [Known Issues](#known-issues). Ask before adding to
this list.

- **Track header dependencies in the `Beatnik` INTERFACE target.**
  `add_library(Beatnik INTERFACE)` in [src/CMakeLists.txt](src/CMakeLists.txt)
  means consumer object files do not depend on `HEADERS_PUBLIC`. When only a
  header changes, `make` sees nothing to rebuild and `spack install` reports a
  sub-second build; the workaround is to `touch` a consumer `.cpp` first. Fix:
  use `target_sources(Beatnik INTERFACE FILE_SET HEADERS BASE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR} FILES ${HEADERS_PUBLIC})`, or attach a
  `PUBLIC_HEADER` property, so consumers track them as real dependencies.
- **Higher profiling levels (2, 3).** Both are reserved but unused. Natural
  additions once the new solver has phases worth timing: per-phase timers at
  level 2, per-rank particle counts at level 3. Mirror Canopy's existing
  `[Canopy Diagnostics] solve() phase breakdown` format for consistency.

## Known Issues

Deferred *defects*: what fails, how it reproduces, and whether it is a regression
from current work or pre-existing. Distinct from
[Future Optimizations](#future-optimizations) and from the design-limitation
subsections above, which are intended behavior.

- **The `regression` test tier is empty, so the ship gate is vacuous.**
  *Pre-existing as of the redesign, not a regression from a code change.*
  `89ec015` removed the pre-redesign solver and `tests/tstFmmVsExact.hpp`, its
  only end-to-end test. `tests/CMakeLists.txt` registers zero regression-tier
  tests, so `ctest -L regression -R SERIAL` and
  `scripts/tuolumne/run_regression_minset.flux` both pass trivially. **A green
  gate currently proves nothing.** Resolves when the new solver lands its first
  end-to-end test. See [CLAUDE.md](CLAUDE.md#minimum-test-set).

- **`examples/01_rising_bubble` does not build.** *Pre-existing as of the
  redesign.* `rocketrig.cpp` includes `Solver.hpp` and `BoundaryCondition.hpp`,
  removed in `89ec015`; `InputFile.hpp` targets the old parameter struct.
  Reproduces with `-DBeatnik_ENABLE_EXAMPLES=ON
  -DBeatnik_EXAMPLES_ALLOW_BROKEN=ON` (a compile error on the missing headers).
  Without that opt-in the example is skipped with a status message, so an
  examples-enabled build still succeeds. Resolves when the driver is ported to
  the new solver API — and its accepted arguments must be re-synced into this
  README at the same time.

- **FMM large-scale crash: GTL dreg-cache exhaustion in the Canopy `Rebalance`
  particle migrate (MITIGATED 2026-06-22; durable fix tracked in Canopy #22).**
  *Pre-existing.* A 256-rank / 64-node 1536² full-rollup run deadlocked at step
  307 of 8400 during the first significant `Rebalance`, with
  `(GTL DEBUG) dreg_evict returned NO_SPACE ... more than 10000 active memory
  regions`. One rank's GPU comm died and the other 255 blocked in the next
  collective, so the job hung (flux still reported it running for ~19 h until
  cancelled). **Root cause:** `dreg_evict NO_SPACE` is a peak-*simultaneity*
  failure — all 10000 GTL registration-cache entries active at once — not a leak.
  Canopy's `RegisteredBufferPool` bounds the `solve()` exchanges (coalesced M2L +
  P2P) to one registered region per direction, but the `Rebalance` particle
  migration (`TreePartitioner::migrate_particles` → `Cabana::Distributor` /
  `Cabana::migrate`) is **not** pooled; at 256 ranks the many-way scatter hands
  O(peers) device buffers to GPU-aware MPI at once, blowing the default.
  **Stopgap (deployed):** `GTL_DREG_CACHE_SIZE=262144` (≈26× headroom), now set
  in [scripts/tuolumne/runtime_env.sh](scripts/tuolumne/runtime_env.sh) so every
  run on the system picks it up. This is out-provisioning only — the peak likely
  grows as the core densifies, so it may not survive to full rollup. **Durable
  fix:** route the rebalance migrate through a pool-backed coalesced exchange,
  one registered region per direction, mirroring the `solve()` fix — **Canopy
  #22**. Once that lands and Beatnik picks up the new Canopy, the env bump becomes
  belt-and-suspenders.

Two further FMM defects — a Slingshot NIC registration exhaustion during Canopy
`Rebalance` (RESOLVED 2026-06-17) and a premature FMM NaN at full rollup
(RESOLVED 2026-06-18, root cause: the FMM far field used the unsoftened `1/r`
Laplace kernel while softening was applied only in the near field) — were closed
before the redesign. Their full investigation records are at
`89ec015^:tasks/fmm_fullrollup_crash.md` and
`89ec015^:tasks/fmm_premature_nan.md`.

## Acknowledgment, Contributors, and Copyright Information

Beatnik is primarily available as open source under a 3-Clause BSD License. It is being developed at the University of New Mexico, Tennessee Tech University, and the University of Alabama under funding the U.S. Department of Energy's Predictive Science Academic Alliance Partnership III (PSAAP-III) program. Contributors to Beatnik development include:

  * Patrick G. Bridges (patrickb@unm.edu)
  * Thomas Hines (tmhines3@ua.edu)
  * Jered Dominguez-Trujillo (jereddt@unm.edu)
  * Jacob McCullough (jmccullough12@unm.edu)
  * Jason Stewart (jastewart@unm.edu)

The general structure of Beatnik and the rocketrig examples were taken from the ExaMPM proxy application (https://github.com/ECP-copa/ExaMPM) developed by the ECP Center for Particle Applications (CoPA), which was also available under a 3-Clause BSD License when used for creating application structure. The CMake organization, macros, and test harness setup are copied from the Cabana library.

## References

1. Gavin Pandya and Steve Shkoller. "3d Interface Models for Raleigh-Taylor Instability." Published as arxiv.org preprint https://arxiv.org/abs/2201.04538, 2022.

1. https://github.com/ECP-copa/Cabana/

1. Innovative Computing Laboratory. "heFFTe." URL: https://icl.utk.edu/fft/

1. Spack developer workflows: https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html
