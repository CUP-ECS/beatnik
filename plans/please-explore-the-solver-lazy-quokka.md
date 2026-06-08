# Compile-time Kokkos backend selection in Solver

## Context

`Beatnik::createSolver` in [src/Solver.hpp:262-336](src/Solver.hpp#L262-L336) currently selects the Kokkos execution/memory space at runtime from a `-x`/`--driver` string argument (`"serial"`, `"openmp"`, `"threads"`, `"cuda"`, `"hip"`). This forces every binary to compile a `Solver<ExecSpace, MemSpace, ModelOrder>` instantiation for every backend enabled at build time, even though only one of those backends is ever executed by a given binary launch. It also creates a runtime failure mode (unknown driver string, or asking for a backend that wasn't compiled in).

Kokkos already chooses an appropriate default execution space at compile time based on which backends are enabled (priority order: CUDA/HIP/SYCL > OpenMP > Threads > Serial) and exposes it as `Kokkos::DefaultExecutionSpace`, with the matching memory space as `Kokkos::DefaultExecutionSpace::memory_space`. This is the standard Kokkos idiom for "use whatever backend was compiled in." Switching to it removes the runtime dispatch, shrinks compile times, and removes the `-x` argument from `rocketrig`.

This change is internal to Beatnik — `CreateBRSolver.hpp` is already templated on `ExecutionSpace`/`MemorySpace`, so no other solver code is affected.

## Approach

Use `Kokkos::DefaultExecutionSpace` and its associated `memory_space` everywhere `createSolver` currently dispatches on the driver string. `Kokkos::DefaultExecutionSpace::name()` provides a printable backend name for the runtime banner.

### Kokkos backend-priority semantics (the part the design hinges on)

- `Kokkos::DefaultExecutionSpace` priority: `Cuda` > `HIP` > `SYCL` > `OpenMPTarget` > `OpenMP` > `Threads` > `HPX` > `Serial`. So **a CUDA + OpenMP build will run the solver on the GPU** with `Kokkos::CudaSpace` memory — matching what `-x cuda` does today. Users lose only the ability to *force* OpenMP execution on a GPU-enabled build, which is exactly the runtime-vs-compile-time flexibility we are intentionally trading away.
- `Kokkos::DefaultHostExecutionSpace` priority (host-only): `OpenMP` > `Threads` > `HPX` > `Serial`. **If `Kokkos_ENABLE_OPENMP=ON`, this resolves to `Kokkos::OpenMP`, not `Kokkos::Serial`.** It only falls back to `Serial` when no host-parallel backend is enabled.
- The current `Solver` template takes a single `ExecutionSpace`/`MemorySpace` pair and does not distinguish device from host parallel. A `grep` of `src/` for `HostSpace` / `DefaultHostExec` confirms that host-side code today (`SiloWriter`, `SpatialMesh::_boundary_topology`, `CutoffBRSolver`'s boundary-info handling, `HaloComm`'s local-mesh creation) uses `Kokkos::HostSpace` only as a *memory* space for mirror views — none of it currently does host-parallel work. So the substitution introduces no regression for host execution. If a future change wants host-parallel regions alongside device kernels, those regions should explicitly name `Kokkos::DefaultHostExecutionSpace` and will pick up `Kokkos::OpenMP` automatically when enabled.

### Changes to [src/Solver.hpp](src/Solver.hpp)

- Delete the runtime-dispatching body of `createSolver` (lines 276-335: the `if/else` ladder over `"serial"`, `"threads"`, `"openmp"`, `"cuda"`, `"hip"`).
- Remove the `const std::string& device` parameter from `createSolver`'s signature (line 264).
- Replace the body with a single instantiation:
  ```cpp
  using ExecSpace = Kokkos::DefaultExecutionSpace;
  using MemSpace  = Kokkos::DefaultExecutionSpace::memory_space;
  return std::make_shared<Beatnik::Solver<ExecSpace, MemSpace, ModelOrder>>(
      comm, global_num_cell, partitioner, atwood, g,
      create_functor, bc, mu, epsilon, delta_t, params);
  ```
- The `Solver` class template itself (lines 103-258) is unchanged — it remains templated on `ExecutionSpace, MemorySpace, ModelOrder` so direct instantiation (e.g. by tests in the future) still works.

### Changes to [examples/01_rocketrig/rocketrig.cpp](examples/01_rocketrig/rocketrig.cpp)

Apply this pattern in each spot:

- `shortargs` (line 56): drop `x:`.
- `longargs` (line 64): drop the `"driver"` entry.
- `ClArgs::driver` field (line 119): delete.
- `parseInput` default on line 216 (`cl.driver = "serial"`): delete.
- `case 'x':` block (lines 335-352): delete.
- `help()` `-x` line (lines 155-157): delete.
- The three `Beatnik::createSolver(...)` calls at lines 732, 738, 744: drop the leading `cl.driver,` argument.
- The "Thread Setting" banner print (lines 788-790): replace with a print of `Kokkos::DefaultExecutionSpace::name()` so the user can see which backend is actually compiled in.

### Build system

No CMake changes required. Beatnik's existing `Beatnik_REQUIRE_<DEVICE>` options in [CMakeLists.txt](CMakeLists.txt) continue to govern which Kokkos backends are enabled; Kokkos's own logic picks the default from among them.

### Documentation / configs

- Sweep [configs/](configs/) and any shell scripts there for `-x`/`--driver` usage and remove those flags (read-only step during plan — to be applied during implementation).
- Update any README/help reference to the `-x` flag.

## Verification

1. Configure and build with a single backend enabled (e.g. `-DBeatnik_REQUIRE_SERIAL=ON`); confirm `rocketrig` builds and runs a short problem (e.g. `rocketrig -n 32 -t 10 -F 0`) and that the banner prints `Serial` as the backend.
2. Build with CUDA enabled and rerun the same command; banner should report `Cuda` and the run should complete on the GPU.
3. Confirm `rocketrig -h` no longer lists `-x` and that passing `-x serial` now fails with the standard "invalid argument" error path.
4. Grep the tree for residual references to `cl.driver`, `device.compare`, or the driver strings to ensure nothing is left behind.
