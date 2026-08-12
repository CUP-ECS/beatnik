/****************************************************************************
 * Copyright (c) 2025 by the Beatnik authors                                *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Beatnik library. Beatnik is distributed under a *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
/**
 * @file Beatnik_Solver.hpp
 * @brief The top-level driver: owns every component and runs the time loop.
 *
 * Port of run_adaptive_mesh_bubble.py::main (lines 1195-1652), with the
 * plotting removed
 *
 * THE CONTROL FLOW
 * ----------------
 * ```
 * setup:
 *   restart ? RestartReader::load : InitialCondition::build
 *   resolve proximity radii from initial_min_edge          [1272-1286]
 *   seed material position                                 [1227/1208]
 *   write the startup checkpoint                           [1313-1324]
 *
 * loop, while local_step < steps:
 *   break if t >= t_end                                    [1402-1403]
 *   dt = chooseStepSize, then clamp by dt_switch and t_end  [1406-1410]
 *   TimeIntegrator::step                                   [1411]
 *   t += dt
 *   ABORT if not all finite  -> "nonfinite mesh-RHS state" [1413-1423]
 *
 *   if (!dynamic_remesh && refine_every && step % refine_every == 0):
 *     AdaptiveMesh::refine                                 [1424-1437]
 *     if anything was marked:                              [1438]
 *       MeshQuality::improveConnectivityByFlips            [1440-1445]
 *       MeshQuality::improveQualityTangential              [1446-1451]
 *       MeshQuality::isotropicCleanup                      [1452-1464]
 *       VolumeProjection::projectToVolume                  [1465-1468]
 *
 *   if (dynamic_remesh):
 *     pick baseline or tight params by t                   [1473-1480]
 *     if step % remesh_every == 0:
 *       DynamicRemesh::remesh                              [1483-1490]
 *       MeshQuality::isotropicCleanup                      [1491-1504]
 *     if a remesh ran:
 *       VolumeProjection::projectToVolume                  [1514-1516]
 *       ABORT if not all finite -> "nonfinite dynamic-remesh state" [1517-1527]
 *
 *   if (field_filter_every && step % .. == 0 && t >= field_filter_after):
 *     if (threshold <= 0 || max|S| >= threshold):
 *       filterCirculationField                             [1528-1545]
 *       ABORT if not all finite -> "nonfinite filtered state" [1546-1556]
 *
 *   if (redistribute_every && step % .. == 0):
 *     MeshQuality::improveQualityTangential                [1557-1563]
 *     VolumeProjection::projectToVolume                    [1564-1565]
 *
 *   record the last finite state                           [1566-1569]
 *   checkpoint if due (by step count and/or by elapsed time) [1570-1590]
 *   print progress if due                                  [1604-1636]
 *
 * teardown:
 *   write the final last-finite checkpoint                 [1641-1652]
 * ```
 *
 * TWO INVARIANTS THAT ARE EASY TO BREAK
 * -------------------------------------
 * **The last-finite state is recorded *after* every mutation of the step, not
 * before** (lines 1566-1569). Recording it earlier would checkpoint a state
 * that the remesh or the filter subsequently corrupted; recording it only on
 * checkpoint steps would lose the work when the run aborts between
 * checkpoints. The final checkpoint writes *that* state, not the current one,
 * so a run that blows up still leaves a restartable file from just before the
 * blow-up.
 *
 * **Every abort is a global decision** (`Comm::allReduceAllFinite`). A rank
 * that breaks out of the loop while its peers continue leaves them blocked in
 * the next collective forever. See that function.
 *
 * WHAT IS NOT PORTED
 * ------------------
 * Everything downstream of `frames`, `times` and `diagnostics`: the matplotlib
 * figure, the mp4 writer, the section-plane diagnostic, the half-surface
 * clipping. The driver accepts and ignores their CLI options — see
 * `examples/02_adaptive_mesh_bubble/`.
 *
 * The `KeyboardInterrupt` handler (line 1637-1639) is ported as a SIGINT
 * handler in the example driver rather than here, because signal handling is a
 * property of the program, not of the library.
 */

#ifndef BEATNIK_SOLVER_HPP
#define BEATNIK_SOLVER_HPP

#include <Beatnik_AdaptiveMesh.hpp>
#include <Beatnik_BRSolverBase.hpp>
#include <Beatnik_CreateBRSolver.hpp>
#include <Beatnik_Diagnostics.hpp>
#include <Beatnik_DynamicRemesh.hpp>
#include <Beatnik_IOInterface.hpp>
#include <Beatnik_InitialCondition.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_MeshQuality.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_Restart.hpp>
#include <Beatnik_SourceQuadrature.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_TimeIntegrator.hpp>
#include <Beatnik_Types.hpp>
#include <Beatnik_VolumeProjection.hpp>
#include <Beatnik_ZModelSolver.hpp>

#include <mpi.h>

#include <memory>
#include <string>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief The complete set of run parameters, one struct per group.
 *
 * Assembled by the example driver from the CLI and handed to the solver whole,
 * so adding a parameter does not change the solver's signature.
 */
struct SolverParams
{
    StateModel state_model = StateModel::Potential;
    InitialConditionParams initial;
    ZModelParams zmodel;
    FmmParams fmm;
    TimeParams time;
    AmrParams amr;
    RemeshParams remesh;
    /// Tight remesh parameters, active past `remesh_tight_after`.
    RemeshParams remesh_tight;
    /// Simulation time past which `remesh_tight` takes over. Negative disables.
    Real remesh_tight_after = -1.0;
    /// Remesh cadence for the baseline and tight parameter sets.
    int remesh_every = 1;
    int remesh_tight_every = 1;
    /// Use the metric remesher (default) rather than indicator-driven AMR.
    bool dynamic_remesh = true;
    CleanupParams cleanup;
    FilterParams filter;
    CheckpointParams checkpoint;
    /// `--progress-time-interval`, default 0.25. Simulation time between
    /// progress lines; <= 0 prints only on the step-count schedule.
    Real progress_time_interval = 0.25;
    /// `--exact-gap-diagnostics`, default false.
    bool exact_gap_diagnostics = false;
};

//---------------------------------------------------------------------------//
/**
 * @brief The adaptive-mesh z-model bubble solver.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class Solver
{
  public:
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;
    using zmodel_type = ZModelSolver<ExecutionSpace, MemorySpace>;
    using integrator_type = TimeIntegrator<ExecutionSpace, MemorySpace>;
    using amr_type = AdaptiveMesh<ExecutionSpace, MemorySpace>;
    using remesh_type = DynamicRemesh<ExecutionSpace, MemorySpace>;
    using quality_type = MeshQuality<ExecutionSpace, MemorySpace>;
    using io_type = CheckpointIO<ExecutionSpace, MemorySpace>;
    using br_solver_type = BRSolverBase<ExecutionSpace, MemorySpace>;
    using quadrature_type = SourceQuadratureBase<ExecutionSpace, MemorySpace>;
    using diagnostics_type =
        DiagnosticsCalculator<ExecutionSpace, MemorySpace>;

    /**
     * @param comm   Communicator to decompose the surface over.
     * @param params The whole run configuration.
     */
    Solver( MPI_Comm comm, const SolverParams& params )
        : _comm( comm )
        , _params( params )
        , _mesh( comm )
        , _state( params.state_model )
    {
    }

    /**
     * @brief Build or restore the initial state and write the first checkpoint.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1196-1324)
     *
     * Order, and why:
     *  1. **Restart or cold start.** Dispatches to `RestartReader::load` or
     *     `InitialCondition::build`; nothing after this point branches on
     *     which. See `Beatnik_Restart.hpp`.
     *  2. **Construct the components.** The BR solver and the quadrature are
     *     built from the parameters through their factories, so the solver
     *     never names a concrete subclass.
     *  3. **Resolve the proximity radii.** `--remesh-proximity-activation-*`
     *     and `--remesh-proximity-material-exclusion-*` are each given as
     *     either an absolute distance or a factor times \f$h^0_{\min}\f$
     *     (lines 1272-1286), and \f$h^0_{\min}\f$ is only known now. Both the
     *     baseline and the tight parameter sets get the same resolved values.
     *  4. **Seed the material position**, if the restart did not supply it.
     *  5. **Write the startup checkpoint**, at the loaded or zero step/time.
     *
     * Step 3 before step 5 matters only for legibility, but step 1 before
     * step 3 is required: the radii depend on a quantity the restart carries.
     *
     * @note MPI. Collective throughout — `InitialCondition::build` distributes
     *       and exchanges, `coldStart` reduces twice, and `CheckpointIO::write`
     *       is a collective MPI-IO write. Every rank must call it.
     */
    void setup()
    {
        // 1. Restart or cold start. Nothing after this point branches on which.
        RestartState start;
        if ( _params.checkpoint.restarting() )
        {
            // T5b. `RestartReader::load` still throws; reached only with
            // --restart-from, which no regression test uses.
            ensureCheckpointIO();
            start = RestartReader<ExecutionSpace, MemorySpace>::load(
                *_io, _params.checkpoint.restart_from, _mesh, _state );
        }
        else
        {
            InitialCondition<ExecutionSpace, MemorySpace> initial(
                _params.initial );
            initial.build( _mesh, _state );
            start =
                RestartReader<ExecutionSpace, MemorySpace>::coldStart( _mesh );
        }

        _time = start.time;
        _step = start.step;
        _initial_volume = start.initial_volume;
        _initial_min_edge = start.initial_min_edge;
        _last_checkpoint_time = _time;
        _last_progress_time = _time;

        // 2. Construct the components through their factories, so the solver
        //    never names a concrete BR subclass or quadrature.
        _br_solver = createBRSolver<ExecutionSpace, MemorySpace>(
            _comm, _params.zmodel, _params.fmm );
        _quadrature = createSourceQuadrature<ExecutionSpace, MemorySpace>(
            _params.zmodel.source_quadrature );
        _zmodel.reset(
            new zmodel_type( _params.zmodel, *_br_solver, *_quadrature ) );
        _integrator.reset( new integrator_type( *_zmodel ) );
        _amr.reset( new amr_type( _params.amr ) );
        _remesh.reset( new remesh_type( _params.remesh ) );
        _quality.reset( new quality_type( _params.cleanup ) );

        // 3. Resolve the proximity radii, which are given as either an absolute
        //    distance or a factor times h0_min and cannot be resolved earlier
        //    because h0_min is only known now. BOTH parameter sets get the same
        //    resolved values (run_adaptive_mesh_bubble.py:1272-1286 resolves
        //    once and hands the result to both).
        resolveProximityRadii( _params.remesh );
        resolveProximityRadii( _params.remesh_tight );

        // 4. Seed the material position. Already done by
        //    `InitialCondition::build` on a cold start; on a restart it comes
        //    out of the checkpoint's `/vertices/u2`, and only a hand-written
        //    file lacking it needs the fallback.
        if ( start.from_checkpoint )
            _state.seedMaterialPosition( _mesh );

        // 5. The startup checkpoint, at the loaded or zero step and time.
        writeCheckpoint();
    }

    /**
     * @brief Run the time loop.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1398-1636)
     *
     * The loop body is transcribed in the file header. Returns when the step
     * budget is exhausted, `--t-end` is reached, or a non-finite state aborts
     * the run; in every case `finalize()` still writes the last finite state.
     *
     * @return False if the run stopped early (non-finite or interrupted), true
     *         if it completed its budget. The example driver uses this for its
     *         exit status.
     *
     * **T1c implements the `steps == 0` GUARD AND NOTHING ELSE.** The loop is
     * T2d's, and it is deliberately not begun here: `main` (1398-1636) makes
     * control-flow decisions — where the last-finite state is recorded,
     * where dt is clamped, in what order the post-step passes run — that
     * only make sense with the RHS in hand, and a half-transcribed loop would
     * commit to them early while looking finished.
     *
     * The guard is not a special case bolted on. `--steps 0` is a legitimate
     * configuration (it is exactly the one T1a generated the gold file with),
     * the Python's loop is `while local_step < steps` and so runs zero times,
     * and `setup()` has already written the startup checkpoint while
     * `finalize()` will write the final one. So returning `true` here is the
     * *correct* behaviour and not a stub: the run completed its budget.
     */
    bool solve()
    {
        if ( _params.time.steps <= 0 )
            return true;

        BEATNIK_NOT_IMPLEMENTED( "Solver", "solve" );
    }

    /**
     * @brief Write the final last-finite checkpoint.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1641-1652)
     *
     * Runs unconditionally after the loop, **including** after a non-finite
     * abort and after an interrupt, and writes the recorded last-finite state
     * rather than the current one. A no-op when `--checkpoint-dir` is empty.
     *
     * **T1c CAVEAT — "last finite" is not yet distinct from "current".** The
     * last-finite state is recorded inside the step loop, which is T2d's, so at
     * 0 timesteps the two coincide by construction and this writes the current
     * state. At `--steps 0` that means the same `(time, step)` and therefore
     * the same filename as `setup`'s startup checkpoint, so the file is written
     * twice — which is exactly what the Python does (`main` lines 1313-1324 and
     * 1641-1652 both fire at t=0/step=0) and is harmless because `writeMesh`
     * truncates. T2d must make the distinction real; until it does, do not read
     * a passing 0-step comparison as evidence that the last-finite path works.
     */
    void finalize() { writeCheckpoint(); }

    /**
     * @brief Advance one accepted step, with all its post-step passes.
     *
     * The body of the loop, factored out so a test can drive a single step.
     *
     * @return False if the step produced a non-finite state.
     */
    bool advanceOneStep()
    {
        BEATNIK_NOT_IMPLEMENTED( "Solver", "advanceOneStep" );
    }

    /**
     * @brief Graph-Laplacian smoothing of the evolved circulation field.
     *
     * Port of run_adaptive_mesh_bubble.py::filter_circulation_field
     * (lines 923-948)
     *
     * \f$\phi \leftarrow \phi + \lambda\,\mathcal{L}_{\text{graph}}\phi\f$,
     * repeated `--field-filter-iters` times (or the vector form on the sheet
     * vector). This is an explicit Jacobi diffusion sweep; with
     * \f$\lambda\f$ = `--field-filter-relaxation` (default 0.01) it is far
     * inside the stability limit, but note that unlike the viscous term in the
     * RHS it is **not** area-normalized and therefore is not a consistent
     * discretization of anything — it is a filter, applied outside the
     * time-stepping, and it changes the solution by an amount that depends on
     * the mesh.
     *
     * Off by default (`--field-filter-every 0`). Gated additionally by
     * `--field-filter-after` (simulation time) and `--field-filter-threshold`
     * (on \f$\|S\|_\infty\f$), so it can be armed to fire only once the sheet
     * starts to spike.
     *
     * The remeshed/filtered state is rebuilt, so the potential is re-centred
     * and the sheet vector re-projected as usual.
     */
    void filterCirculationField()
    {
        BEATNIK_NOT_IMPLEMENTED( "Solver", "filterCirculationField" );
    }

    /**
     * @brief Whether a checkpoint is due at the current step and time.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1570-1577)
     *
     * Due if `--checkpoint-every-steps` divides the step count, **or** if at
     * least `--checkpoint-every-time` of simulation time has elapsed since the
     * last checkpoint (with a \f$10^{-14}\f$ slack so an exactly-landing step
     * counts). Either criterion alone suffices; both may be active.
     */
    bool checkpointDue() const
    {
        BEATNIK_NOT_IMPLEMENTED( "Solver", "checkpointDue" );
    }

    /// The surface.
    const mesh_type& mesh() const { return _mesh; }
    /// The solution fields.
    const state_type& state() const { return _state; }
    /// Current simulation time.
    Real time() const { return _time; }
    /// Current accepted-step counter.
    long long step() const { return _step; }
    /// Run configuration, including the proximity radii `setup()` resolved.
    const SolverParams& params() const { return _params; }

    /// \f$V_0\f$, the target the volume projection drives toward all run.
    Real initialVolume() const { return _initial_volume; }

    /// \f$h^0_{\min}\f$, the scale dt and the proximity radii are expressed in.
    Real initialMinEdge() const { return _initial_min_edge; }

    /// Path of the last checkpoint written, empty if none. Exposed so a test
    /// can find the file to compare without reconstructing the naming rule —
    /// which would be a second implementation of `CheckpointIO::timeKey`, and
    /// the two would drift.
    const std::string& lastCheckpointPath() const
    {
        return _last_checkpoint_path;
    }

  private:
    /// Construct the checkpoint IO lazily, so a run with no `--checkpoint-dir`
    /// never creates one and `writeCheckpoint()` stays a single guard.
    void ensureCheckpointIO()
    {
        if ( !_io )
            _io.reset( new io_type( _comm, _params.checkpoint.directory,
                                    _params.checkpoint.prefix ) );
    }

    /// Assemble the header from the current state and write one checkpoint.
    /// A no-op without `--checkpoint-dir`, matching the Python's
    /// `if checkpoint_dir is not None` guard on both write sites.
    void writeCheckpoint()
    {
        if ( !_params.checkpoint.writing() )
            return;
        ensureCheckpointIO();

        CheckpointHeader header;
        header.state_model = _state.model();
        header.time = _time;
        header.step = _step;
        header.initial_volume = _initial_volume;
        header.initial_min_edge = _initial_min_edge;
        // Always true for a Beatnik-written file: it is `/vertices/u2`, a slot
        // in the vertex user pack `writeMesh` writes unconditionally.
        header.has_material_position = true;

        _last_checkpoint_path = _io->write( header, _mesh );
        _last_checkpoint_time = _time;
    }

    /// Resolve one parameter set's two proximity radii against
    /// \f$h^0_{\min}\f$.
    ///
    /// Port of run_adaptive_mesh_bubble.py::main (lines 1272-1286)
    ///
    /// An absolute value wins; a non-positive one falls back to
    /// `factor * initial_min_edge`, and **only if the factor is itself
    /// positive** — a zero factor leaves the radius zero, which is how the
    /// Python disables the term rather than a case it forgot.
    void resolveProximityRadii( RemeshParams& remesh ) const
    {
        if ( remesh.proximity_activation_distance <= Real( 0 ) &&
             remesh.proximity_activation_factor > Real( 0 ) )
            remesh.proximity_activation_distance =
                remesh.proximity_activation_factor * _initial_min_edge;

        if ( remesh.proximity_material_exclusion_radius <= Real( 0 ) &&
             remesh.proximity_material_exclusion_factor > Real( 0 ) )
            remesh.proximity_material_exclusion_radius =
                remesh.proximity_material_exclusion_factor * _initial_min_edge;
    }

    MPI_Comm _comm;
    SolverParams _params;

    mesh_type _mesh;
    state_type _state;

    std::unique_ptr<br_solver_type> _br_solver;
    std::unique_ptr<quadrature_type> _quadrature;
    std::unique_ptr<zmodel_type> _zmodel;
    std::unique_ptr<integrator_type> _integrator;
    std::unique_ptr<amr_type> _amr;
    std::unique_ptr<remesh_type> _remesh;
    std::unique_ptr<quality_type> _quality;
    std::unique_ptr<io_type> _io;

    Real _time = 0.0;
    long long _step = 0;
    Real _initial_volume = 0.0;
    Real _initial_min_edge = 0.0;
    Real _last_checkpoint_time = 0.0;
    Real _last_progress_time = 0.0;

    /// Path of the most recent checkpoint, for `lastCheckpointPath()`.
    std::string _last_checkpoint_path;

    /// Running totals reported on the progress line.
    long long _refine_events = 0;
    long long _edge_flips = 0;
    long long _remesh_events = 0;
};

} // namespace Beatnik

#endif // BEATNIK_SOLVER_HPP
