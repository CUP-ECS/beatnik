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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <stdexcept>
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
    using device_type = Kokkos::Device<ExecutionSpace, MemorySpace>;

    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using scalar_view = Kokkos::View<Real*, device_type>;
    // TODO(types): templated pending Tessera/Canopy interface; collapse to a
    // concrete type once known.
    using vector_view = Kokkos::View<Real* [3], device_type>;

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

        // 5. Initialize the AMR reference state. UNCONDITIONAL, not gated on
        //    `--refine-every > 0`: `FaceFieldId::{ReferenceArea,
        //    ReferenceCurvature, RefineMark}` are slots in Tessera's face
        //    AoSoA, which is allocated uninitialized, and `writeMesh` writes the
        //    whole face pack — so a run that skipped this would checkpoint
        //    whatever was in memory. It is also step 1 of the reference's own
        //    reset list (`Beatnik_AdaptiveMesh.hpp`, THE REFERENCE STATE):
        //    `TriangleSurfaceState.__post_init__` seeds both from the current
        //    geometry whenever they are not supplied, which on a cold start and
        //    on a restart is always.
        _amr->resetReferenceState( _mesh );

        // 6. The startup checkpoint, at the loaded or zero step and time.
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
     * The `steps == 0` guard is not a special case bolted on. `--steps 0` is a
     * legitimate configuration (it is exactly the one T1a generated the gold
     * file with), the Python's loop is `while local_step < steps` and so runs
     * zero times, and `setup()` has already written the startup checkpoint while
     * `finalize()` will write the final one. So returning `true` there is the
     * *correct* behaviour: the run completed its budget.
     *
     * **T2d — the loop is here, and the post-step passes it cannot yet run
     * THROW rather than being skipped.** `requireSupportedConfiguration()` is
     * called once before the loop and rejects, by name and by task ID, any
     * configuration whose post-step passes are still stubs (T4d/T4e/T5c; T4a,
     * T4b and T4c have landed and their rejections are gone). It
     * is a configuration check and not a mid-loop guard on purpose: the
     * conditions are global and time-independent, so failing before the first
     * step is both cheaper and more honest than aborting at step 5. Silently
     * skipping them instead would let a run *look* like the reference's default
     * configuration while omitting the adaptivity that configuration is about.
     */
    bool solve()
    {
        if ( _params.time.steps <= 0 )
            return true;

        requireSupportedConfiguration();

        for ( int local_step = 0; local_step < _params.time.steps;
              ++local_step )
        {
            // The reference breaks BEFORE incrementing the counters
            // (run_adaptive_mesh_bubble.py:1402-1403), so a run that lands
            // exactly on --t-end does not take a zero-length step.
            if ( _params.time.have_t_end &&
                 _time >= _params.time.t_end - Real( 1.0e-14 ) )
                break;

            if ( !advanceOneStep() )
                return false;
        }
        return true;
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
     * **T2d — the distinction is now real.** `advanceOneStep` records the three
     * owned vertex fields, the positions, and the `(time, step)` pair after
     * every mutation of a step that ended finite; this restores them into the
     * mesh before writing, so a run that blows up at step N still leaves a
     * restartable file describing step N-1. At `--steps 0` nothing was ever
     * recorded and the current state is written, which is the T1c behaviour and
     * is what keeps regression test 1 unchanged: same `(time, step)`, therefore
     * the same filename as `setup`'s startup checkpoint, written twice — exactly
     * what the Python does (`main` 1313-1324 and 1641-1652 both fire at
     * t=0/step=0) and harmless because `writeMesh` truncates.
     */
    void finalize()
    {
        restoreLastFiniteState();
        writeCheckpoint();
    }

    /**
     * @brief Advance one accepted step, with all its post-step passes.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1404-1636), the loop body
     *
     * The body of the loop, factored out so a test can drive a single step.
     *
     * Order, and every piece of it is the reference's:
     *  1. `++step` **before** dt is chosen, because the checkpoint and progress
     *     schedules are both `step % n == 0` and the reference increments first
     *     (line 1405).
     *  2. `chooseStepSize`, then the two clamps that are the *caller's* and not
     *     that function's — `--dt-switch-time` and landing exactly on `--t-end`
     *     (lines 1407-1410).
     *  3. One TVD-RK3 step; advance `time` by the dt actually taken.
     *  4. The global finiteness check. **Every abort is a global decision**
     *     (`SurfaceState::allFinite` reduces it), so no rank can leave the loop
     *     while its peers block in the next collective.
     *  5. Record the last-finite state — **after** every mutation of the step,
     *     not before (lines 1566-1569).
     *  6. Checkpoint if due, then the throttled progress line.
     *
     * @return False if the step produced a non-finite state.
     *
     * @note MPI. Collective throughout, and **uniformly so**: the BR ring inside
     *       the integrator deadlocks unless every rank reaches it the same
     *       number of times per step, including a rank that owns zero sources
     *       (T2c). Every branch here is therefore driven by a globally identical
     *       quantity — the step counter, the simulation time, or a reduced
     *       verdict — and never by a rank-local count.
     */
    bool advanceOneStep()
    {
        ++_step;

        Real dt = _integrator->chooseStepSize( _mesh, _state, _params.time,
                                               _initial_min_edge );
        if ( _params.time.dt_switch_time >= Real( 0 ) &&
             _time >= _params.time.dt_switch_time )
            dt = std::min( dt, _params.time.dt_after_switch );
        if ( _params.time.have_t_end )
            dt = std::min( dt, _params.time.t_end - _time );

        _integrator->step( _mesh, _state, dt );
        _time += dt;
        _last_dt = dt;

        if ( !_state.allFinite( _mesh ) )
        {
            reportStop( "nonfinite mesh-RHS state" );
            return false;
        }

        // -- the indicator-driven refine branch (T4a) ---------------------
        // run_adaptive_mesh_bubble.py:1424-1468. The cadence condition is
        // globally identical (it is built from the step counter and two
        // parameters), so every rank enters the collective refiner together.
        //
        // The filter branch (T5c) is still a stub, as are the collapse/flip/
        // smooth thirds of the remesh branch below (T4d) and both of its
        // proximity paths (T4e); `requireSupportedConfiguration` has already
        // rejected any configuration that would reach one — so there is nothing
        // to skip here and nothing that could be silently skipped. **T4c landed
        // the tangential pass and the redistribute branch**, so those two are no
        // longer in that list.
        if ( !_params.dynamic_remesh && _params.amr.refine_every > 0 &&
             _step % _params.amr.refine_every == 0 )
        {
            _last_refinement = _amr->refine( _mesh, _state );
            if ( _last_refinement.marked_faces > 0 )
            {
                _refine_events += _last_refinement.marked_faces;

                // The reference's three quality repairs go here
                // (lines 1440-1464). Two of the three are still rejected at
                // setup, by name and task ID, so the counter below is a
                // transcription of the reference's own gate rather than a
                // placeholder:
                //
                //   improveConnectivityByFlips   T4d (blocked, Tessera G5c)
                //   isotropicCleanup             T4d (blocked, Tessera G5c)
                const int flips = 0;

                // MeshQuality::improveQualityTangential (lines 1446-1450) —
                // T4c. Called UNCONDITIONALLY once anything was marked, with
                // `--smooth-iters` (default 1) and `--smooth-relaxation`
                // (0.12); the pass makes its own no-op decision when either is
                // zero, exactly as the reference's early return does.
                //
                // NOTHING RE-BASES THE AMR REFERENCE STATE AFTER IT. The
                // reference passes `reset_reference=False` here (`:1449`) and
                // `reset_reference=(smooth_iters == 0)` to the flip pass above
                // — which, with `--flip-passes 0`, returns having changed
                // nothing and re-bases nothing. An `AdaptiveMesh::
                // resetReferenceState` call here would change every subsequent
                // refinement decision, so there is none; T4d owns the question
                // for the flip path.
                _quality->improveQualityTangential(
                    _mesh, _params.filter.smooth_iters,
                    _params.filter.smooth_relaxation );

                // `project_state_to_volume` is gated on a repair having
                // ACTUALLY RUN (lines 1465-1468), not on the refinement having
                // happened — `flips > 0 or smooth_iters > 0 or
                // isotropic_cleanup`. **T4c: this is where the projection first
                // runs on the REFINE path**, because dropping the
                // `--smooth-iters` rejection makes the middle clause true at
                // the default `--smooth-iters 1` — which is exactly why T4a
                // transcribed the gate in full rather than folding it to
                // `false`. (T4b was where `projectToVolume` first ran at all,
                // on the remesh path.)
                const bool repaired = ( flips > 0 ) ||
                                      ( _params.filter.smooth_iters > 0 ) ||
                                      _params.cleanup.enabled;
                if ( _params.zmodel.preserve_volume && repaired )
                    VolumeProjection<ExecutionSpace,
                                     MemorySpace>::projectToVolume( _mesh,
                                                                    _initial_volume );
            }
        }

        // -- the metric-remesh branch (T4b) --------------------------------
        // run_adaptive_mesh_bubble.py:1469-1527. Mutually exclusive with the
        // branch above by construction (`:1424` versus `:1469`), and the
        // cadence condition is again built only from the step counter and two
        // parameters, so every rank enters the collective remesher together.
        if ( _params.dynamic_remesh )
        {
            // The tight parameter set (`:1473-1480`) would swap `_params.remesh`
            // for `_params.remesh_tight` past `--remesh-tight-after`. It is
            // unported and `requireSupportedConfiguration` rejects
            // `--remesh-tight-after >= 0`, so the active set is always the
            // baseline one and the swap is not written as dead code.
            const int every = _params.remesh_every;
            const bool remeshed = ( every > 0 ) && ( _step % every == 0 );
            if ( remeshed )
            {
                _last_remesh = _remesh->remesh( _mesh, _state );
                _remesh_events += _last_remesh.splits + _last_remesh.collapses +
                                  _last_remesh.flips;

                // `mesh_quality.isotropic_cleanup` runs here (`:1493-1504`).
                // T4d, blocked on Tessera G5c, and `--isotropic-cleanup` is
                // rejected at setup rather than skipped.

                // The reference rebuilds the state from the remeshed arrays,
                // which (a) re-bases the reference area and curvature —
                // `dynamic_remesh_state_with_material` passes both as `None`
                // (`:1099-1101`) and `TriangleSurfaceState.__post_init__`
                // re-seeds them from the current geometry — and (b) re-centres
                // the potential against the NEW area weights
                // (`mesh_solver.py:155-162`). Both are done here, in that
                // order, because both are properties of state construction
                // rather than of the remesher.
                _amr->resetReferenceState( _mesh );
                {
                    MeshGeometry<ExecutionSpace, MemorySpace> geometry;
                    geometry.compute( _mesh.positions(),
                                      _mesh.totalVertexCount(),
                                      _mesh.faceVertices() );
                    _state.centerPotential( _mesh, geometry );
                }
            }

            // `project_state_to_volume` is gated on a remesh having RUN, not on
            // it having changed anything (`:1513-1516`) — unlike the refine
            // branch above, which gates on a repair having run. So under
            // `--dynamic-remesh` (and without `--no-preserve-volume`) the
            // absolute volume projection executes every remesh step. **T4b is
            // therefore where `VolumeProjection::projectToVolume` first runs**;
            // T4a's entry and T2d's `Affects:` note both guessed T4c/T4d.
            if ( remeshed )
            {
                if ( _params.zmodel.preserve_volume )
                    VolumeProjection<ExecutionSpace, MemorySpace>::
                        projectToVolume( _mesh, _initial_volume );

                if ( !_state.allFinite( _mesh ) )
                {
                    reportStop( "nonfinite dynamic-remesh state" );
                    return false;
                }
            }
        }

        // -- the periodic redistribution branch (T4c) -----------------------
        // run_adaptive_mesh_bubble.py:1557-1565. **NOT load balancing**, in
        // spite of the name: it is the tangential pass on its own cadence plus
        // the volume projection, and nothing else — the Python is serial and
        // repartitions nothing, so this branch is not blocked on T5d and
        // `Comm::redistribute` stays T5d's. Off by default
        // (`--redistribute-every 0`); the cadence is built from the step counter
        // and one parameter, so every rank enters the collective pass together.
        //
        // The projection's gate here is NOT the refine branch's: the reference
        // asks only `not no_preserve_volume and args.smooth_iters > 0`
        // (`:1564-1565`), i.e. it is the tangential pass having been asked for,
        // with no `flips`/`isotropic_cleanup` clause.
        if ( _params.filter.redistribute_every > 0 &&
             _step % _params.filter.redistribute_every == 0 )
        {
            _quality->improveQualityTangential(
                _mesh, _params.filter.smooth_iters,
                _params.filter.smooth_relaxation );

            if ( _params.zmodel.preserve_volume &&
                 _params.filter.smooth_iters > 0 )
                VolumeProjection<ExecutionSpace, MemorySpace>::projectToVolume(
                    _mesh, _initial_volume );
        }

        recordLastFiniteState();

        if ( checkpointDue() )
            writeCheckpoint();

        writeProgressIfDue( dt );
        return true;
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
     *
     * Always false without `--checkpoint-dir`: the reference guards both
     * criteria on `checkpoint_dir is not None` (lines 1571, 1573), so a run with
     * no output directory is not "due but suppressed", it is not due.
     */
    bool checkpointDue() const
    {
        if ( !_params.checkpoint.writing() )
            return false;

        bool due = false;
        if ( _params.checkpoint.every_steps > 0 )
            due = due || ( _step % _params.checkpoint.every_steps == 0 );
        if ( _params.checkpoint.every_time > Real( 0 ) )
            due = due || ( _time - _last_checkpoint_time >=
                           _params.checkpoint.every_time - Real( 1.0e-14 ) );
        return due;
    }

    /// The surface.
    const mesh_type& mesh() const { return _mesh; }

    /// The surface, mutably. **T2d — added for the test, and the reason is the
    /// M1 storage model rather than convenience:** every geometric accessor
    /// (`positions()`, `faceVertices()`, `edgeVertices()`) is non-const, because
    /// the first two return Cabana slices of a non-const member and the third
    /// builds and caches `Tessera::MeshGeometry` against `generation()`. So a
    /// caller that wants to *measure* the surface — regression test 2 checks the
    /// enclosed volume every step — cannot do it through the const accessor.
    /// Same constraint that widened twelve signatures at T2c.
    mesh_type& mesh() { return _mesh; }
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

    /// What the most recent refinement pass did, or a default-constructed
    /// value if none has run. Exposed so the T4a regression test can check the
    /// projection against the realized face count and log R12's two signals
    /// against the round index without re-deriving either — the projection in
    /// particular is only knowable *before* the edit, so a test cannot
    /// reconstruct it afterwards.
    const RefinementDiagnostics& lastRefinement() const
    {
        return _last_refinement;
    }

    /// What the most recent dynamic-remesh call did, or a default-constructed
    /// value if none has run. T4b, and exposed for the same reason
    /// `lastRefinement()` is: the split pass's **candidate** count and the
    /// cap's verdict are knowable only inside the call, and T4b's exit
    /// criterion asserts on the difference between them.
    const RemeshDiagnostics& lastRemesh() const { return _last_remesh; }

    /// Path of the last checkpoint written, empty if none. Exposed so a test
    /// can find the file to compare without reconstructing the naming rule —
    /// which would be a second implementation of `CheckpointIO::timeKey`, and
    /// the two would drift.
    const std::string& lastCheckpointPath() const
    {
        return _last_checkpoint_path;
    }

  private:
    /// Reject, before the first step, any configuration whose post-step passes
    /// are still stubs — by name and by the task that owns them.
    ///
    /// **Why a throw and not a skip.** The reference's *default* configuration
    /// runs the dynamic remesher every step, so a Beatnik run that quietly
    /// omitted it would produce a plausible trajectory that is not the
    /// reference's and would be compared against gold files generated with
    /// adaptivity on. Every condition below is global and time-independent, so
    /// this is a configuration error and belongs before the loop; a mid-loop
    /// throw would be collective-safe too, but it would report at step 5 what
    /// was decidable at step 0.
    void requireSupportedConfiguration() const
    {
        // The two adaptivity modes are mutually exclusive per run
        // (`run_adaptive_mesh_bubble.py:1424` versus `:1469-1471`), and each
        // has its own set of unimplemented follow-on passes. Each is rejected
        // by name and by task ID rather than silently skipped, because a run
        // that omitted one would produce a plausible trajectory that is not the
        // reference's.
        //
        // **T4b CORRECTION — these are NOT rejected unconditionally, and the
        // comment here used to say they were.** All of them were already
        // guarded by `refining`, which is false under `--dynamic-remesh`; that
        // was harmless only while `--dynamic-remesh` was rejected outright, and
        // the moment T4b dropped that rejection every one of them became a
        // live hole — `--isotropic-cleanup` defaults to TRUE and the reference
        // runs `mesh_quality.isotropic_cleanup` after every dynamic remesh. So
        // the remesh branch now carries its own rejections below, and the
        // comment describes what the code does.
        const bool refining =
            !_params.dynamic_remesh && _params.amr.refine_every > 0;
        const bool remeshing =
            _params.dynamic_remesh && _params.remesh_every > 0;

        if ( refining && _params.filter.flip_passes > 0 )
            throw std::logic_error(
                "Beatnik::Solver::solve: --flip-passes > 0 under "
                "--refine-every needs MeshQuality::improveConnectivityByFlips, "
                "which is task T4d and still throws. T4d is additionally "
                "BLOCKED upstream: Tessera implements no edge flip "
                "(../tessera/tasks/edge-flip.md, gap G5c). Pass "
                "--flip-passes 0." );

        // **T4c: the `--smooth-iters > 0` rejection is GONE.**
        // `MeshQuality::improveQualityTangential` is implemented, so the refine
        // branch's unconditional tangential pass
        // (`run_adaptive_mesh_bubble.py:1446-1450`) runs at the default
        // `--smooth-iters 1` — and so, through the repair gate, does the volume
        // projection. `--redistribute-every > 0` is accepted for the same
        // reason; its rejection at the bottom of this function is gone too.

        if ( ( refining || remeshing ) && _params.cleanup.enabled )
            throw std::logic_error(
                "Beatnik::Solver::solve: --isotropic-cleanup (the default) "
                "under --refine-every or --dynamic-remesh needs "
                "MeshQuality::isotropicCleanup, which is task T4d and still "
                "throws; its valence-equalizing flips are blocked on the same "
                "Tessera gap G5c. Pass --no-isotropic-cleanup." );

        //-- the metric remesher's own rejections (T4b) ----------------------//
        //
        // T4b landed the sizing field and the SPLIT third of
        // `dynamic_remesh.py`. The collapse, flip and smoothing thirds are T4d
        // (the first two blocked on Tessera gaps G5b and G5c), and the two
        // proximity paths are T4e. Rather than invent a `--dynamic-remesh-split
        // -only` switch, a run is accepted exactly when the reference's OWN
        // knobs make those thirds no-ops — so what Beatnik runs is what the
        // reference would run, not a Beatnik-only subset of it. Each rejection
        // below names the method, the task, and the knob that satisfies it.
        if ( remeshing && _params.remesh.use_proximity )
            throw std::logic_error(
                "Beatnik::Solver::solve: --remesh-proximity under "
                "--dynamic-remesh needs "
                "DynamicRemesh::nonlocalFaceCentroidDistance, which is task "
                "T4e and still throws -- a genuinely global spatial query with "
                "two exclusion criteria, which no ghost depth makes local. It "
                "is off by default (dynamic_remesh.py:33); pass "
                "--no-remesh-proximity." );

        if ( remeshing && _params.remesh.surgical_proximity )
            throw std::logic_error(
                "Beatnik::Solver::solve: --remesh-surgical-proximity under "
                "--dynamic-remesh needs "
                "DynamicRemesh::splitSurgicalProximityEdges and "
                "::nonlocalFaceProximityPairs, which are task T4e and still "
                "throw. Off by default (dynamic_remesh.py:41); pass "
                "--no-remesh-surgical-proximity." );

        // The collapse third. `--remesh-collapse-factor 0` makes the
        // reference's candidate predicate `length < 0 * target`
        // (dynamic_remesh.py:373) false for every edge, so the pass returns
        // before any mutation -- and the reference's own tight profile ships
        // that value (run_adaptive_mesh_bubble.py:520), so it is an in-family
        // configuration rather than an invention.
        //
        // `--remesh-max-collapses 0` is NOT a second lever, however plausible
        // it looks: the driver maps a non-positive value to `None` = UNLIMITED
        // (`:1350-1352`), which `RemeshParams::max_collapses_per_pass`
        // reproduces. Accepting on it would accept a run in which the reference
        // still collapses.
        if ( remeshing && _params.remesh.collapse_factor > Real( 0 ) )
            throw std::logic_error(
                "Beatnik::Solver::solve: --remesh-collapse-factor > 0 (the "
                "default is 0.45) under --dynamic-remesh needs "
                "DynamicRemesh::collapseShortEdges, which is task T4d and "
                "still throws. T4d is additionally BLOCKED upstream: Tessera "
                "implements no edge collapse (../tessera/tasks/edge-collapse.md, "
                "gap G5b). Pass --remesh-collapse-factor 0, which is what the "
                "reference's own tight profile passes and which makes its "
                "collapse pass an exact no-op. NOTE: --remesh-max-collapses 0 "
                "does NOT disable collapse -- it means UNLIMITED." );

        if ( remeshing && _params.remesh.smoothing_iterations > 0 )
            throw std::logic_error(
                "Beatnik::Solver::solve: --remesh-smooth-iters > 0 (the "
                "default is 1) under --dynamic-remesh needs "
                "DynamicRemesh::tangentialSmooth, which is task T4d and still "
                "throws. Pass --remesh-smooth-iters 0, which makes the "
                "reference's own pass return immediately "
                "(dynamic_remesh.py:463-465)." );

        if ( remeshing &&
             _params.remesh.flip_min_gain < kFlipsDisabledMinGain )
            throw std::logic_error(
                "Beatnik::Solver::solve: --remesh-flip-min-gain below 1e12 "
                "(the default is 1e-3) under --dynamic-remesh needs "
                "DynamicRemesh::flipEdgesForQuality, which is task T4d and "
                "still throws. T4d is additionally BLOCKED upstream: Tessera "
                "implements no edge flip (../tessera/tasks/edge-flip.md, gap "
                "G5c). Pass --remesh-flip-min-gain 1e12: triangle quality lies "
                "in [0,1], so the reference's accept test "
                "min(new) > min(old)*(1+gain) (dynamic_remesh.py:449-450) is "
                "then unsatisfiable and its flip pass mutates nothing." );

        // The tight parameter set is UNPORTED: `--remesh-tight-*` has no
        // representation in `RemeshParams` beyond the prose note on it, so
        // `SolverParams::remesh_tight` is a copy of the baseline set and a run
        // past `--remesh-tight-after` would silently keep remeshing at the
        // baseline parameters instead of tightening. Unassigned to a task; see
        // the T4b entry in tasks/framework-progress-log.md.
        if ( remeshing && _params.remesh_tight_after >= Real( 0 ) )
            throw std::logic_error(
                "Beatnik::Solver::solve: --remesh-tight-after >= 0 selects the "
                "reference's tight remesh parameter set "
                "(run_adaptive_mesh_bubble.py:1358-1396), which is NOT ported: "
                "no --remesh-tight-* option reaches RemeshParams, so the run "
                "would silently continue at the baseline parameters. Pass "
                "--remesh-tight-after -1." );

        if ( _params.filter.field_filter_every > 0 )
            throw std::logic_error(
                "Beatnik::Solver::solve: --field-filter-every > 0 needs "
                "Solver::filterCirculationField, which is task T5c and still "
                "throws. Pass --field-filter-every 0." );
    }

    /// Rank 0 prints the reference's stop line verbatim
    /// (run_adaptive_mesh_bubble.py:1417-1421), so a Beatnik log and a Python
    /// log of the same failure read the same.
    void reportStop( const char* reason ) const
    {
        int rank = 0;
        MPI_Comm_rank( _comm, &rank );
        if ( rank != 0 )
            return;
        char line[256];
        std::snprintf( line, sizeof( line ), "stopping at step=%lld t=%.4f: %s",
                       _step, static_cast<double>( _time ), reason );
        std::cout << line << std::endl;
    }

    /// Record the last finite state, **after** every mutation of the step.
    ///
    /// Port of run_adaptive_mesh_bubble.py::main (lines 1566-1569)
    ///
    /// Under the M1 storage model the solution lives in the Tessera vertex user
    /// pack, so "keeping a reference to the state" is not available the way the
    /// Python's `last_finite_state = state` is: the next step overwrites those
    /// slots in place. What is kept instead is a copy of the four owned arrays
    /// plus the `(time, step)` pair — which is exactly what `finalize()` needs
    /// to write and nothing more.
    void recordLastFiniteState()
    {
        const int n = _mesh.ownedVertexCount();
        if ( static_cast<int>( _last_finite_position.extent( 0 ) ) != n )
        {
            _last_finite_position = vector_view(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_last_finite_position" ),
                n );
            _last_finite_material = vector_view(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_last_finite_material" ),
                n );
            _last_finite_potential = scalar_view(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_last_finite_potential" ),
                n );
            _last_finite_sheet = vector_view(
                Kokkos::view_alloc( Kokkos::WithoutInitializing,
                                    "beatnik_last_finite_sheet" ),
                n );
        }

        auto pos = _mesh.positions();
        auto phi = _mesh.potential();
        auto sheet = _mesh.sheetVector();
        auto material = _mesh.materialPosition();
        auto lp = _last_finite_position;
        auto lm = _last_finite_material;
        auto lf = _last_finite_potential;
        auto ls = _last_finite_sheet;
        Kokkos::parallel_for(
            "beatnik_record_last_finite",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                {
                    lp( i, d ) = pos( i, d );
                    lm( i, d ) = material( i, d );
                    ls( i, d ) = sheet( i, d );
                }
                lf( i ) = phi( i );
            } );
        Kokkos::fence();

        _last_finite_time = _time;
        _last_finite_step = _step;
        _have_last_finite = true;
    }

    /// Put the recorded last-finite state back into the mesh, so `finalize()`
    /// writes *that* rather than the current one.
    ///
    /// A no-op when nothing was ever recorded (`--steps 0`, or an abort in the
    /// first step), which is what makes T1c's 0-timestep behaviour unchanged.
    /// Also a no-op if the owned vertex count has changed since the record,
    /// which cannot happen today (no adaptivity) and would mean the copy no
    /// longer describes this mesh; that is reported rather than papered over,
    /// because writing a mismatched field pack would produce a plausible file.
    void restoreLastFiniteState()
    {
        if ( !_have_last_finite )
            return;

        const int n = _mesh.ownedVertexCount();
        if ( static_cast<int>( _last_finite_position.extent( 0 ) ) != n )
        {
            int rank = 0;
            MPI_Comm_rank( _comm, &rank );
            if ( rank == 0 )
                std::cout << "Beatnik::Solver::finalize: the recorded "
                             "last-finite state has "
                          << _last_finite_position.extent( 0 )
                          << " owned vertices but the mesh now has " << n
                          << "; writing the CURRENT state instead."
                          << std::endl;
            return;
        }

        auto pos = _mesh.positions();
        auto phi = _mesh.potential();
        auto sheet = _mesh.sheetVector();
        auto material = _mesh.materialPosition();
        auto lp = _last_finite_position;
        auto lm = _last_finite_material;
        auto lf = _last_finite_potential;
        auto ls = _last_finite_sheet;
        Kokkos::parallel_for(
            "beatnik_restore_last_finite",
            Kokkos::RangePolicy<ExecutionSpace>( 0, n ),
            KOKKOS_LAMBDA( const int i ) {
                for ( int d = 0; d < 3; ++d )
                {
                    pos( i, d ) = lp( i, d );
                    material( i, d ) = lm( i, d );
                    sheet( i, d ) = ls( i, d );
                }
                phi( i ) = lf( i );
            } );
        Kokkos::fence();

        // Only owned rows were restored; `writeMesh` writes owned entities only,
        // but the exchange keeps the mesh self-consistent for anything that
        // inspects it after `finalize()`.
        _mesh.haloExchange();

        _time = _last_finite_time;
        _step = _last_finite_step;
    }

    /// The reference's progress throttle: step 1, every `steps/10` steps, and
    /// whenever `--progress-time-interval` of simulation time has elapsed
    /// (run_adaptive_mesh_bubble.py:1604-1608).
    ///
    /// `Diagnostics::compute` is collective, so the condition must be identical
    /// on every rank — it is built from the step counter and the simulation
    /// time, both of which are.
    void writeProgressIfDue( Real dt )
    {
        const bool by_time =
            _params.progress_time_interval > Real( 0 ) &&
            ( _time - _last_progress_time >= _params.progress_time_interval );
        const long long stride =
            std::max<long long>( _params.time.steps / 10, 1 );
        if ( !( _step == 1 || _step % stride == 0 || by_time ) )
            return;

        _last_progress_time = _time;
        const Diagnostics diag = diagnostics_type::compute(
            _mesh, _state, _initial_volume, _params.remesh,
            _params.exact_gap_diagnostics );

        int rank = 0;
        MPI_Comm_rank( _comm, &rank );
        if ( rank == 0 )
            diagnostics_type::writeProgressLine( std::cout, diag, _step, _time,
                                                 dt, _refine_events,
                                                 _edge_flips, _remesh_events );
    }

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
    /// The dt the most recent step actually took, after both caller clamps.
    Real _last_dt = 0.0;

    /// The last finite state, recorded after every successful step and written
    /// by `finalize()`. See `recordLastFiniteState`.
    vector_view _last_finite_position;
    vector_view _last_finite_material;
    vector_view _last_finite_sheet;
    scalar_view _last_finite_potential;
    Real _last_finite_time = 0.0;
    long long _last_finite_step = 0;
    bool _have_last_finite = false;

    /// Path of the most recent checkpoint, for `lastCheckpointPath()`.
    std::string _last_checkpoint_path;

    /// What the most recent refinement pass did. T4a.
    RefinementDiagnostics _last_refinement;

    /// What the most recent dynamic-remesh call did. T4b.
    RemeshDiagnostics _last_remesh;

    /// Running totals reported on the progress line.
    long long _refine_events = 0;
    long long _edge_flips = 0;
    long long _remesh_events = 0;
};

} // namespace Beatnik

#endif // BEATNIK_SOLVER_HPP
