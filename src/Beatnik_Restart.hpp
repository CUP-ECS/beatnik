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
 * @file Beatnik_Restart.hpp
 * @brief Restart setup, isolated from the cold-start path.
 *
 * WHY THIS IS ITS OWN FILE
 * ------------------------
 * Restart is the one setup path with a *semantic* consequence rather than a
 * merely mechanical one: it re-bases the AMR change indicators. Keeping it
 * separate means the cold-start path — which is what the regression tests
 * exercise — has no branch on it, and a broken or unimplemented restart cannot
 * block those tests from passing.
 *
 * Concretely: `Solver::setup` dispatches to either `InitialCondition::build` or
 * `RestartReader::load`, and nothing downstream of that point knows which ran,
 * except through the values in `RestartState`. Regression tests 1-3 never take
 * this path.
 *
 * THE RE-BASING PROBLEM
 * ---------------------
 * Port of run_adaptive_mesh_bubble.py::load_state_checkpoint
 * (lines 993-1033) — specifically the `reference_face_area=None,
 * reference_face_curvature=None` arguments at lines 1003-1004 and 1011-1012
 *
 * A checkpoint stores the mesh, the fields, and five scalars. It does **not**
 * store `reference_face_area` or `reference_face_curvature`, the per-face
 * baselines the area-change and curvature-change AMR indicators measure
 * against. On load, both are reconstructed from the restarted geometry, i.e.
 * set to "no change so far".
 *
 * The consequence: **a run restarted at step N does not reproduce the
 * trajectory of an uninterrupted run past step N.** Immediately after restart
 * both change indicators read zero, so the next refinement pass marks nothing
 * that the uninterrupted run would have marked from accumulated change. The
 * meshes then diverge, and with them the solutions. The divergence is a genuine
 * behavioral difference, not a floating-point artifact — no tolerance makes it
 * go away.
 *
 * This is faithful to the reference: the Python has exactly the same property,
 * deliberately (the fields are dropped, not accidentally omitted). It is
 * recorded as risk R3 in `tasks/framework.md`, and it is why no regression test
 * compares a restarted run against a continuous one.
 *
 * Note the third AMR indicator, the sagitta / curvature-resolution one, is
 * **absolute** and therefore unaffected. A run configured with
 * `--curvature-resolution-threshold > 0` and both change thresholds effectively
 * disabled would be restart-invariant in its refinement decisions. That is the
 * shape of a future mitigation, if one is ever wanted.
 */

#ifndef BEATNIK_RESTART_HPP
#define BEATNIK_RESTART_HPP

#include <Beatnik_IOInterface.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_Params.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <string>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief What a restart (or a cold start) establishes for the run.
 *
 * The cold-start path fills the same struct, so the loop below the setup does
 * not branch on which one ran.
 */
struct RestartState
{
    /// Simulation time to resume from. 0 on a cold start.
    Real time = 0.0;

    /// Step counter to resume from. 0 on a cold start. The loop runs
    /// `--steps` *further* steps from here, so a restart's `--steps` is a
    /// local budget, not a global target
    /// (`run_adaptive_mesh_bubble.py:1398-1401`).
    long long step = 0;

    /// Target enclosed volume for the whole run. Loaded from the checkpoint, or
    /// computed from the initial surface on a cold start.
    Real initial_volume = 0.0;

    /// \f$h^0_{\min}\f$, the scale the adaptive dt and the proximity radii are
    /// expressed in. Loaded, or computed from the initial surface.
    Real initial_min_edge = 0.0;

    /// True when this state came from a checkpoint.
    bool from_checkpoint = false;

    /// True when the AMR change-indicator references were re-based by the load.
    /// Always equal to `from_checkpoint`; kept separate so the reason is
    /// visible at the use site rather than implied.
    bool amr_reference_rebased = false;
};

//---------------------------------------------------------------------------//
/**
 * @brief Loads a checkpoint into a mesh and state, and reports the
 *        consequences.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space.
 */
template <class ExecutionSpace, class MemorySpace>
class RestartReader
{
  public:
    using mesh_type = SurfaceMesh<ExecutionSpace, MemorySpace>;
    using state_type = SurfaceState<ExecutionSpace, MemorySpace>;
    using io_type = CheckpointIO<ExecutionSpace, MemorySpace>;

    /**
     * @brief Restore a run from a checkpoint.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1199-1214)
     *
     * Reads the file, adopts the mesh, installs the fields, and applies the
     * fallbacks documented on `CheckpointIO::read`. If the file carried no
     * `remesh_material_position`, the material coordinate is seeded from the
     * loaded vertex positions (line 1208-1209) — which means the material
     * exclusion measures "distance since the *restart*", not since t=0, another
     * small behavioral difference from an uninterrupted run.
     *
     * The AMR reference state is **not** loaded and **not** reconstructed here;
     * the caller re-bases it via `AdaptiveMesh::resetReferenceState`. Making
     * that the caller's explicit step, rather than a side effect hidden in the
     * loader, is the point of this file.
     *
     * @param io        Checkpoint reader.
     * @param path      `--restart-from` value.
     * @param[out] mesh  Surface to populate.
     * @param[out] state Fields to populate.
     * @return The resumed time, step, and the two carried scalars.
     *
     * @throws std::runtime_error if the file is missing, unreadable, or carries
     *         an unrecognized `state_model`. A restart failure is a
     *         configuration error, so it aborts rather than silently cold
     *         starting — a silent cold start would produce a plausible-looking
     *         run of the wrong thing.
     */
    static RestartState load( io_type& io, const std::string& path,
                              mesh_type& mesh, state_type& state )
    {
        (void)io;
        (void)path;
        (void)mesh;
        (void)state;
        BEATNIK_NOT_IMPLEMENTED( "RestartReader", "load" );
    }

    /**
     * @brief Fill a `RestartState` for a cold start.
     *
     * Port of run_adaptive_mesh_bubble.py::main (lines 1225-1227, 1238-1240)
     *
     * Computes `initial_volume` and `initial_min_edge` from the freshly built
     * surface and sets the time and step to zero.
     *
     * @note MPI. Both quantities are global reductions —
     *       `Comm::allReduceSum` and `Comm::allReduceMin` — so on a run with a
     *       different rank count they differ in the last bits from a
     *       single-rank run. Since every subsequent dt and every proximity
     *       radius scales off them, trajectories are not bit-identical across
     *       rank counts. See risk R2 in `tasks/framework.md`.
     */
    static RestartState coldStart( const mesh_type& mesh )
    {
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "RestartReader", "coldStart" );
    }
};

} // namespace Beatnik

#endif // BEATNIK_RESTART_HPP
