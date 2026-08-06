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
 * @file Beatnik_IOInterface.hpp
 * @brief ADAPTER (2 of 3). Checkpoint read/write, backed by **Tessera's HDF5**
 *        mesh I/O.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Tessera or HDF5 type. Tessera has **not
 * been read** while writing this header; the schema below is dictated entirely
 * by the Python gold files, which is what the regression comparison actually
 * requires. The first task that opens `../tessera` (see `tasks/framework.md`,
 * task M2) will reconcile the two.
 *
 * THE CHECKPOINT SCHEMA
 * ---------------------
 * Beatnik writes HDF5; the Python reference writes `np.savez_compressed`
 * `.npz`. `tests/regression_tests/compare_output.py` maps between them through
 * a hand-maintained table (`FIELD_MAP` at the top of that script) — the two
 * names must be kept in sync **there**, not guessed at either end.
 *
 * Python writer: `run_adaptive_mesh_bubble.py::save_state_checkpoint`
 * (lines 955-990). Contents, in `.npz` key / HDF5 dataset order:
 *
 * | `.npz` key                  | HDF5 dataset          | Shape    | Meaning |
 * |-----------------------------|-----------------------|----------|---------|
 * | `state_model`               | `/state_model`        | scalar str | `potential` or `sheet-vector` |
 * | `time`                      | `/time`               | scalar f8 | simulation time |
 * | `step`                      | `/step`               | scalar i8 | accepted-step counter |
 * | `initial_volume`            | `/initial_volume`     | scalar f8 | enclosed volume of the *initial* surface |
 * | `initial_min_edge`          | `/initial_min_edge`   | scalar f8 | minimum edge length of the *initial* surface |
 * | `vertices`                  | `/mesh/vertices`      | (Nv,3) f8 | node positions |
 * | `faces`                     | `/mesh/faces`         | (Nf,3) i8 | triangle connectivity |
 * | `potential`                 | `/fields/potential`   | (Nv,) f8  | present iff `state_model == potential` |
 * | `sheet_vector`              | `/fields/sheet_vector`| (Nv,3) f8 | present iff `state_model == sheet-vector` |
 * | `remesh_material_position`  | `/fields/remesh_material_position` | (Nv,3) f8 | carried Lagrangian coordinate |
 *
 * Deliberately **absent**: everything derived or diagnostic. No
 * `frame_diagnostics` history, no volume-drift / quality / gap series, no
 * velocities or RHS, no CLI arguments, no `MeshZModelParams`, no frame history
 * for the video. A checkpoint is restartable but cannot reproduce a plot.
 *
 * Also absent, and this matters: `reference_face_area` and
 * `reference_face_curvature`. The Python loader resets both to `None`
 * (`run_adaptive_mesh_bubble.py:993-1033`), so **a restart re-bases the AMR
 * change indicators** against the restarted geometry. See
 * `Beatnik_Restart.hpp`, which is where that consequence is isolated.
 *
 * FILE NAMING
 * -----------
 * Port of run_adaptive_mesh_bubble.py::checkpoint_time_key (lines 951-952) and
 * ::save_state_checkpoint (lines 986-989). Every save writes **two** files:
 *
 *   `<prefix>_t<timekey>_step<step:07d>.h5`   and   `<prefix>_latest.h5`
 *
 * where `timekey` is `f"{time:012.6f}"` with `-` replaced by `m` and `.` by
 * `p` — e.g. t = 1.5 gives `0001p500000`, t = -0.25 gives `m00000p250000`.
 * The `_latest` file is overwritten each time. Beatnik reproduces the naming
 * exactly, changing only the extension, so a directory of Python gold files
 * and a directory of Beatnik output pair up by filename.
 *
 * WHEN CHECKPOINTS APPEAR
 * -----------------------
 * Port of run_adaptive_mesh_bubble.py::main (lines 1313-1324, 1570-1590,
 * 1641-1652):
 *   1. Once at startup, at the loaded (or zero) step and time.
 *   2. Every `--checkpoint-every-steps` accepted steps, and/or every
 *      `--checkpoint-every-time` of simulation time, whichever fires.
 *   3. Once after the loop, holding the **last finite state** — including
 *      after a nonfinite abort or a Ctrl-C.
 */

#ifndef BEATNIK_IOINTERFACE_HPP
#define BEATNIK_IOINTERFACE_HPP

#include <Beatnik_Types.hpp>

#include <mpi.h>

#include <string>

namespace Beatnik
{

//---------------------------------------------------------------------------//
/**
 * @brief The scalar metadata carried alongside the mesh in a checkpoint.
 *
 * Grouped so the reader can return it in one piece and the restart path can
 * hand it straight to the solver.
 */
struct CheckpointHeader
{
    /// Which unknown the file stores. Determines which of `potential` /
    /// `sheet_vector` is present.
    StateModel state_model = StateModel::Potential;

    /// Simulation time of the snapshot.
    Real time = 0.0;

    /// Accepted-step counter at the snapshot. A restart continues from here,
    /// so the step numbers in a restarted run's filenames are contiguous with
    /// the original's.
    long long step = 0;

    /// Enclosed volume of the **initial** surface. This is the target the
    /// volume projection drives toward for the whole run, so it must survive a
    /// restart or the bubble will drift to whatever volume it happened to have
    /// at restart time. Units of length^3.
    Real initial_volume = 0.0;

    /// Minimum edge length of the **initial** surface. The adaptive dt scaling
    /// and the proximity activation/exclusion radii are all expressed as
    /// multiples of it, so it too must survive a restart. Units of length.
    Real initial_min_edge = 0.0;

    /// True when the file carried a `remesh_material_position` field. When
    /// false the restart path seeds it from the vertex positions
    /// (`run_adaptive_mesh_bubble.py:1208-1209`).
    bool has_material_position = false;
};

//---------------------------------------------------------------------------//
/**
 * @brief Parallel checkpoint writer/reader. Tessera HDF5 lives behind this.
 *
 * @tparam ExecutionSpace Kokkos execution space.
 * @tparam MemorySpace    Kokkos memory space holding the fields to write.
 */
template <class ExecutionSpace, class MemorySpace>
class CheckpointIO
{
  public:
    /**
     * @param comm      Communicator the surface is decomposed over.
     * @param directory Output directory. Created if absent, as the Python
     *                  writer does (`directory.mkdir(parents=True,
     *                  exist_ok=True)`, line 966).
     * @param prefix    Filename prefix, `--checkpoint-prefix`.
     */
    CheckpointIO( MPI_Comm comm, std::string directory, std::string prefix )
        : _comm( comm )
        , _directory( std::move( directory ) )
        , _prefix( std::move( prefix ) )
    {
    }

    /**
     * @brief Format the time component of a checkpoint filename.
     *
     * Port of run_adaptive_mesh_bubble.py::checkpoint_time_key (lines 951-952)
     *
     * `f"{time:012.6f}"` then `-` -> `m`, `.` -> `p`. Zero-padded to width 12
     * *including* the sign and the decimal point, so keys sort lexicographically
     * in time order for nonnegative times.
     *
     * @param time Simulation time.
     * @return The key, e.g. `0001p500000`.
     */
    static std::string timeKey( Real time )
    {
        (void)time;
        BEATNIK_NOT_IMPLEMENTED( "CheckpointIO", "timeKey" );
    }

    /**
     * @brief Write one checkpoint, plus the overwritten `_latest` copy.
     *
     * Port of run_adaptive_mesh_bubble.py::save_state_checkpoint
     * (lines 955-990)
     *
     * @param header   Scalar metadata; `state_model` selects which field is
     *                 written.
     * @param mesh     Surface providing vertices and connectivity.
     * @param state    Solution state providing `potential` or `sheet_vector`.
     * @param material `(Nv,3)` carried material coordinate, or an empty view to
     *                 omit the dataset.
     * @return Path of the timestamped file (not the `_latest` copy).
     *
     * @tparam MeshType, StateType, MaterialView
     *         // TODO(types): templated pending Tessera/Canopy interface;
     *         // collapse to a concrete type once known.
     *
     * @note MPI. Collective. Vertices and faces are distributed, so this is a
     *       gather-and-write: see
     *       `Beatnik_Communication.hpp::gatherForCheckpoint`. Face
     *       connectivity must be renumbered from local to global vertex
     *       indices *before* the gather, or the written triangles reference the
     *       wrong nodes.
     */
    template <class MeshType, class StateType, class MaterialView>
    std::string write( const CheckpointHeader& header, const MeshType& mesh,
                       const StateType& state, const MaterialView& material )
    {
        (void)header;
        (void)mesh;
        (void)state;
        (void)material;
        BEATNIK_NOT_IMPLEMENTED( "CheckpointIO", "write" );
    }

    /**
     * @brief Read a checkpoint back into a mesh and state.
     *
     * Port of run_adaptive_mesh_bubble.py::load_state_checkpoint
     * (lines 993-1033)
     *
     * Fallbacks the Python applies, reproduced here so an older or
     * hand-written file still loads:
     *   - missing `time`  -> 0.0
     *   - missing `step`  -> 0
     *   - missing `initial_volume`   -> recomputed from the loaded mesh
     *   - missing `initial_min_edge` -> recomputed from the loaded mesh
     *   - missing `remesh_material_position` -> seeded from the vertices
     *
     * @param path     File to read. Absolute, or relative to the CWD — *not*
     *                 relative to `_directory`, matching `--restart-from`.
     * @param[out] mesh     Surface to populate (via `SurfaceMesh::adopt`).
     * @param[out] state    Solution state to populate.
     * @param[out] material `(Nv,3)` material coordinate to populate.
     * @return The scalar metadata.
     *
     * @note MPI. Collective. Rank 0 reads and the result is partitioned; the
     *       reference implementation is serial-read + scatter rather than a
     *       true parallel read, because the gold files are small and the
     *       partition is decided by Tessera anyway.
     */
    template <class MeshType, class StateType, class MaterialView>
    CheckpointHeader read( const std::string& path, MeshType& mesh,
                           StateType& state, MaterialView& material )
    {
        (void)path;
        (void)mesh;
        (void)state;
        (void)material;
        BEATNIK_NOT_IMPLEMENTED( "CheckpointIO", "read" );
    }

    /// Directory checkpoints are written to.
    const std::string& directory() const { return _directory; }

    /// Filename prefix.
    const std::string& prefix() const { return _prefix; }

  private:
    MPI_Comm _comm;
    std::string _directory;
    std::string _prefix;
};

} // namespace Beatnik

#endif // BEATNIK_IOINTERFACE_HPP
