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
 * RECONCILED AGAINST TESSERA 2026-08-11 (task M2). The pre-M2 version of this
 * header was written without reading `../tessera`: it invented a `/mesh/...`
 * plus `/fields/...` schema and a rank-0 gather-and-write. Neither survives.
 * Every semantic that changed is flagged inline with **M2 CHANGE**. The Tessera
 * sources this was read against are `Tessera_HDF5Writer.hpp`,
 * `Tessera_HDF5Reader.hpp` and `Tessera_IoCommon.hpp`.
 *
 * ADAPTER CONTRACT
 * ----------------
 * No other Beatnik header may name a Tessera or HDF5 type.
 *
 * WHO WRITES WHAT — THE M2 DECISION
 * ---------------------------------
 * The task left one thing genuinely open: does Tessera write the checkpoint, or
 * does Beatnik gather to rank 0 and write it? **Tessera writes it.**
 *
 * `Tessera::writeMesh( mesh, stem )` is a collective MPI-IO write of the
 * **owned** entities of every rank, each exactly once, into a clean partition
 * of `<stem>.h5`. Dense global vertex/edge indices come from an `MPI_Exscan`
 * over owned-only counts, and connectivity is translated into those dense
 * indices before it is written — which is precisely the local-to-global
 * renumbering the pre-M2 header warned the gather path would have to do by
 * hand. It also carries **every vertex user field**, so all three Beatnik
 * fields are written for free (see the field pack in
 * `Beatnik_MeshInterface.hpp`).
 *
 * So the gather is not merely unnecessary, it is worse on every axis: O(global)
 * memory on rank 0, a serialized write, and a hand-rolled reimplementation of
 * the one genuinely error-prone step. **M2 CHANGE —
 * `Beatnik_Communication.hpp::gatherForCheckpoint` is deleted.**
 *
 * What Tessera does *not* write is the run's scalar metadata. Its only
 * root-attribute types are `int` and `uint64` (`Tessera_IoCommon.hpp`:
 * `writeIntAttr` / `writeU64Attr`), and the checkpoint needs a `double` time,
 * two `double` scalars and a *string*. Beatnik therefore appends a small
 * `/beatnik` group after `writeMesh` returns. The division of labour is:
 *
 *   Tessera writes the mesh and the fields, collectively, in parallel.
 *   Beatnik appends five scalars from rank 0.
 *
 * THE CHECKPOINT SCHEMA
 * ---------------------
 * Beatnik writes HDF5; the Python reference writes `np.savez_compressed`
 * `.npz`. `tests/regression_tests/compare_output.py` maps between them through
 * a hand-maintained table (`FIELD_MAP` at the top of that script) — the two
 * names must be kept in sync **there**, not guessed at either end. `FIELD_MAP`
 * was updated to the table below in the same change as this header.
 *
 * Python writer: `run_adaptive_mesh_bubble.py::save_state_checkpoint`
 * (lines 955-990). **M2 CHANGE — the HDF5 column is now Tessera's actual
 * layout**, not the grouping this header used to invent:
 *
 * | `.npz` key                 | HDF5 dataset                | Shape      | By |
 * |----------------------------|-----------------------------|------------|----|
 * | `vertices`                 | `/vertices/position`        | (Nv,3) f8  | T |
 * | `faces`                    | `/faces/verts`              | (Nf,3) u8  | T |
 * | `potential`                | `/vertices/u0`              | (Nv,) f8   | T |
 * | `sheet_vector`             | `/vertices/u1`              | (Nv,3) f8  | T |
 * | `remesh_material_position` | `/vertices/u2`              | (Nv,3) f8  | T |
 * | `state_model`              | `/beatnik/state_model`      | scalar str | B |
 * | `time`                     | `/beatnik/time`             | scalar f8  | B |
 * | `step`                     | `/beatnik/step`             | scalar i8  | B |
 * | `initial_volume`           | `/beatnik/initial_volume`   | scalar f8  | B |
 * | `initial_min_edge`         | `/beatnik/initial_min_edge` | scalar f8  | B |
 *
 * (T = written by Tessera's `writeMesh`, B = appended by Beatnik afterwards.)
 *
 * `/faces/verts` is `uint64` and holds **dense global vertex indices**, not
 * local ones and not gids: the writer assigns vertex `i` of rank `r` the index
 * `exscan_offset(r) + i` and writes `/vertices/position` at those same offsets,
 * so a face's three entries index rows of `/vertices/position` directly. That
 * is exactly the `.npz` `faces` convention, so the comparator needs no
 * translation — only a dtype widening.
 *
 * Tessera additionally writes the whole `/edges` group, `/vertices/gid`,
 * `/faces/{gid,edges,level}`, the conforming-mode closure datasets, the root
 * attributes (`format_version`, `refinement_mode`, `dim`, `scalar_bytes`,
 * `Nv`/`Ne`/`Nf`, the user-field counts and extents) and a `<stem>.xmf` XDMF
 * sidecar for Paraview. The Python has no analogue for any of it; the
 * comparator ignores everything not in `FIELD_MAP`, and `readMesh` requires all
 * of it.
 *
 * **THE `u0` / `u1` / `u2` HAZARD — read this before adding a vertex field.**
 * Tessera names user fields *positionally*: the `j`-th entry of the vertex user
 * pack is written as `/vertices/u<j>`. Nothing in the file says `u0` is the
 * potential. The mapping is `Beatnik::VertexFieldId` in
 * `Beatnik_MeshInterface.hpp`, and **reordering that enum silently relabels
 * every existing checkpoint** — a comparison would then pass or fail on the
 * wrong dataset, which is the one failure mode `compare_output.py`'s
 * hand-maintained `FIELD_MAP` exists to prevent. Because the name is positional
 * rather than semantic, the table above is not self-enforcing the way a named
 * dataset would be.
 *
 * The writer therefore also emits `/beatnik/vertex_field_names`, a `(Count,)`
 * array of UTF-8 strings in `VertexFieldId` order, holding the **`.npz` key**
 * of each slot: `["potential", "sheet_vector", "remesh_material_position"]`.
 * `compare_output.py` does not *infer* paths from it — `FIELD_MAP` stays the
 * hand-maintained source of truth — it **verifies** it, and fails loudly on a
 * disagreement. Adding or reordering a vertex field means updating
 * `VertexFieldId`, this table, `FIELD_MAP`, and `make_fixtures.py`, in one
 * change.
 *
 * BOTH STATE FIELDS ARE ALWAYS PRESENT
 * ------------------------------------
 * **M2 CHANGE.** The Python writes `potential` *or* `sheet_vector` according to
 * the state model, and never both. Beatnik cannot: the two are slots in one
 * Cabana tuple, and `writeMesh` writes the whole user pack unconditionally. A
 * Beatnik checkpoint therefore always carries `/vertices/u0` **and**
 * `/vertices/u1`, one of which is the evolved unknown and the other a cache
 * (under `Potential`, `u1` is whatever `updateSheetVector` last left there).
 *
 * `/beatnik/state_model` is what says which is which, and `compare_output.py`
 * was changed in the same commit to compare only the *active* field and to
 * ignore the inactive one rather than reporting it as a one-sided presence.
 * `remesh_material_position` keeps the strict both-or-neither rule, because
 * there it is a real signal.
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
 * READING IS `readMesh`, NOT `adopt`
 * ----------------------------------
 * **M2 CHANGE, and the largest one on the read side.**
 * `Tessera::readMesh( mesh, halo, stem )` reconstructs the whole distributed
 * mesh — vertices, edges, faces, connectivity, ownership, ghosts, both CSRs and
 * the three halo plans — *and every vertex user field* from the file. It does
 * so with a fresh dense-block partition and a `migrate()`, deliberately
 * unrelated to the partition that wrote the file, so a checkpoint round-trips
 * across a **rank count change**.
 *
 * So the pre-M2 read path — rank 0 reads, `Comm::broadcastFromRoot`,
 * `SurfaceMesh::adopt` — is gone, and with it the `state` and `material`
 * out-parameters: under the M1 field-pack model the solution *is* in the mesh,
 * so there is nothing left to hand back separately. `read` below takes the mesh
 * alone and returns the header. (`adopt` and `broadcastFromRoot` survive, but
 * for the *other* caller: the R1 mitigation that reads the initial surface out
 * of a Python `.npz` gold file, which is not a Tessera file and cannot go
 * through `readMesh`.)
 *
 * Three consequences the caller must know:
 *
 *  1. **A checkpoint is tied to the exact build.** `readMesh` validates
 *     `format_version`, `refinement_mode`, `dim`, `scalar_bytes`, the
 *     user-field counts and every field extent against the compile-time mesh
 *     type, and on any mismatch calls **`MPI_Abort`** — it does not throw, so
 *     `read` cannot turn it into the `std::runtime_error`
 *     `Beatnik_Restart.hpp` documents. A single-precision build cannot read a
 *     double-precision checkpoint, and a hanging-node build cannot read
 *     Beatnik's conforming files.
 *  2. **The halo comes back 1-deep and must be widened.** `readMesh` hands its
 *     freshly-constructed halo to `migrate()`, and a never-built halo records
 *     `depth == 0`, which Tessera reads as the historical 1. The Beatnik RHS is
 *     a two-ring stencil (risk R8), so `read` must follow `readMesh` with
 *     `Tessera::rebuildHalo( mesh, halo, 2 )`. Nothing else in the restart path
 *     re-states the depth, and `buildVertexStencil( mesh, 2 )` now throws
 *     rather than returning short rows, so forgetting this is loud — but it is
 *     still the adapter's job, not the caller's.
 *  3. **The Python's missing-field fallbacks are dead code here.** A Tessera
 *     file always carries all three vertex fields and the `/beatnik` group, so
 *     the "missing `remesh_material_position` -> seed from vertices" and
 *     "missing `initial_volume` -> recompute" branches can never fire on a file
 *     Beatnik wrote. They are retained on `read` only for a hand-written or
 *     externally-produced file, and `CheckpointHeader::has_material_position`
 *     is always `true` for a Beatnik-written checkpoint.
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
 * Beatnik reproduces the naming exactly, changing only the extension, so a
 * directory of Python gold files and a directory of Beatnik output pair up by
 * filename. Note Tessera appends the `.h5` itself, so what is handed to
 * `writeMesh` is the **stem**, without an extension.
 *
 * **M2 CHANGE — `_latest` is a symlink, not a second copy.** The Python
 * rewrites the whole payload a second time under the `_latest` name. Doing that
 * here would double the cost of every checkpoint at production scale, and a
 * plain byte copy is not equivalent either: the `<stem>.xmf` sidecar names its
 * `.h5` by stem, so a copied `_latest.xmf` would point at the wrong file. Rank
 * 0 instead replaces `<prefix>_latest.h5` and `<prefix>_latest.xmf` with
 * symlinks to the timestamped pair — same directory, so the sidecar's relative
 * reference still resolves — which is O(1) and always consistent.
 * `--restart-from <dir>/<prefix>_latest.h5` follows the link transparently.
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
 * This is the `/beatnik` group of the schema table — the part Tessera does not
 * write, because its root attributes are `int` and `uint64` only. Grouped so
 * the reader can return it in one piece and the restart path can hand it
 * straight to the solver.
 */
struct CheckpointHeader
{
    /// Which unknown the file stores; `/beatnik/state_model`. **M2 CHANGE — it
    /// no longer determines which field is *present*** (a Beatnik file always
    /// carries both `u0` and `u1`), only which one is the evolved unknown and
    /// which is a stale cache.
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

    /// True when the file carried a material-position field. **Always true for
    /// a Beatnik-written checkpoint** — it is `/vertices/u2`, a slot in the
    /// vertex user pack that `writeMesh` writes unconditionally. Retained for a
    /// hand-written file, where a false value makes the restart path seed the
    /// material coordinate from the vertex positions
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
     * @param comm      Communicator the surface is decomposed over. Must be the
     *                  mesh's own communicator: `writeMesh` and `readMesh` are
     *                  collective on `mesh.comm()`, and the barriers around the
     *                  `/beatnik` append must match it.
     * @param directory Output directory. Created if absent, as the Python
     *                  writer does (`directory.mkdir(parents=True,
     *                  exist_ok=True)`, line 966). Created by rank 0 before any
     *                  rank opens a file, since MPI-IO will not create it.
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
     * *including* the sign and the decimal point, so keys sort
     * lexicographically in time order for nonnegative times.
     *
     * Beatnik's own, with no Tessera involvement: Tessera takes a stem and
     * appends `.h5`/`.xmf`, and has no opinion about what the stem contains.
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
     * @brief Write one checkpoint, plus the relinked `_latest` alias.
     *
     * Port of run_adaptive_mesh_bubble.py::save_state_checkpoint
     * (lines 955-990)
     *
     * **M2 CHANGE — the signature loses `state` and `material`.** Under the M1
     * field pack all three per-vertex fields live inside the mesh, and
     * `writeMesh` writes the whole vertex user pack, so passing them separately
     * would name storage that no longer exists.
     *
     * TESSERA MAPPING, in order:
     *   1. rank 0 creates `_directory`; barrier.
     *   2. `Tessera::writeMesh( mesh, stem )` — collective MPI-IO, owned
     *      entities only, dense-renumbered connectivity, all vertex user
     *      fields, plus the `<stem>.xmf` sidecar rank 0 writes at the end.
     *   3. barrier — `writeMesh` closes the file but does not barrier after the
     *      sidecar, and step 4 reopens it.
     *   4. rank 0 reopens `<stem>.h5` read-write with the **default (serial)**
     *      file-access property list and creates the `/beatnik` group: the five
     *      scalars of `CheckpointHeader` plus `vertex_field_names`. An HDF5 file
     *      is format-identical whichever driver wrote it, so a serial reopen of
     *      an MPI-IO-written file is well defined; the barriers are what keep
     *      it from racing the other ranks.
     *   5. rank 0 relinks `<prefix>_latest.h5` / `.xmf`; barrier.
     *
     * @param header Scalar metadata. `state_model` is recorded, not used to
     *               select a field — see the schema section of the file header.
     * @param mesh   Surface providing vertices, connectivity **and** all three
     *               per-vertex fields.
     * @return Path of the timestamped file (not the `_latest` alias).
     *
     * @tparam MeshType `SurfaceMesh<ExecutionSpace, MemorySpace>`. Still
     *         templated only to keep this header free of a mesh include; it is
     *         no longer "pending an interface", so the pre-M2 `TODO(types)`
     *         marker is gone.
     *
     * @note MPI. **Collective on `mesh.comm()`**, and every rank must call it
     *       even if it owns nothing — `Tessera::detail::writeHyperslab` selects
     *       an empty hyperslab rather than skipping the call, precisely because
     *       skipping deadlocks. No gather: ghosts are excluded and connectivity
     *       is renumbered to dense global indices by the writer, so neither of
     *       the two bugs the pre-M2 note warned about (duplicated vertices,
     *       local indices in a concatenated file) is reachable from here.
     */
    template <class MeshType>
    std::string write( const CheckpointHeader& header, const MeshType& mesh )
    {
        (void)header;
        (void)mesh;
        BEATNIK_NOT_IMPLEMENTED( "CheckpointIO", "write" );
    }

    /**
     * @brief Read a checkpoint back into a mesh.
     *
     * Port of run_adaptive_mesh_bubble.py::load_state_checkpoint
     * (lines 993-1033)
     *
     * **M2 CHANGE — the signature loses `state` and `material`**, for the same
     * reason `write` does, and the implementation is `Tessera::readMesh`, not
     * `Comm::broadcastFromRoot` + `SurfaceMesh::adopt`.
     *
     * TESSERA MAPPING:
     *   1. `Tessera::readMesh( mesh, halo, stem )` — collective. Each rank
     *      takes a fresh contiguous block of dense face indices *unrelated to
     *      the writing partition*, fetches the vertex/edge records it
     *      references, and hands the result to `migrate()`, which recomputes
     *      ownership and the ghost layer. All three vertex user fields are
     *      restored with their vertices. **A checkpoint therefore round-trips
     *      across a change of rank count.**
     *   2. `Tessera::rebuildHalo( mesh, halo, 2 )` — mandatory, see point 2 of
     *      the read section in the file header: what `readMesh` leaves is a
     *      1-deep halo and the RHS needs two rings.
     *   3. rank 0 reads `/beatnik`, and the header is broadcast.
     *
     * Fallbacks the Python applies. **All of them are unreachable on a
     * Beatnik-written file** (see point 3 of the read section); they exist for
     * a hand-written one:
     *   - missing `time`  -> 0.0
     *   - missing `step`  -> 0
     *   - missing `initial_volume`   -> recomputed from the loaded mesh
     *   - missing `initial_min_edge` -> recomputed from the loaded mesh
     *   - missing material position  -> seeded from the vertices
     *
     * @param path File to read. Absolute, or relative to the CWD — *not*
     *             relative to `_directory`, matching `--restart-from`. A
     *             trailing `.h5` is stripped before it is handed to Tessera,
     *             which appends its own.
     * @param[out] mesh Surface to populate, fields included.
     * @return The scalar metadata.
     *
     * @note MPI. Collective. Not a rank-0 read and scatter: the whole point of
     *       `readMesh`'s fresh block partition is that no rank ever holds the
     *       global mesh.
     *
     * @warning A structural mismatch between the file and this build —
     *          precision, dimension, refinement mode, or the vertex field pack
     *          — is an **`MPI_Abort` inside Tessera**, not an exception. It
     *          cannot be caught and reported as the `std::runtime_error`
     *          `Beatnik_Restart.hpp` promises. What `read` *can* check first,
     *          and should, is the cheap Beatnik-side half: that the file
     *          exists, that `/beatnik` is present, and that `state_model` is
     *          one Beatnik recognizes.
     */
    template <class MeshType>
    CheckpointHeader read( const std::string& path, MeshType& mesh )
    {
        (void)path;
        (void)mesh;
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
