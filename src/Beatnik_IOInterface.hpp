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
 * `writeIntAttr` / `writeU64Attr`), and the checkpoint needs a *string*
 * `state_model`, a `double` `time`, an `int64` `step`, and two more `double`s
 * (`initial_volume`, `initial_min_edge`) — **five** in total, none of them
 * expressible as a Tessera root attribute. Beatnik therefore appends a small
 * `/beatnik` group after `writeMesh` returns. The division of labour is:
 *
 *   Tessera writes the mesh and the fields, collectively, in parallel.
 *   Beatnik appends five scalars from rank 0.
 *
 * **The five are fixed by the gold files, not chosen here.** They are exactly
 * `compare_output.py`'s `REQUIRED_FIELDS` minus the two mesh arrays, which are
 * exactly the five 0-d keys
 * `run_adaptive_mesh_bubble.py::save_state_checkpoint` puts in its `payload`
 * (lines 966-972). The schema table below is the authority; `CheckpointHeader`
 * carries one field per scalar plus one (`has_material_position`) that is
 * derived on read and never written.
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
 * GROUPED OUTPUT — ONE PARAVIEW DATASET, NOT N
 * --------------------------------------------
 * Paraview's XDMF readers are not file-series readers, so N per-frame `.xmf`
 * sidecars are N unrelated datasets with no time slider over them. Grouping has
 * to be stated in the light data, and `Tessera::MeshSeries` states it: it writes
 * each frame exactly as before and additionally maintains **one master `.xmf`**,
 * an XDMF temporal collection naming every frame with its time, rewritten after
 * every frame (so a killed run still leaves a valid master over the frames that
 * exist). `write` therefore routes its frame through a `MeshSeries` member
 * rather than calling `Tessera::writeMesh` directly.
 *
 * A `--checkpoint-dir out --checkpoint-prefix checkpoint` run that takes N
 * checkpoints leaves, for N *distinct* `(time, step)` pairs:
 *
 *   N x `out/checkpoint_t<timekey>_step<step>.h5`   the frames
 *   N x `out/checkpoint_t<timekey>_step<step>.xmf`  per-frame sidecars
 *   1 x `out/checkpoint.xmf`                       THE MASTER — open this one
 *   1 x `out/checkpoint.xmfindex`                  Tessera's restart record
 *   2 x `out/checkpoint_latest.{h5,xmf}`           symlinks to the newest frame
 *
 * **`out/checkpoint.xmf` is the file to open in Paraview, with the *temporal*
 * XDMF3 reader (`Xdmf3ReaderT`).** Only that reader walks a temporal
 * collection; the plain XDMF3 reader opens the master and shows a single
 * timestep, which looks like a broken file and is not one.
 * `grep -c '<Time Value=' out/checkpoint.xmf` is the check that distinguishes
 * the two: if it equals N, the file is right and the reader choice was wrong.
 *
 * THE EQUAL-TIME RULE, and why it is here rather than in Tessera
 * -------------------------------------------------------------
 * `MeshSeries::write` throws when `time` is not *strictly* greater than the
 * previous frame's. Beatnik legitimately writes the same `(time, step)` twice:
 * `Solver::finalize()` re-writes the last finite state, which carries the same
 * time and step as the previous checkpoint whenever the last accepted step also
 * checkpointed (`Beatnik_Solver.hpp`, "the same filename as `setup`'s startup
 * checkpoint, written twice"). An unguarded port therefore throws at
 * `finalize()` on any run whose last step checkpointed. So `write` decides:
 *
 *   time > last, or no frame yet   -> `_series.write(...)`; the frame joins the
 *                                     master.
 *   time == last AND stem == last  -> the timed `Tessera::writeMesh(...)`
 *                                     directly. The frame is rewritten (as it
 *                                     already was before this change) but NOT
 *                                     appended: the master already names that
 *                                     exact file at that exact time, so nothing
 *                                     is lost and a duplicate timestep is kept
 *                                     off the time slider.
 *   anything else                  -> **throw**. A decreasing time, or an equal
 *                                     time under a different name, is
 *                                     unreachable — `recordLastFiniteState()`
 *                                     runs immediately before
 *                                     `checkpointDue()`, so the restored state
 *                                     is always at or after the last
 *                                     checkpoint. Reaching it means an
 *                                     invariant broke, and the loud failure is
 *                                     the point. Requiring the **stem** to
 *                                     match as well as the time is what keeps
 *                                     this branch from masking a future change
 *                                     that writes a *different* state at the
 *                                     same `(time, step)`.
 *
 * Two consequences worth knowing before they are discovered:
 *
 *  - **The per-frame sidecars now carry a `<Time Value=>` child**, because both
 *    the series and the equal-time branch go through the *timed* `writeMesh`
 *    overload rather than the timeless one this file used to call. That is a
 *    change to emitted light data, accepted rather than avoided: it makes a
 *    single frame self-describing, and nothing reads those sidecars —
 *    `compare_output.py` reads `.h5` datasets only. No `.h5` dataset changes.
 *  - **A series is not reopened across a restart.** On a restart the master
 *    would be rewritten with only the post-restart frames while Tessera appends
 *    to the pre-existing `.xmfindex`, leaving the two describing different frame
 *    lists. Unreachable today — `read` below is a throwing stub — and the fix
 *    belongs with the restart path (framework.md T5b owns it). Recorded in
 *    README "Known Issues".
 *
 * FILE NAMING
 * -----------
 * Port of run_adaptive_mesh_bubble.py::checkpoint_time_key (lines 951-952) and
 * ::save_state_checkpoint (lines 986-989). Every save writes a **frame pair**,
 * plus the two files that are per-run rather than per-frame:
 *
 *   `<prefix>_t<timekey>_step<step:07d>.h5`  + its `.xmf` sidecar   the frame
 *   `<prefix>_latest.h5` / `.xmf`                                   symlinks
 *   `<prefix>.xmf` / `.xmfindex`                    the master, rewritten
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
 * **`<prefix>_latest.xmf` deliberately still names the newest *frame* sidecar,
 * not the master.** It is half of the "latest checkpoint" pair whose other half
 * a restart consumes; repointing it at the collection master would break that
 * pairing, and the master is a third thing beside the pair rather than a newer
 * version of one of them.
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

#include <Beatnik_AdaptiveMesh.hpp>
#include <Beatnik_MeshInterface.hpp>
#include <Beatnik_SurfaceState.hpp>
#include <Beatnik_Types.hpp>

#include <Tessera.hpp>

#include <hdf5.h>
#include <mpi.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace Beatnik
{

namespace IoDetail
{

//---------------------------------------------------------------------------//
/// `mkdir -p`, for the checkpoint directory. Rank 0 only; see `write`.
///
/// The Python does `directory.mkdir(parents=True, exist_ok=True)`
/// (`save_state_checkpoint`, line 966). MPI-IO will not create a directory, so
/// this must happen before any rank opens the file — which is what the first
/// barrier in `write` is for.
inline void makeDirectories( const std::string& path )
{
    if ( path.empty() )
        return;
    std::string partial;
    partial.reserve( path.size() );
    std::size_t i = 0;
    if ( path[0] == '/' )
    {
        partial += '/';
        i = 1;
    }
    while ( i <= path.size() )
    {
        if ( i == path.size() || path[i] == '/' )
        {
            if ( !partial.empty() && partial != "/" )
            {
                // EEXIST is the expected case, not an error: `exist_ok=True`.
                if ( ::mkdir( partial.c_str(), 0755 ) != 0 && errno != EEXIST )
                    throw std::runtime_error(
                        "Beatnik::CheckpointIO: cannot create checkpoint "
                        "directory '" +
                        partial + "'" );
            }
            if ( i == path.size() )
                break;
            partial += '/';
        }
        else
        {
            partial += path[i];
        }
        ++i;
    }
}

//---------------------------------------------------------------------------//
/// Write one scalar dataset of type `type` at `path`, rank 0, serial driver.
///
/// A 0-d (`H5S_SCALAR`) dataset, which is what `compare_output.py` reads with
/// `handle[path][()]` and then `np.asarray(...).reshape(())` — the same shape
/// `np.savez` gives a 0-d array, so the two sides are symmetric. `make_fixtures
/// .py` writes the fixtures the same way.
inline void writeScalar( hid_t loc, const char* name, hid_t type,
                         const void* value )
{
    const hid_t space = H5Screate( H5S_SCALAR );
    const hid_t dset = H5Dcreate2( loc, name, type, space, H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT );
    H5Dwrite( dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, value );
    H5Dclose( dset );
    H5Sclose( space );
}

/// Write a scalar variable-length UTF-8 string dataset. `h5py` reads this back
/// as `bytes`, which `compare_output.py::scalar_str` decodes.
inline void writeString( hid_t loc, const char* name, const std::string& value )
{
    const hid_t type = H5Tcopy( H5T_C_S1 );
    H5Tset_size( type, H5T_VARIABLE );
    H5Tset_cset( type, H5T_CSET_UTF8 );
    const char* raw = value.c_str();
    writeScalar( loc, name, type, &raw );
    H5Tclose( type );
}

/// Write a rank-1 array of variable-length UTF-8 strings.
inline void writeStringArray( hid_t loc, const char* name,
                              const std::vector<const char*>& values )
{
    const hid_t type = H5Tcopy( H5T_C_S1 );
    H5Tset_size( type, H5T_VARIABLE );
    H5Tset_cset( type, H5T_CSET_UTF8 );
    const hsize_t dims[1] = { static_cast<hsize_t>( values.size() ) };
    const hid_t space = H5Screate_simple( 1, dims, nullptr );
    const hid_t dset = H5Dcreate2( loc, name, type, space, H5P_DEFAULT,
                                   H5P_DEFAULT, H5P_DEFAULT );
    H5Dwrite( dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data() );
    H5Dclose( dset );
    H5Sclose( space );
    H5Tclose( type );
}

/// Replace `link` with a symlink to `target`. Rank 0 only.
///
/// `unlink` first, because `symlink` fails with EEXIST rather than replacing.
/// A missing link is the normal first-checkpoint case, so ENOENT is ignored.
/// Failure to relink is NOT fatal: the timestamped file — the one this call
/// returns and the one a test reads — is already safely on disk, and losing a
/// convenience alias must not lose a checkpoint.
inline void relink( const std::string& link, const std::string& target )
{
    ::unlink( link.c_str() );
    if ( ::symlink( target.c_str(), link.c_str() ) != 0 )
        std::fprintf( stderr,
                      "Beatnik::CheckpointIO: warning: cannot link %s -> %s "
                      "(the timestamped checkpoint was written; only the "
                      "_latest alias is missing)\n",
                      link.c_str(), target.c_str() );
}

} // namespace IoDetail

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
        // Declaration order runs this after `_directory` and `_prefix` have
        // been moved into, which is why `_series` is declared after them.
        // `MeshSeries` writes nothing until its first `write()`, so a value
        // member on a run with no checkpoints costs a string and an empty
        // vector and no I/O.
        , _series( masterStem( _directory, _prefix ) )
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
        // `f"{time:012.6f}"`. C's "%012.6f" is the same specification: width 12
        // including the sign and the point, zero-padded, six fractional digits.
        // 32 bytes covers any double the formatter can produce for a width-12
        // request that overflows its width (a large |t| widens the integer part
        // rather than truncating), and `snprintf` cannot overrun it.
        char buffer[32];
        std::snprintf( buffer, sizeof( buffer ), "%012.6f",
                       static_cast<double>( time ) );
        std::string key( buffer );
        for ( char& c : key )
        {
            if ( c == '-' )
                c = 'm';
            else if ( c == '.' )
                c = 'p';
        }
        return key;
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
     *   2. `_series.write( mesh, stem, time )` — collective MPI-IO, owned
     *      entities only, dense-renumbered connectivity, all vertex user
     *      fields, plus the `<stem>.xmf` sidecar rank 0 writes at the end, plus
     *      the rewritten master `<directory>/<prefix>.xmf` and one appended
     *      `.xmfindex` line. On the exact `(time, stem)` repeat that
     *      `Solver::finalize` writes, the timed `Tessera::writeMesh` is called
     *      directly instead and the series is left alone — see GROUPED OUTPUT
     *      in the file header.
     *   3. barrier — `writeMesh` closes the file but does not barrier after the
     *      sidecar, and step 4 reopens it.
     *   4. rank 0 reopens `<stem>.h5` read-write with the **default (serial)**
     *      file-access property list and creates the `/beatnik` group: the
     *      five scalars of `CheckpointHeader` plus `vertex_field_names`. An
     *      HDF5 file is format-identical whichever driver wrote it, so a serial
     *      reopen of an MPI-IO-written file is well defined; the barriers are
     *      what keep it from racing the other ranks.
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
        int rank = 0;
        MPI_Comm_rank( _comm, &rank );

        const std::string stem = checkpointStem( header.time, header.step );

        // 1. Rank 0 creates the directory. MPI-IO will not, so this must
        //    complete before any rank opens the file -- hence the barrier.
        if ( rank == 0 )
            IoDetail::makeDirectories( _directory );
        MPI_Barrier( _comm );

        // 2. Tessera writes the mesh, the connectivity and the whole vertex
        //    user pack, collectively, owned entities only, dense-renumbered.
        //    Note it takes the STEM and appends `.h5` / `.xmf` itself.
        //
        //    Routed through `_series`, so the frame additionally joins the
        //    master temporal collection `<directory>/<prefix>.xmf` -- see the
        //    GROUPED OUTPUT section of the file header. `MeshSeries::write`
        //    throws on a non-increasing time, and Beatnik legitimately writes
        //    the same `(time, step)` twice (`Solver::finalize` re-writes the
        //    last finite state), so the equal-time case is handled here rather
        //    than inside the series.
        //
        //    `Real` may be `float` while the series stores `double`; convert
        //    once into a local so the comparison and the stored value cannot
        //    disagree.
        const double time = static_cast<double>( header.time );
        if ( _series.numFrames() == 0 || time > _last_frame_time )
        {
            _series.write( mesh.tesseraMesh(), stem, time );
            _last_frame_time = time;
            _last_frame_stem = stem;
        }
        else if ( time == _last_frame_time && stem == _last_frame_stem )
        {
            // The same frame, written a second time. The master already names
            // this exact stem at this exact time, so appending it again would
            // both trip the strictly-increasing rule and put a duplicate
            // timestep on Paraview's time slider. Rewrite the frame -- today's
            // documented behaviour is that it is written twice and truncated,
            // and this change is confined to the light data -- but through the
            // TIMED `writeMesh` overload, so the sidecar is byte-identical to
            // the one the series would have written.
            Tessera::writeMesh( mesh.tesseraMesh(), stem, time );
        }
        else
        {
            // Unreachable by construction: `recordLastFiniteState()` runs
            // immediately before `checkpointDue()`, so the state `finalize()`
            // restores is always at or after the last checkpoint's step, and a
            // time that is equal carries the same stem. Reaching here means
            // that invariant broke, which is a bug to be seen and not a shape
            // to be accommodated.
            throw std::runtime_error(
                "Beatnik::CheckpointIO::write: checkpoint time went backwards, "
                "or repeated with a different filename. Frame '" +
                stem + "' at time " + std::to_string( time ) +
                " follows frame '" + _last_frame_stem + "' at time " +
                std::to_string( _last_frame_time ) +
                ". This is an invariant break, not a supported input: a "
                "checkpoint's time must increase, except for the exact "
                "(time, step) repeat that Solver::finalize writes." );
        }

        // 3. `writeMesh` closes the file and barriers before rank 0 writes the
        //    XDMF sidecar, but does not barrier after it. Step 4 reopens the
        //    `.h5`, so barrier here rather than relying on that.
        MPI_Barrier( _comm );

        // 4. Rank 0 appends the `/beatnik` group with the DEFAULT (serial)
        //    file-access property list. An HDF5 file is format-identical
        //    whichever driver wrote it, so a serial reopen of an
        //    MPI-IO-written file is well defined; the barriers are what keep it
        //    from racing the other ranks.
        if ( rank == 0 )
            appendBeatnikGroup( stem, header );
        MPI_Barrier( _comm );

        const std::string path = stem + ".h5";

        // 5. Relink `_latest`. A SYMLINK, not a second write -- see FILE NAMING
        //    in the file header for why a byte copy is not equivalent either.
        if ( rank == 0 )
        {
            const std::string base = baseName( stem );
            const std::string latest =
                _directory.empty() ? _prefix + "_latest"
                                   : _directory + "/" + _prefix + "_latest";
            // Relative targets, so the pair still resolves if the directory is
            // moved, and so the `.xmf` sidecar's own relative reference to its
            // `.h5` (which Tessera writes by basename) stays correct.
            IoDetail::relink( latest + ".h5", base + ".h5" );
            IoDetail::relink( latest + ".xmf", base + ".xmf" );
        }
        MPI_Barrier( _comm );

        return path;
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
    /// `<directory>/<prefix>_t<timekey>_step<step:07d>`, no extension: Tessera
    /// appends `.h5` and `.xmf` itself.
    std::string checkpointStem( Real time, long long step ) const
    {
        char step_field[24];
        std::snprintf( step_field, sizeof( step_field ), "%07lld", step );
        const std::string base =
            _prefix + "_t" + timeKey( time ) + "_step" + step_field;
        return _directory.empty() ? base : _directory + "/" + base;
    }

    /// `<directory>/<prefix>`, the stem of the master temporal collection. No
    /// frame is ever named `<prefix>.h5` -- every frame carries a `_t.._step..`
    /// field -- so this cannot collide with one, and it is the only `.xmf` in
    /// the directory with no step in its name, which is Tessera's own "the one
    /// to open" convention. Computed once, from the constructor's member-init
    /// list; `checkpointStem()` keeps its own construction.
    static std::string masterStem( const std::string& directory,
                                   const std::string& prefix )
    {
        return directory.empty() ? prefix : directory + "/" + prefix;
    }

    /// The trailing path component of a stem, for the relative symlink target.
    static std::string baseName( const std::string& path )
    {
        const std::size_t slash = path.find_last_of( '/' );
        return ( slash == std::string::npos ) ? path : path.substr( slash + 1 );
    }

    /**
     * @brief Append the `/beatnik` group to an already-written `<stem>.h5`.
     *
     * Rank 0 only, serial driver, between the barriers in `write`. Writes
     * exactly the five scalars `compare_output.py` requires
     * (`REQUIRED_FIELDS`), plus the `vertex_field_names` declaration.
     *
     * **The set is FIVE, and it is not a matter of taste.** It is fixed by the
     * gold `.npz`'s own keys and by `compare_output.py`'s `REQUIRED_FIELDS` /
     * `EXACT_SCALARS` / `FLOAT_SCALARS` tables: `state_model` (string),
     * `time` (f8), `step` (i8), `initial_volume` (f8), `initial_min_edge` (f8).
     * Writing four fails the comparison structurally on the missing one;
     * writing six is ignored but drifts the schema table above out of date.
     *
     * `step` is written as `H5T_NATIVE_INT64` rather than as one of Tessera's
     * `int`/`uint64` root attributes, because `compare_output.py` compares it
     * **exactly** against the gold's `int64` and a run can exceed `int`.
     */
    void appendBeatnikGroup( const std::string& stem,
                             const CheckpointHeader& header ) const
    {
        const std::string filename = stem + ".h5";
        const hid_t file =
            H5Fopen( filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT );
        if ( file < 0 )
            throw std::runtime_error(
                "Beatnik::CheckpointIO::write: cannot reopen '" + filename +
                "' to append the /beatnik scalar group. Tessera::writeMesh "
                "reported no error, so the file should exist." );

        const hid_t group = H5Gcreate2( file, "/beatnik", H5P_DEFAULT,
                                        H5P_DEFAULT, H5P_DEFAULT );

        IoDetail::writeString( group, "state_model",
                               toString( header.state_model ) );

        const double time = static_cast<double>( header.time );
        IoDetail::writeScalar( group, "time", H5T_NATIVE_DOUBLE, &time );

        const std::int64_t step = static_cast<std::int64_t>( header.step );
        IoDetail::writeScalar( group, "step", H5T_NATIVE_INT64, &step );

        const double volume = static_cast<double>( header.initial_volume );
        IoDetail::writeScalar( group, "initial_volume", H5T_NATIVE_DOUBLE,
                               &volume );

        const double min_edge = static_cast<double>( header.initial_min_edge );
        IoDetail::writeScalar( group, "initial_min_edge", H5T_NATIVE_DOUBLE,
                               &min_edge );

        // The slot -> meaning declaration. Taken from `SurfaceState`, whose
        // table is checked against `VertexFieldId::Count` here so adding a
        // vertex field cannot silently under-declare.
        using state_type = SurfaceState<ExecutionSpace, MemorySpace>;
        static_assert(
            VertexFieldId::Count ==
                static_cast<int>( sizeof( state_type::vertex_field_names ) /
                                  sizeof( const char* ) ),
            "SurfaceState::vertex_field_names must have one entry per "
            "VertexFieldId slot: it is what /beatnik/vertex_field_names "
            "declares and what compare_output.py cross-checks FIELD_MAP "
            "against. Adding a vertex field means updating VertexFieldId, that "
            "table, the schema table in this header, and FIELD_MAP and H5_PATH "
            "under tests/regression_tests/, in one change." );
        std::vector<const char*> names;
        names.reserve( VertexFieldId::Count );
        for ( int i = 0; i < VertexFieldId::Count; ++i )
            names.push_back( state_type::vertex_field_names[i] );
        IoDetail::writeStringArray( group, "vertex_field_names", names );

        // T4a: the same declaration for the FACE user pack, which this task
        // added and which risk R14 flagged for exactly this reason --
        // `/faces/u<N>` is as positional as `/vertices/u<N>`. R14 asked that
        // M2's mechanism be extended rather than a second one invented, so this
        // is the identical shape: an independent name table, a static_assert
        // that it covers every slot, and a cross-check in `compare_output.py`.
        using amr_type = AdaptiveMesh<ExecutionSpace, MemorySpace>;
        static_assert(
            FaceFieldId::Count ==
                static_cast<int>( sizeof( amr_type::face_field_names ) /
                                  sizeof( const char* ) ),
            "AdaptiveMesh::face_field_names must have one entry per "
            "FaceFieldId slot: it is what /beatnik/face_field_names declares. "
            "Adding a face field means updating FaceFieldId, that table, the "
            "schema table in this header, and FACE_FIELD_NAMES in "
            "tests/regression_tests/compare_output.py, in one change." );
        std::vector<const char*> face_names;
        face_names.reserve( FaceFieldId::Count );
        for ( int i = 0; i < FaceFieldId::Count; ++i )
            face_names.push_back( amr_type::face_field_names[i] );
        IoDetail::writeStringArray( group, "face_field_names", face_names );

        H5Gclose( group );
        H5Fclose( file );
    }

    MPI_Comm _comm;
    std::string _directory;
    std::string _prefix;

    /// The master temporal collection every frame is appended to. Declared
    /// after `_directory` and `_prefix` so the member-init list's
    /// `masterStem( _directory, _prefix )` reads them already initialized.
    Tessera::MeshSeries _series;

    /// The last frame appended to `_series`. **Meaningful only when
    /// `_series.numFrames() > 0`** -- before the first frame they are the
    /// zero-initialized defaults and must not be compared against.
    /// `MeshSeries` exposes `numFrames()` and `masterStem()` but neither the
    /// last time nor the last stem, so the equal-time guard needs its own copy;
    /// `_series.numFrames()` remains the only frame counter.
    double _last_frame_time = 0.0;
    std::string _last_frame_stem;
};

} // namespace Beatnik

#endif // BEATNIK_IOINTERFACE_HPP
