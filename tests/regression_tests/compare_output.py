#!/usr/bin/env python3
############################################################################
# Copyright (c) 2025 by the Beatnik authors                                #
# All rights reserved.                                                     #
#                                                                          #
# This file is part of the Beatnik library. Beatnik is distributed under a #
# BSD 3-clause license. For the licensing terms see the LICENSE file in    #
# the top-level directory.                                                 #
#                                                                          #
# SPDX-License-Identifier: BSD-3-Clause                                    #
############################################################################
"""Compare a Beatnik HDF5 checkpoint against a Python ``.npz`` gold file.

    python compare_output.py beatnik.h5 gold.npz --rtol 1e-10 --atol 1e-12

Exit code 0 on a match, nonzero on any failure.

WHY THIS IS NOT A NUMERIC DIFF
------------------------------
The two writers produce the same surface in **different row order**, and
nothing in the ``.npz`` files identifies a vertex: there are no global ids.
Worse, the Python reference is not bit-reproducible against a GPU run at all
(atomics, reduction order), so the two coordinate sets agree only to tolerance.
The comparison therefore has to *recover the correspondence* before it can
compare anything.

**Quantize, then sort.**  Each coordinate is turned into an integer by rounding
to a grid of spacing ``--match-eps``::

    key_i = round(x_i / eps)

and the vertices are sorted lexicographically on the resulting integer triple.
The obvious alternative -- sorting with a "within epsilon" comparator -- is
wrong, and quietly so: such a comparator is **not transitive** (a~b and b~c
does not give a~c), so the sort's result depends on the input order and on the
sort algorithm, which is exactly the dependence this is trying to remove.
Integer keys are transitive by construction.

Quantization has one failure mode of its own, and it is reported rather than
hidden: if two vertices fall in the **same cell**, the pairing between them is
arbitrary. That is an AMBIGUOUS match and is a failure, not a warning -- see
``--max-ambiguous`` to allow a bounded number.

**Connectivity is remapped, never compared raw.**  Face indices refer to row
numbers, which differ between the two files by exactly the permutation
recovered above. Comparing raw index triples across two independently ordered
meshes is meaningless -- it would fail on identical meshes. Instead each file's
faces are pushed through its own vertex permutation into the common sorted
ordering, each triangle is canonicalized (rotated so its smallest index is
first, preserving winding), and the face lists are sorted and compared.

Note the rotation preserves winding deliberately: ``(2,0,1)`` and ``(0,1,2)``
are the same triangle, but ``(0,2,1)`` is the same triangle with the opposite
orientation, and an inward-wound face is a real bug (it flips the sign of the
enclosed volume). Sorting the three indices, rather than rotating them, would
hide it.

STRUCTURAL MISMATCHES ARE FAILURES
----------------------------------
Different vertex or face counts, a missing required field, or a disagreeing
dtype or shape all fail immediately. The counts are still printed, because
"12480 vs 12482 vertices" localizes the bug far faster than "comparison
failed".
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import h5py
except ImportError:  # pragma: no cover - reported at run time, not import time
    h5py = None


# ===========================================================================
# NAME MAPPING -- HAND-EDITABLE
# ===========================================================================
# The `.npz` keys come from run_adaptive_mesh_bubble.py::save_state_checkpoint
# (lines 955-990); the HDF5 paths come from Beatnik_IOInterface.hpp. The two
# writers are independent, so the names will not agree, and they must NOT be
# auto-inferred: an inference that silently picks the wrong dataset reports a
# passing comparison of the wrong data.
#
# Edit this table when either writer changes a name. Nothing else in the script
# knows a field name.
#
# UPDATED BY TASK M2. The mesh and field paths below are *Tessera's* layout, not
# a Beatnik invention: `Tessera::writeMesh` owns the file and Beatnik only
# appends the `/beatnik` scalar group afterwards. Two consequences that are not
# obvious from the names:
#
#   * `/faces/verts` is uint64 and holds DENSE GLOBAL vertex indices, assigned
#     by an MPI_Exscan over owned-only counts -- the same offsets at which
#     `/vertices/position` is written. So it indexes rows of the vertex table
#     directly, exactly like the `.npz` `faces`, and needs no translation here.
#   * `/vertices/u<N>` is a POSITIONAL name. Tessera writes the N-th slot of the
#     vertex user pack under that name and records nothing about its meaning;
#     the meaning is Beatnik::VertexFieldId's declaration order. Reordering that
#     enum silently relabels every checkpoint on disk, which is precisely the
#     "passing comparison of the wrong data" this table exists to prevent -- so
#     the writer also emits /beatnik/vertex_field_names and `load_h5` VERIFIES
#     this table against it. That is a cross-check, not an inference: the table
#     below stays the source of truth and a disagreement is a hard failure.
#
#   npz key -> HDF5 dataset path
FIELD_MAP: Dict[str, str] = {
    # scalars -- written by Beatnik after Tessera closes the file
    "state_model": "/beatnik/state_model",
    "time": "/beatnik/time",
    "step": "/beatnik/step",
    "initial_volume": "/beatnik/initial_volume",
    "initial_min_edge": "/beatnik/initial_min_edge",
    # mesh -- written by Tessera
    "vertices": "/vertices/position",
    "faces": "/faces/verts",
    # per-vertex fields -- Tessera's vertex user pack, in VertexFieldId order
    "potential": "/vertices/u0",
    "sheet_vector": "/vertices/u1",
    "remesh_material_position": "/vertices/u2",
}

# Where the writer declares what each `/vertices/u<N>` slot means, as the `.npz`
# key of that slot in VertexFieldId order. Absent from a `.npz` and from a file
# written before M2; checked only when present.
VERTEX_FIELD_NAMES_PATH = "/beatnik/vertex_field_names"

# The same declaration for the FACE user pack, added at T4a with
# Beatnik::FaceFieldId. `/faces/u<N>` is positional exactly as `/vertices/u<N>`
# is (risk R14), so the writer declares the meaning of each slot and this script
# verifies it. Unlike the vertex pack there is nothing to compare the face data
# ITSELF against -- the Python `.npz` gold files carry no per-face state -- so
# no face dataset appears in FIELD_MAP and none is loaded. What is checked is
# only that the file's own declaration still says what this script expects,
# which is what turns a silent enum reordering into a named failure.
#
# Absent from a `.npz` and from any file written before T4a; checked only when
# present.
FACE_FIELD_NAMES_PATH = "/beatnik/face_field_names"
FACE_FIELD_NAMES = (
    "reference_face_area",
    "reference_face_curvature",
    "refine_mark",
)

# Present in every checkpoint regardless of state model.
REQUIRED_FIELDS = (
    "state_model",
    "time",
    "step",
    "initial_volume",
    "initial_min_edge",
    "vertices",
    "faces",
)

# Present according to `state_model`; see the schema table in
# Beatnik_IOInterface.hpp.
STATE_FIELD = {
    "potential": "potential",
    "sheet-vector": "sheet_vector",
}

# Per-vertex fields that must be permuted along with `vertices`. Optional ones
# are compared only when present in BOTH files.
PER_VERTEX_FIELDS = ("potential", "sheet_vector", "remesh_material_position")

# Scalars compared exactly rather than with rtol/atol.
EXACT_SCALARS = ("state_model", "step")

# Scalars compared with the numeric tolerances.
FLOAT_SCALARS = ("time", "initial_volume", "initial_min_edge")


# ===========================================================================
# Loading
# ===========================================================================


class LoadError(Exception):
    """A file could not be read, or is missing a required field."""


def load_npz(path: str) -> Dict[str, np.ndarray]:
    """Read the gold `.npz`, returning a plain dict of arrays."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise LoadError(f"cannot read npz {path!r}: {exc}") from exc


def check_vertex_field_names(declared: np.ndarray) -> None:
    """Verify `FIELD_MAP`'s `/vertices/u<N>` paths against the file's own map.

    Tessera names vertex user fields positionally, so `/vertices/u0` carries no
    evidence that it is the potential -- see the note above `FIELD_MAP`. The
    writer declares the meaning of each slot; this checks that `FIELD_MAP` still
    agrees with it, and raises if it does not.

    Deliberately NOT an inference. Resolving the paths *from* the declaration
    would make this script agree with whatever the writer did, including a
    silent field reordering, which is the failure being guarded against.

    :param declared: the `/beatnik/vertex_field_names` dataset: the `.npz` key
        of slot ``N`` at index ``N``.
    :raises LoadError: on any disagreement, naming both sides.
    """
    names = [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(declared).ravel().tolist()
    ]
    for slot, npz_key in enumerate(names):
        expected = f"/vertices/u{slot}"
        mapped = FIELD_MAP.get(npz_key)
        if mapped != expected:
            raise LoadError(
                f"vertex field slot {slot} is declared to be {npz_key!r}, but "
                f"FIELD_MAP sends {npz_key!r} to {mapped!r} rather than "
                f"{expected!r}. Beatnik::VertexFieldId and FIELD_MAP have "
                "drifted -- fix the table, do not widen the check."
            )


def check_face_field_names(declared: np.ndarray) -> None:
    """Verify the file's `/faces/u<N>` slot declaration against this script.

    The face analogue of `check_vertex_field_names`, and the T4a half of risk
    R14's mitigation. There is no `FIELD_MAP` entry to check against because no
    face dataset is compared, so the expected list is spelled out in
    `FACE_FIELD_NAMES` -- which makes this a check that the two independent
    statements of `Beatnik::FaceFieldId`'s order still agree.

    :param declared: the `/beatnik/face_field_names` dataset: the name of slot
        ``N`` at index ``N``.
    :raises LoadError: on any disagreement, naming both sides.
    """
    names = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(declared).ravel().tolist()
    )
    if names != FACE_FIELD_NAMES:
        raise LoadError(
            f"the file declares face user-field slots {names!r}, but this "
            f"script expects {FACE_FIELD_NAMES!r}. Beatnik::FaceFieldId and "
            "FACE_FIELD_NAMES have drifted -- fix the table, do not widen the "
            "check."
        )


def load_h5(path: str) -> Dict[str, np.ndarray]:
    """Read the Beatnik `.h5`, returning a dict keyed by **npz** names.

    Datasets are looked up through `FIELD_MAP`, so this function returns the
    same key space as `load_npz` and everything downstream is symmetric. Every
    other dataset in the file -- Tessera's gids, edges, levels, closure
    bookkeeping and root attributes -- is ignored: the comparison is against a
    Python `.npz` that has no analogue for any of it.
    """
    if h5py is None:
        raise LoadError(
            "h5py is required to read HDF5 output but is not installed"
        )
    out: Dict[str, np.ndarray] = {}
    try:
        with h5py.File(path, "r") as handle:
            if VERTEX_FIELD_NAMES_PATH in handle:
                check_vertex_field_names(handle[VERTEX_FIELD_NAMES_PATH][()])
            if FACE_FIELD_NAMES_PATH in handle:
                check_face_field_names(handle[FACE_FIELD_NAMES_PATH][()])
            for npz_key, h5_path in FIELD_MAP.items():
                if h5_path in handle:
                    out[npz_key] = np.asarray(handle[h5_path][()])
    except LoadError:
        raise
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise LoadError(f"cannot read hdf5 {path!r}: {exc}") from exc
    return out


def load_any(path: str) -> Dict[str, np.ndarray]:
    """Dispatch on the file extension."""
    lowered = path.lower()
    if lowered.endswith(".npz"):
        return load_npz(path)
    if lowered.endswith((".h5", ".hdf5")):
        return load_h5(path)
    raise LoadError(
        f"unrecognized extension for {path!r} (expected .npz, .h5 or .hdf5)"
    )


def scalar_str(value: np.ndarray) -> str:
    """Decode a 0-d string array, whether it is bytes or unicode."""
    item = np.asarray(value).reshape(()).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


# ===========================================================================
# Matching
# ===========================================================================


def quantized_lexsort(points: np.ndarray, eps: float) -> Tuple[np.ndarray, int]:
    """Return a permutation sorting `points` by their quantized integer key.

    The key is ``round(x / eps)`` per coordinate, sorted lexicographically on
    ``(kx, ky, kz)``. Integer keys make the ordering transitive and therefore
    independent of the input order -- which a "within epsilon" comparator is
    not; see the module docstring.

    :param points: ``(N, 3)`` coordinates.
    :param eps: quantization cell size, in the same units as `points`.
    :returns: ``(permutation, n_ambiguous)`` where `permutation` sorts the rows
        and `n_ambiguous` counts rows sharing a cell with their predecessor
        after sorting, i.e. pairings this scheme cannot make deterministically.
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"expected an (N,3) point array, got {points.shape}")

    # np.rint is round-half-to-even, matching the "round" in the docstring, and
    # is applied to the SCALED value so the grid is anchored at the origin --
    # not at the data's own minimum, which would make the key depend on the
    # data set and defeat the purpose.
    keys = np.rint(points / eps).astype(np.int64)

    # lexsort's last key is primary, so reverse to get (kx, ky, kz) ordering.
    order = np.lexsort((keys[:, 2], keys[:, 1], keys[:, 0]))

    sorted_keys = keys[order]
    if sorted_keys.shape[0] > 1:
        same_cell = np.all(sorted_keys[1:] == sorted_keys[:-1], axis=1)
        n_ambiguous = int(np.count_nonzero(same_cell))
    else:
        n_ambiguous = 0

    return order, n_ambiguous


def canonical_faces(faces: np.ndarray, vertex_perm: np.ndarray) -> np.ndarray:
    """Remap faces through a vertex permutation, canonicalize, and sort.

    :param faces: ``(F, 3)`` connectivity in the file's own vertex numbering.
    :param vertex_perm: the permutation returned by `quantized_lexsort`, i.e.
        ``sorted_row_i = original_row_{perm[i]}``. The INVERSE is what maps an
        original vertex index to its sorted position, and that is what
        connectivity needs.
    :returns: ``(F, 3)`` faces in the sorted vertex numbering, each rotated so
        its smallest index comes first (winding preserved), and the rows sorted
        lexicographically.
    """
    inverse = np.empty_like(vertex_perm)
    inverse[vertex_perm] = np.arange(vertex_perm.shape[0])

    remapped = inverse[np.asarray(faces, dtype=np.int64)]

    # Rotate each triangle so its smallest index is first. A rotation preserves
    # winding; sorting the triple would not, and would hide an inverted face.
    argmin = np.argmin(remapped, axis=1)
    idx = (argmin[:, None] + np.arange(3)[None, :]) % 3
    rotated = np.take_along_axis(remapped, idx, axis=1)

    order = np.lexsort((rotated[:, 2], rotated[:, 1], rotated[:, 0]))
    return rotated[order]


# ===========================================================================
# Reporting
# ===========================================================================


class Report:
    """Accumulates failures and prints a diagnosable summary."""

    def __init__(self) -> None:
        self.failures: list = []
        self.lines: list = []

    def info(self, message: str) -> None:
        self.lines.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)
        self.lines.append("FAIL: " + message)

    @property
    def ok(self) -> bool:
        return not self.failures

    def emit(self, stream=sys.stdout) -> None:
        for line in self.lines:
            print(line, file=stream)
        if self.ok:
            print("\nRESULT: match", file=stream)
        else:
            print(
                f"\nRESULT: {len(self.failures)} failure(s)", file=stream
            )
            for failure in self.failures:
                print("  - " + failure, file=stream)


def compare_array(
    report: Report,
    name: str,
    lhs: np.ndarray,
    rhs: np.ndarray,
    rtol: float,
    atol: float,
    worst_n: int,
) -> None:
    """Compare two aligned float arrays and report the error norms.

    Reports, per field: max absolute error, max relative error, the L2 norm of
    the difference, and the indices of the worst offenders. All four are
    printed whether or not the field passes, because a field that passes at
    1e-10 but is 100x worse than its neighbours is the interesting one.
    """
    lhs = np.asarray(lhs, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    if lhs.shape != rhs.shape:
        report.fail(
            f"{name}: shape mismatch, cpp {lhs.shape} vs gold {rhs.shape}"
        )
        return

    diff = np.abs(lhs - rhs)
    denom = np.maximum(np.abs(rhs), np.finfo(float).tiny)
    rel = diff / denom

    max_abs = float(diff.max()) if diff.size else 0.0
    max_rel = float(rel.max()) if rel.size else 0.0
    l2 = float(np.sqrt(np.sum(diff * diff)))

    tolerated = diff <= (atol + rtol * np.abs(rhs))
    n_bad = int(np.count_nonzero(~tolerated))

    report.info(
        f"  {name:<26} n={lhs.size:<9} max|e|={max_abs:.6e}  "
        f"max|e|/|g|={max_rel:.6e}  L2={l2:.6e}  outside tol={n_bad}"
    )

    if n_bad:
        flat = np.argsort(diff.ravel())[::-1][:worst_n]
        for offset in flat:
            index = np.unravel_index(offset, diff.shape)
            report.info(
                f"      worst {tuple(int(i) for i in index)}: "
                f"cpp={lhs[index]:+.17e} gold={rhs[index]:+.17e} "
                f"|e|={diff[index]:.6e}"
            )
        report.fail(
            f"{name}: {n_bad} of {lhs.size} entries outside "
            f"rtol={rtol:g} atol={atol:g} (max|e|={max_abs:.6e})"
        )


def compare_scalar_float(
    report: Report, name: str, lhs, rhs, rtol: float, atol: float
) -> None:
    lhs_v = float(np.asarray(lhs).reshape(()).item())
    rhs_v = float(np.asarray(rhs).reshape(()).item())
    err = abs(lhs_v - rhs_v)
    report.info(
        f"  {name:<26} cpp={lhs_v:+.17e} gold={rhs_v:+.17e} |e|={err:.6e}"
    )
    if err > atol + rtol * abs(rhs_v):
        report.fail(
            f"{name}: {lhs_v!r} vs {rhs_v!r} outside "
            f"rtol={rtol:g} atol={atol:g}"
        )


# ===========================================================================
# The comparison
# ===========================================================================


def compare(
    cpp: Dict[str, np.ndarray],
    gold: Dict[str, np.ndarray],
    *,
    rtol: float,
    atol: float,
    match_eps: float,
    max_ambiguous: int,
    worst_n: int,
) -> Report:
    """Compare a loaded Beatnik checkpoint against a loaded gold file."""
    report = Report()

    # --- required fields present in both -----------------------------------
    for key in REQUIRED_FIELDS:
        if key not in cpp:
            report.fail(f"cpp file is missing required field {key!r}")
        if key not in gold:
            report.fail(f"gold file is missing required field {key!r}")
    if not report.ok:
        return report

    # --- state model, which decides which solution field to expect ---------
    cpp_model = scalar_str(cpp["state_model"])
    gold_model = scalar_str(gold["state_model"])
    report.info(f"state_model: cpp={cpp_model!r} gold={gold_model!r}")
    if cpp_model != gold_model:
        report.fail(
            f"state_model: {cpp_model!r} vs {gold_model!r}"
        )
        return report
    if cpp_model not in STATE_FIELD:
        report.fail(f"unrecognized state_model {cpp_model!r}")
        return report

    solution_field = STATE_FIELD[cpp_model]
    for source, name in ((cpp, "cpp"), (gold, "gold")):
        if solution_field not in source:
            report.fail(
                f"{name} file declares state_model={cpp_model!r} but has no "
                f"{solution_field!r} field"
            )
    if not report.ok:
        return report

    # --- structure ---------------------------------------------------------
    cpp_vertices = np.asarray(cpp["vertices"], dtype=float)
    gold_vertices = np.asarray(gold["vertices"], dtype=float)
    cpp_faces = np.asarray(cpp["faces"])
    gold_faces = np.asarray(gold["faces"])

    report.info(
        f"structure: vertices cpp={cpp_vertices.shape} "
        f"gold={gold_vertices.shape}; faces cpp={cpp_faces.shape} "
        f"gold={gold_faces.shape}"
    )

    for name, arr in (("cpp vertices", cpp_vertices),
                      ("gold vertices", gold_vertices)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            report.fail(f"{name}: expected (N,3), got {arr.shape}")
    for name, arr in (("cpp faces", cpp_faces), ("gold faces", gold_faces)):
        if arr.ndim != 2 or arr.shape[1] != 3:
            report.fail(f"{name}: expected (F,3), got {arr.shape}")
        elif not np.issubdtype(arr.dtype, np.integer):
            report.fail(f"{name}: expected an integer dtype, got {arr.dtype}")
    if not report.ok:
        return report

    if cpp_vertices.shape[0] != gold_vertices.shape[0]:
        report.fail(
            f"vertex count: cpp={cpp_vertices.shape[0]} "
            f"gold={gold_vertices.shape[0]}"
        )
    if cpp_faces.shape[0] != gold_faces.shape[0]:
        report.fail(
            f"face count: cpp={cpp_faces.shape[0]} "
            f"gold={gold_faces.shape[0]}"
        )
    if not report.ok:
        return report

    # --- scalars -----------------------------------------------------------
    report.info("scalars:")
    for key in EXACT_SCALARS:
        if key == "state_model":
            continue  # already compared
        cpp_v = np.asarray(cpp[key]).reshape(()).item()
        gold_v = np.asarray(gold[key]).reshape(()).item()
        report.info(f"  {key:<26} cpp={cpp_v} gold={gold_v}")
        if int(cpp_v) != int(gold_v):
            report.fail(f"{key}: {cpp_v} vs {gold_v} (compared exactly)")
    for key in FLOAT_SCALARS:
        compare_scalar_float(report, key, cpp[key], gold[key], rtol, atol)

    # --- recover the vertex correspondence ---------------------------------
    cpp_perm, cpp_ambiguous = quantized_lexsort(cpp_vertices, match_eps)
    gold_perm, gold_ambiguous = quantized_lexsort(gold_vertices, match_eps)

    n_vertices = cpp_vertices.shape[0]
    matched = n_vertices - max(cpp_ambiguous, gold_ambiguous)
    report.info(
        f"matching (eps={match_eps:g}): {matched}/{n_vertices} unambiguous, "
        f"ambiguous cpp={cpp_ambiguous} gold={gold_ambiguous}"
    )
    if max(cpp_ambiguous, gold_ambiguous) > max_ambiguous:
        report.fail(
            f"ambiguous vertex pairing: {cpp_ambiguous} cpp and "
            f"{gold_ambiguous} gold vertices share a quantization cell at "
            f"eps={match_eps:g} (limit {max_ambiguous}). Lower --match-eps if "
            "the mesh is genuinely finer than the cell; raise it if the two "
            "meshes agree only loosely."
        )
        # Not returning: the field comparison below is still informative, and
        # an ambiguous pairing usually shows up as a handful of large errors
        # rather than as noise, which is worth seeing.

    # --- per-vertex fields, through the recovered permutation --------------
    report.info("per-vertex fields (sorted by quantized position):")
    compare_array(
        report,
        "vertices",
        cpp_vertices[cpp_perm],
        gold_vertices[gold_perm],
        rtol,
        atol,
        worst_n,
    )

    # The state field the model does NOT select. The Python writes one or the
    # other and never both; Beatnik always writes both, because they are two
    # slots of one Cabana tuple and Tessera::writeMesh writes the whole vertex
    # user pack unconditionally (see Beatnik_IOInterface.hpp, "BOTH STATE FIELDS
    # ARE ALWAYS PRESENT"). Under `potential`, `sheet_vector` holds whatever
    # updateSheetVector last cached there -- it is not an evolved unknown and
    # has no gold counterpart, so it is skipped rather than reported as a
    # one-sided presence. This is the ONLY field allowed to be one-sided.
    inactive_state_field = next(
        name for name in STATE_FIELD.values() if name != solution_field
    )

    for key in PER_VERTEX_FIELDS:
        in_cpp = key in cpp
        in_gold = key in gold
        if not in_cpp and not in_gold:
            continue
        if key == inactive_state_field and in_cpp != in_gold:
            report.info(
                f"  {key:<26} skipped: not the active state field for "
                f"state_model={cpp_model!r}"
            )
            continue
        if in_cpp != in_gold:
            # The solution field is required (checked above); the material
            # position is optional in the schema, so a one-sided presence there
            # is a genuine mismatch worth reporting rather than skipping.
            report.fail(
                f"{key}: present in "
                f"{'cpp' if in_cpp else 'gold'} only"
            )
            continue
        compare_array(
            report,
            key,
            np.asarray(cpp[key])[cpp_perm],
            np.asarray(gold[key])[gold_perm],
            rtol,
            atol,
            worst_n,
        )

    # --- connectivity, remapped through the same permutation ---------------
    cpp_canon = canonical_faces(cpp_faces, cpp_perm)
    gold_canon = canonical_faces(gold_faces, gold_perm)
    n_face_diff = int(np.count_nonzero(np.any(cpp_canon != gold_canon, axis=1)))
    report.info(
        f"faces: {cpp_canon.shape[0] - n_face_diff}/{cpp_canon.shape[0]} "
        "triangles identical after remap + canonicalize + sort"
    )
    if n_face_diff:
        bad = np.flatnonzero(np.any(cpp_canon != gold_canon, axis=1))[:worst_n]
        for i in bad:
            report.info(
                f"      row {int(i)}: cpp={tuple(int(v) for v in cpp_canon[i])} "
                f"gold={tuple(int(v) for v in gold_canon[i])}"
            )
        report.fail(
            f"faces: {n_face_diff} of {cpp_canon.shape[0]} triangles differ "
            "after remapping through the vertex permutation"
        )

    return report


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a Beatnik HDF5 checkpoint against a zmodel3d-amr .npz "
            "gold file. Row order is recovered by quantized lexicographic "
            "sorting of the vertex coordinates."
        )
    )
    parser.add_argument("cpp", help="Beatnik output (.h5) -- or a .npz")
    parser.add_argument("gold", help="Python gold file (.npz) -- or a .h5")
    parser.add_argument(
        "--rtol",
        type=float,
        default=1.0e-10,
        help="relative tolerance for float comparisons (default 1e-10)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1.0e-12,
        help="absolute tolerance for float comparisons (default 1e-12)",
    )
    parser.add_argument(
        "--match-eps",
        type=float,
        default=1.0e-9,
        help=(
            "quantization cell size used to recover the vertex ordering "
            "(default 1e-9). Must be much larger than the coordinate "
            "disagreement between the two files and much smaller than the "
            "smallest edge length."
        ),
    )
    parser.add_argument(
        "--max-ambiguous",
        type=int,
        default=0,
        help=(
            "tolerate this many vertices sharing a quantization cell "
            "(default 0). Any ambiguity means some pairing is arbitrary."
        ),
    )
    parser.add_argument(
        "--worst",
        type=int,
        default=5,
        help="how many worst offenders to print per failing field (default 5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="print only the final RESULT line and any failures",
    )
    return parser


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        cpp = load_any(args.cpp)
        gold = load_any(args.gold)
    except LoadError as exc:
        print(f"compare_output: {exc}", file=sys.stderr)
        return 2

    report = compare(
        cpp,
        gold,
        rtol=args.rtol,
        atol=args.atol,
        match_eps=args.match_eps,
        max_ambiguous=args.max_ambiguous,
        worst_n=args.worst,
    )

    if args.quiet:
        trimmed = Report()
        trimmed.failures = report.failures
        trimmed.emit()
    else:
        report.emit()

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
