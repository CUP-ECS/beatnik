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
"""Regenerate the committed synthetic fixtures for ``compare_output.py``.

    python make_fixtures.py [--outdir fixtures]

The fixtures are **committed**, not generated during the build: a comparator
tested only against data produced by the same run of the same script is not
tested at all, and a fixture that regenerates on every build can drift with the
generator without anyone noticing. Run this only to deliberately change what
the fixtures contain, and commit the result in the same change.

WHAT IS GENERATED
-----------------
``synthetic_gold.npz``
    A small closed icosahedron-derived surface with a full potential-model
    checkpoint payload, in the schema of
    ``run_adaptive_mesh_bubble.py::save_state_checkpoint`` (lines 955-990).

``synthetic_match.h5``
    **The same data, in a different row order**, laid out the way a real
    Beatnik checkpoint is (see ``as_beatnik_h5``: Tessera's dataset paths, a
    ``uint64`` face table, the always-present inactive ``sheet_vector`` slot,
    and the ``/beatnik/vertex_field_names`` declaration). Vertices are permuted
    by a fixed pseudo-random permutation and the connectivity is renumbered to
    match, so the surface is identical and only the storage order differs.
    This is the positive fixture: it exercises exactly the thing
    ``compare_output.py`` exists to do -- recover a correspondence that the
    files themselves do not record -- and it must PASS.

``synthetic_perturbed.h5``
    The same permuted data with **one vertex displaced** by a distance far
    above the comparison tolerance but far below the smallest edge, so the
    surface is still valid and the quantized matching still pairs every vertex
    unambiguously. This is the negative fixture: the failure has to come from
    the value comparison, not from a structural check or a matching failure,
    or it would not be testing the numeric path. It must FAIL.

Everything is deterministic: a fixed seed, and no dependence on platform
floating point beyond IEEE arithmetic on the literal inputs.
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np

# Fixed seed. The permutation and the field values must be reproducible, or
# regenerating the fixtures produces a spurious diff.
SEED = 20260806

# Where the perturbation goes and how big it is. 1e-6 is:
#   - 1e4 x the default --atol (1e-12) and well above --rtol * |x|, so the
#     comparison must notice it;
#   - 1e3 x the default --match-eps (1e-9), so the perturbed vertex lands in a
#     different quantization cell and the two files sort DIFFERENTLY -- which
#     is the realistic failure, and is what makes this a test of the numeric
#     path rather than of a shape check;
#   - ~1e-6 of the smallest edge length, so the surface remains valid.
PERTURB_INDEX = 7
PERTURB_DELTA = np.array([1.0e-6, 0.0, 0.0])

# The npz key -> HDF5 path mapping, duplicated from compare_output.py's
# FIELD_MAP. Deliberately duplicated rather than imported: if the two ever
# disagree, the fixture test fails, which is the point. An import would make
# the test pass no matter what the mapping said.
#
# UPDATED BY TASK M2 to Tessera's real layout -- `Tessera::writeMesh` owns the
# file and Beatnik appends only the `/beatnik` group. See the note above
# compare_output.py's FIELD_MAP for why `/vertices/u<N>` is a positional name
# and what guards it.
H5_PATH = {
    "state_model": "/beatnik/state_model",
    "time": "/beatnik/time",
    "step": "/beatnik/step",
    "initial_volume": "/beatnik/initial_volume",
    "initial_min_edge": "/beatnik/initial_min_edge",
    "vertices": "/vertices/position",
    "faces": "/faces/verts",
    "potential": "/vertices/u0",
    "sheet_vector": "/vertices/u1",
    "remesh_material_position": "/vertices/u2",
}

# Beatnik::VertexFieldId order, as the writer declares it in
# /beatnik/vertex_field_names. compare_output.py checks H5_PATH's `u<N>` paths
# against this, so a fixture carrying it exercises that check.
VERTEX_FIELD_NAMES = ("potential", "sheet_vector", "remesh_material_position")


def icosahedron(radius: float, center: np.ndarray):
    """The 12-vertex, 20-face regular icosahedron, outward oriented.

    The same construction as ``mesh.py::icosphere_mesh`` at subdivision 0
    (lines 372-415), so the fixture has the shape of a real Beatnik surface --
    closed, manifold, consistently wound -- rather than an arbitrary point
    cloud. Small enough to read in a hex dump if something goes wrong.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1.0, phi, 0.0),
            (1.0, phi, 0.0),
            (-1.0, -phi, 0.0),
            (1.0, -phi, 0.0),
            (0.0, -1.0, phi),
            (0.0, 1.0, phi),
            (0.0, -1.0, -phi),
            (0.0, 1.0, -phi),
            (phi, 0.0, -1.0),
            (phi, 0.0, 1.0),
            (-phi, 0.0, -1.0),
            (-phi, 0.0, 1.0),
        ],
        dtype=float,
    )
    vertices /= np.linalg.norm(vertices, axis=1)[:, None]
    faces = np.asarray(
        [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ],
        dtype=np.int64,
    )
    vertices = center[None, :] + radius * vertices

    # Orient every face outward, as icosphere_mesh does (lines 448-456).
    oriented = []
    for a, b, c in faces:
        normal = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
        if float(np.dot(normal, (vertices[a] + vertices[b] + vertices[c]) / 3.0
                        - center)) < 0.0:
            oriented.append((a, c, b))
        else:
            oriented.append((a, b, c))
    return vertices, np.asarray(oriented, dtype=np.int64)


def enclosed_volume(vertices, faces):
    """Port of run_adaptive_mesh_bubble.py::mesh_enclosed_volume (1036-1040)."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def min_edge_length(vertices, faces):
    """Port of run_adaptive_mesh_bubble.py::mesh_edge_lengths (545-555)."""
    edges = set()
    for i, j, k in faces:
        for a, b in ((i, j), (j, k), (k, i)):
            edges.add((int(a), int(b)) if a < b else (int(b), int(a)))
    arr = np.asarray(sorted(edges), dtype=int)
    return float(np.min(np.linalg.norm(vertices[arr[:, 1]] - vertices[arr[:, 0]],
                                       axis=1)))


def build_payload():
    """The reference payload, in the gold-file schema."""
    rng = np.random.default_rng(SEED)
    center = np.asarray([0.0, 0.0, 0.25])
    vertices, faces = icosahedron(0.25, center)

    # A smooth, non-symmetric potential so a mis-pairing of two vertices shows
    # up as a large error rather than cancelling. Symmetric data would let a
    # broken permutation pass.
    rel = vertices - center[None, :]
    potential = (
        0.37 * rel[:, 0]
        + 0.11 * rel[:, 1] * rel[:, 2]
        - 0.53 * rel[:, 2] ** 2
    )
    potential -= potential.mean()

    # The material position at t=0 is the vertex position
    # (run_adaptive_mesh_bubble.py:1227); nudge it slightly so the comparator is
    # exercised on a field that is NOT identical to `vertices`, which would let
    # a bug that compares the wrong array slip through.
    material = vertices + 1.0e-3 * rng.standard_normal(vertices.shape)

    return {
        "state_model": np.asarray("potential"),
        "time": np.asarray(0.123456),
        "step": np.asarray(7),
        "initial_volume": np.asarray(enclosed_volume(vertices, faces)),
        "initial_min_edge": np.asarray(min_edge_length(vertices, faces)),
        "vertices": vertices,
        "faces": faces,
        "potential": potential,
        "remesh_material_position": material,
    }


def permute(payload, perm):
    """Reorder the vertices and renumber the connectivity to match.

    This is the whole point of the positive fixture: the surface is unchanged,
    only its storage order differs, exactly as it will when Beatnik's
    distributed gather orders vertices differently from the serial Python.
    """
    inverse = np.empty_like(perm)
    inverse[perm] = np.arange(perm.shape[0])

    out = dict(payload)
    out["vertices"] = payload["vertices"][perm]
    out["potential"] = payload["potential"][perm]
    out["remesh_material_position"] = payload["remesh_material_position"][perm]
    out["faces"] = inverse[payload["faces"]]

    # Also shuffle the face rows, since nothing constrains their order either.
    rng = np.random.default_rng(SEED + 1)
    face_perm = rng.permutation(out["faces"].shape[0])
    out["faces"] = out["faces"][face_perm]
    return out


def as_beatnik_h5(payload):
    """Shape a gold payload the way a real Beatnik checkpoint is shaped.

    Two M2 differences from the `.npz`, both of which the comparator has
    specific handling for, so a fixture that omitted them would leave that
    handling untested:

    ``sheet_vector`` is always present.
        It is `/vertices/u1`, a slot in the same Cabana tuple as the potential,
        and `Tessera::writeMesh` writes the whole vertex user pack. Under the
        `potential` state model it is a stale cache with no gold counterpart,
        and `compare_output.py` must skip it rather than fail on the one-sided
        presence. Filled with values that are *wrong on purpose* -- if the
        comparator ever starts comparing it, the fixture must fail loudly rather
        than pass by luck.

    ``faces`` is uint64.
        Tessera writes dense global vertex indices as `H5T_NATIVE_UINT64`.
    """
    out = dict(payload)
    out["faces"] = np.asarray(payload["faces"], dtype=np.uint64)
    out["sheet_vector"] = np.full(
        (payload["vertices"].shape[0], 3), -7.0, dtype=float
    )
    return out


def write_h5(path, payload):
    """Write the payload at the HDF5 paths compare_output.py expects."""
    string_dt = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        for key, value in payload.items():
            arr = np.asarray(value)
            dataset_path = H5_PATH[key]
            if arr.dtype.kind in ("U", "S", "O"):
                handle.create_dataset(
                    dataset_path, data=str(arr), dtype=string_dt
                )
            else:
                handle.create_dataset(dataset_path, data=arr)
        # The slot -> meaning declaration compare_output.py cross-checks
        # H5_PATH against. See VERTEX_FIELD_NAMES.
        handle.create_dataset(
            "/beatnik/vertex_field_names",
            data=list(VERTEX_FIELD_NAMES),
            dtype=string_dt,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fixtures"),
        help="directory to write the fixtures into",
    )
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    payload = build_payload()

    gold_path = os.path.join(args.outdir, "synthetic_gold.npz")
    np.savez_compressed(gold_path, **payload)

    rng = np.random.default_rng(SEED + 2)
    perm = rng.permutation(payload["vertices"].shape[0])
    permuted = permute(payload, perm)

    match_path = os.path.join(args.outdir, "synthetic_match.h5")
    write_h5(match_path, as_beatnik_h5(permuted))

    perturbed = dict(permuted)
    perturbed["vertices"] = permuted["vertices"].copy()
    perturbed["vertices"][PERTURB_INDEX] += PERTURB_DELTA
    bad_path = os.path.join(args.outdir, "synthetic_perturbed.h5")
    write_h5(bad_path, as_beatnik_h5(perturbed))

    print(f"wrote {gold_path}")
    print(f"wrote {match_path}   (same surface, permuted -- must PASS)")
    print(f"wrote {bad_path}     (one vertex moved  -- must FAIL)")
    print(f"vertex permutation: {perm.tolist()}")


if __name__ == "__main__":
    main()
