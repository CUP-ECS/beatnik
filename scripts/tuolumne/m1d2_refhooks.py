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
"""M1-D2 (`tasks/milestone1.md`): instrumentation of the Python reference.

Two subcommands, both run by `scripts/tuolumne/m1d2_split_ab.flux`:

``run REF PARAMS_JSON -- <reference argv>``
    Runs `examples/run_adaptive_mesh_bubble.py`'s ``main()`` from the
    reference checkout ``REF`` with ``<reference argv>``, after wrapping (never
    editing) five reference functions so that every step prints one
    ``[m1d2ref] step=`` line: the step's ``dt``, the split and collapse
    candidate counts *before* the per-pass cap, the accepted splits,
    collapses, quality flips and smooth steps, and the number of edges the
    isotropic cleanup's valence flips changed. The wrappers read their
    arguments and return the wrapped function's result unchanged. The
    ``DynamicRemeshParams`` and the cleanup keyword arguments the run used are
    written to ``PARAMS_JSON`` for ``probe``.

``probe REF PARAMS_JSON LABEL=FILE ...``
    Applies one reference remesh step -- ``dynamic_remesh_arrays`` followed by
    ``isotropic_cleanup``, with the run's exact parameters -- to each
    checkpoint ``FILE`` (a reference ``.npz`` or a Beatnik ``.h5``) and prints
    one ``[m1d2probe]`` line with the same counts. The edit decisions depend
    only on vertex positions and faces, so no field is carried.

The reference repository is read, never modified.
"""

from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np

COUNT_KEYS = (
    "split_candidates",
    "splits",
    "collapse_candidates",
    "collapses",
    "flips",
    "smooth_steps",
    "cleanup_edges_changed",
)


def _edge_lengths_and_targets(dr, vertices, faces, target, params):
    edges = np.asarray(sorted(dr.unique_edges(faces)), dtype=int)
    if edges.size == 0:
        return np.zeros(0), np.zeros(0)
    length = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    local = np.minimum(target[edges[:, 0]], target[edges[:, 1]])
    return length, np.maximum(local, params.h_min)


def _edge_set(faces):
    f = np.asarray(faces, dtype=int)
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return set(map(tuple, e.tolist()))


def install_hooks(dr, mq, rec):
    """Wrap the reference's pass functions; each adds to the dict ``rec``."""

    orig_split = dr.split_long_edges
    orig_collapse = dr.collapse_short_edges
    orig_flip = dr.flip_edges_for_quality
    orig_smooth = dr.tangential_smooth_vertices
    orig_valence = mq._valence_equalizing_flips

    def split_long_edges(vertices, faces, fields, target, params):
        length, local = _edge_lengths_and_targets(dr, vertices, faces, target, params)
        rec["split_candidates"] += int(np.count_nonzero(length > params.split_factor * local))
        out = orig_split(vertices, faces, fields, target, params)
        rec["splits"] += int(out[3])
        return out

    def collapse_short_edges(vertices, faces, fields, target, params):
        length, local = _edge_lengths_and_targets(dr, vertices, faces, target, params)
        rec["collapse_candidates"] += int(
            np.count_nonzero(length < params.collapse_factor * local)
        )
        out = orig_collapse(vertices, faces, fields, target, params)
        rec["collapses"] += int(out[3])
        return out

    def flip_edges_for_quality(vertices, faces, params):
        out = orig_flip(vertices, faces, params)
        rec["flips"] += int(out[2])
        return out

    def tangential_smooth_vertices(vertices, faces, params):
        out = orig_smooth(vertices, faces, params)
        rec["smooth_steps"] += int(out[1])
        return out

    def _valence_equalizing_flips(vertices, faces, *args, **kwargs):
        out = orig_valence(vertices, faces, *args, **kwargs)
        rec["cleanup_edges_changed"] += len(_edge_set(faces) - _edge_set(out))
        return out

    dr.split_long_edges = split_long_edges
    dr.collapse_short_edges = collapse_short_edges
    dr.flip_edges_for_quality = flip_edges_for_quality
    dr.tangential_smooth_vertices = tangential_smooth_vertices
    mq._valence_equalizing_flips = _valence_equalizing_flips


def _import_reference(ref):
    sys.path.insert(0, ref)
    sys.path.insert(0, ref + "/examples")
    import zmodel3d.dynamic_remesh as dr
    import zmodel3d.mesh_quality as mq

    return dr, mq


def _fmt(rec):
    return " ".join(f"{k}={rec[k]}" for k in COUNT_KEYS)


def cmd_run(ref, params_json, ref_argv):
    dr, mq = _import_reference(ref)
    rec = dict.fromkeys(COUNT_KEYS, 0)
    install_hooks(dr, mq, rec)
    import run_adaptive_mesh_bubble as ex

    captured = {}
    pending = {"dt": None}

    orig_dt = ex.choose_step_dt

    def choose_step_dt(*a, **k):
        dt = orig_dt(*a, **k)
        pending["dt"] = float(dt)
        return dt

    orig_arrays = ex.dynamic_remesh_arrays

    def dynamic_remesh_arrays(*a, **k):
        if "params" not in captured:
            captured["params"] = dataclasses.asdict(k["params"])
        return orig_arrays(*a, **k)

    orig_cleanup = mq.isotropic_cleanup

    def isotropic_cleanup(vertices, faces, **k):
        if "cleanup" not in captured:
            captured["cleanup"] = dict(k)
        return orig_cleanup(vertices, faces, **k)

    orig_save = ex.save_state_checkpoint

    def save_state_checkpoint(directory, state, **k):
        step = int(k["step"])
        # The run's closing "final finite checkpoint" repeats the last step;
        # its counts were already printed.
        if step > 0 and step != pending.get("printed"):
            pending["printed"] = step
            nv = int(np.asarray(state.vertices).shape[0])
            nf = int(np.asarray(state.faces).shape[0])
            print(
                f"[m1d2ref] step={step} time={float(k['current_time']):.17g} "
                f"dt={pending['dt']:.17g} V={nv} F={nf} {_fmt(rec)}",
                flush=True,
            )
        for key in COUNT_KEYS:
            rec[key] = 0
        return orig_save(directory, state, **k)

    ex.choose_step_dt = choose_step_dt
    ex.dynamic_remesh_arrays = dynamic_remesh_arrays
    mq.isotropic_cleanup = isotropic_cleanup
    ex.save_state_checkpoint = save_state_checkpoint

    sys.argv = [ex.__file__] + list(ref_argv)
    print(f"[m1d2ref] reference argv: {' '.join(sys.argv[1:])}", flush=True)
    ex.main()

    with open(params_json, "w") as fh:
        json.dump(captured, fh, indent=1, sort_keys=True)
    print(f"[m1d2ref] params: {json.dumps(captured, sort_keys=True)}", flush=True)


def _load_state(path):
    if path.endswith(".npz"):
        d = np.load(path)
        return int(d["step"]), float(d["time"]), d["vertices"], d["faces"].astype(int)
    import h5py

    with h5py.File(path, "r") as h:
        return (
            int(h["beatnik/step"][()]),
            float(h["beatnik/time"][()]),
            h["vertices/position"][:],
            h["faces/verts"][:].astype(int),
        )


def cmd_probe(ref, params_json, items):
    dr, mq = _import_reference(ref)
    rec = dict.fromkeys(COUNT_KEYS, 0)
    install_hooks(dr, mq, rec)
    with open(params_json) as fh:
        captured = json.load(fh)
    params = dr.DynamicRemeshParams(**captured["params"])
    cleanup = captured["cleanup"]
    for item in items:
        label, path = item.split("=", 1)
        step, time, vertices, faces = _load_state(path)
        for key in COUNT_KEYS:
            rec[key] = 0
        q_before = dr._finite_min(dr.triangle_quality(vertices, faces))
        v1, f1, _, _ = dr.dynamic_remesh_arrays(vertices, faces, {}, params=params)
        v2, f2 = mq.isotropic_cleanup(v1, f1, **cleanup)
        print(
            f"[m1d2probe] label={label} step={step} time={time:.17g} "
            f"V={vertices.shape[0]} F={faces.shape[0]} minQ={q_before:.6f} "
            f"{_fmt(rec)} V_after={v2.shape[0]} F_after={f2.shape[0]}",
            flush=True,
        )


def main():
    if len(sys.argv) >= 4 and sys.argv[1] == "run":
        argv = sys.argv[4:]
        if argv and argv[0] == "--":
            argv = argv[1:]
        cmd_run(sys.argv[2], sys.argv[3], argv)
    elif len(sys.argv) >= 5 and sys.argv[1] == "probe":
        cmd_probe(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
