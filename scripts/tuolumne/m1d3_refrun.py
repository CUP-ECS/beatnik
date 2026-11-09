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
"""M1-D3 (`tasks/milestone1.md`): the Python reference run with one variable
changed, to separate the causes of the milestone-1 late drift.

``convert H5 NPZ``
    Writes a Beatnik ``.h5`` checkpoint as a reference restart ``.npz``: the
    nine keys the reference's ``save_state_checkpoint`` writes, with
    ``potential`` and ``remesh_material_position`` taken from the vertex
    fields named ``potential`` and ``remesh_material_position`` in
    ``beatnik/vertex_field_names``. Pure file I/O.

``run REF [--dt-from LAUNCH_LOG] -- <reference argv>``
    Runs ``examples/run_adaptive_mesh_bubble.py``'s ``main()`` from the
    reference tree ``REF`` with ``<reference argv>``. With ``--dt-from``,
    ``choose_step_dt`` is replaced so that the run's n-th step uses Beatnik's
    n-th step size, read from the ``[m1traj] step= time=`` lines of a Beatnik
    ``launch.log``; nothing else changes. Without it the reference is
    unchanged. Every checkpoint save prints one ``[m1d3ref]`` line with the
    step, time, last ``dt``, V and F.

The reference tree is read, never modified.
"""

import re
import sys

import numpy as np


def cmd_convert(h5_path, npz_path):
    import h5py

    with h5py.File(h5_path, "r") as h:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in h["beatnik/vertex_field_names"][()]]
        field = {n: h["vertices/u%d" % i][:] for i, n in enumerate(names)}
        payload = {
            "state_model": np.asarray("potential"),
            "time": np.asarray(float(h["beatnik/time"][()])),
            "step": np.asarray(int(h["beatnik/step"][()])),
            "initial_volume": np.asarray(float(h["beatnik/initial_volume"][()])),
            "initial_min_edge": np.asarray(
                float(h["beatnik/initial_min_edge"][()])),
            "vertices": np.asarray(h["vertices/position"][:], dtype=float),
            "faces": np.asarray(h["faces/verts"][:], dtype=int),
            "potential": np.asarray(field["potential"], dtype=float),
            "remesh_material_position": np.asarray(
                field["remesh_material_position"], dtype=float),
        }
    if payload["potential"].shape != (payload["vertices"].shape[0],):
        sys.exit("convert: potential shape %s" % (payload["potential"].shape,))
    np.savez_compressed(npz_path, **payload)
    print("[m1d3conv] %s -> %s step=%d time=%.17g V=%d F=%d" % (
        h5_path, npz_path, int(payload["step"]), float(payload["time"]),
        payload["vertices"].shape[0], payload["faces"].shape[0]), flush=True)


def beatnik_dts(launch_log):
    t = {0: 0.0}
    with open(launch_log) as fh:
        for line in fh:
            m = re.match(r"\[m1traj\] step=(\d+) time=(\S+)", line)
            if m:
                t[int(m.group(1))] = float(m.group(2))
    steps = sorted(t)
    if steps != list(range(len(steps))):
        sys.exit("run: %s does not hold steps 0..N contiguously" % launch_log)
    return np.diff(np.array([t[s] for s in steps]))


def cmd_run(ref, dt_from, ref_argv):
    sys.path.insert(0, ref)
    sys.path.insert(0, ref + "/examples")
    import run_adaptive_mesh_bubble as ex

    last = {"dt": float("nan"), "n": 0}
    orig_dt = ex.choose_step_dt
    dts = beatnik_dts(dt_from) if dt_from else None
    if dts is not None:
        print("[m1d3ref] dt pinned to %d Beatnik step sizes from %s" % (
            len(dts), dt_from), flush=True)

    def choose_step_dt(*a, **k):
        if dts is None:
            dt = orig_dt(*a, **k)
        else:
            if last["n"] >= len(dts):
                raise RuntimeError("dt schedule exhausted at call %d"
                                   % last["n"])
            dt = float(dts[last["n"]])
        last["n"] += 1
        last["dt"] = float(dt)
        return dt

    orig_save = ex.save_state_checkpoint

    def save_state_checkpoint(directory, state, **k):
        print("[m1d3ref] step=%d time=%.17g dt=%.17g V=%d F=%d" % (
            int(k["step"]), float(k["current_time"]), last["dt"],
            np.asarray(state.vertices).shape[0],
            np.asarray(state.faces).shape[0]), flush=True)
        return orig_save(directory, state, **k)

    ex.choose_step_dt = choose_step_dt
    ex.save_state_checkpoint = save_state_checkpoint
    sys.argv = [ex.__file__] + list(ref_argv)
    print("[m1d3ref] reference argv: " + " ".join(sys.argv[1:]), flush=True)
    ex.main()


def main():
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "convert":
        cmd_convert(a[1], a[2])
    elif len(a) >= 3 and a[0] == "run":
        ref, rest = a[1], a[2:]
        dt_from = None
        if rest[:1] == ["--dt-from"]:
            dt_from, rest = rest[1], rest[2:]
        if rest[:1] != ["--"]:
            sys.exit(__doc__)
        cmd_run(ref, dt_from, rest[1:])
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
