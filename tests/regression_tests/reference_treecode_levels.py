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
"""The reference treecode's claim-A error on Beatnik checkpoints, at any
icosphere level (canopy_profiling, the L5/L6 accuracy question).

    reference_treecode_levels.py H5 [H5 ...] [--ncrit 64 8]

**A MEASUREMENT TOOL, NOT A TEST.** In no tier; the exit status says whether
the measurement completed, never whether any bound was met.

`reference_treecode_error.py` (T9r) does the same measurement on the level-4
gold `.npz` set. This reads Beatnik's own `.h5` checkpoints instead, so the
reference can be evaluated at levels 5 and 6, where no gold set exists, on the
states a Beatnik run produced. For each file it builds
`MeshPotentialZModelState(vertices, faces, potential)` and evaluates
`potential_mesh_birkhoff_rott_velocity` at `"direct"` and at `"treecode"` with
theta 0.3, order 2 (the reference's defaults) and each requested `ncrit`
(default 64, the reference's own, and 8, Beatnik's milestone value). The error
is claim A's quantity, `max_i |u_tree - u_direct| / max_i |u_direct|` with row
2-norms. A zero field is reported in absolute form.

`zmodel3d` comes from `PYTHONPATH`; do not run under `python3 -I`.
"""

import argparse
import sys
import time as wallclock

import h5py
import numpy as np

EPS = 0.025
TREE_THETA = 0.3
TREE_ORDER = 2


def load(path):
    with h5py.File(path, "r") as h:
        return dict(step=int(np.asarray(h["/beatnik/step"][()]).ravel()[0]),
                    time=float(np.asarray(h["/beatnik/time"][()]).ravel()[0]),
                    vertices=np.asarray(h["/vertices/position"][()], dtype=float),
                    faces=np.asarray(h["/faces/verts"][()]),
                    potential=np.asarray(h["/vertices/u0"][()], dtype=float))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5", nargs="+")
    ap.add_argument("--ncrit", type=int, nargs="+", default=[64, 8])
    args = ap.parse_args(argv)

    try:
        import zmodel3d
        from zmodel3d.mesh_solver import (MeshPotentialZModelState,
                                          MeshZModelParams,
                                          potential_mesh_birkhoff_rott_velocity)
    except ImportError as exc:
        raise SystemExit(f"[reflv] FAIL: cannot import zmodel3d ({exc})")

    print(f"[reflv] python={sys.version.split()[0]} numpy={np.__version__} "
          f"zmodel3d={zmodel3d.__file__}")
    common = dict(eps=EPS, use_matlab_blob=False, source_quadrature="vertex")
    p_direct = MeshZModelParams(br_approximation="direct", **common)
    trees = {n: MeshZModelParams(br_approximation="treecode",
                                 br_treecode_theta=TREE_THETA,
                                 br_treecode_order=TREE_ORDER,
                                 br_treecode_ncrit=n, **common)
             for n in args.ncrit}
    print(f"[reflv] params eps={EPS} use_matlab_blob=False "
          f"source_quadrature=vertex theta={TREE_THETA} order={TREE_ORDER} "
          f"ncrit={args.ncrit}", flush=True)

    for path in args.h5:
        d = load(path)
        nv = d["vertices"].shape[0]
        if d["faces"].shape[0] != 2 * nv - 4 or d["potential"].shape != (nv,):
            raise SystemExit(f"[reflv] FAIL: {path}: shapes {d['vertices'].shape}"
                             f" {d['faces'].shape} {d['potential'].shape}")
        state = MeshPotentialZModelState(vertices=d["vertices"],
                                         faces=d["faces"],
                                         potential=d["potential"])
        t0 = wallclock.time()
        u_direct = potential_mesh_birkhoff_rott_velocity(state, p_direct)
        t_direct = wallclock.time() - t0
        max_direct = float(np.max(np.linalg.norm(u_direct, axis=1)))
        for n, p in trees.items():
            t0 = wallclock.time()
            u_tree = potential_mesh_birkhoff_rott_velocity(state, p)
            t_tree = wallclock.time() - t0
            if not (np.all(np.isfinite(u_direct)) and np.all(np.isfinite(u_tree))):
                raise SystemExit(f"[reflv] FAIL: {path}: non-finite velocity")
            max_abs = float(np.max(np.linalg.norm(u_tree - u_direct, axis=1)))
            rel = (f"{max_abs / max_direct:.17g}" if max_direct > 0.0
                   else "absolute")
            print(f"[reflv] row vertices={nv} step={d['step']} "
                  f"time={d['time']:.17g} ncrit={n} rel={rel} "
                  f"max_abs={max_abs:.17g} max_direct={max_direct:.17g} "
                  f"direct_wall={t_direct:.2f} tree_wall={t_tree:.2f} "
                  f"file={path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
