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
"""T9r: the reference treecode's own claim-A error on the level-4 member's 81
gold states (`tasks/add-canopy-t6.md` T9r).

    reference_treecode_error.py [--gold DIR]

**A MEASUREMENT TOOL, NOT A TEST.** It is registered in no tier, has no ctest
case and no manifest line, and its exit status says whether the *measurement*
completed -- never whether the reference met any bound.

WHAT IT MEASURES
----------------
For each `checkpoint_t*_stepNNNNNNN.npz` in the milestone-0 level-4 gold set it
builds the reference's `MeshPotentialZModelState(vertices, faces, potential)`
and evaluates `potential_mesh_birkhoff_rott_velocity` twice: at
`br_approximation="direct"` and at `"treecode"`. The error per state is

    rel = max_i |u_tree[i] - u_direct[i]| / max_i |u_direct[i]|

with row 2-norms, which is claim A's quantity in
`Beatnik_Test_Milestone0Fmm.cpp` (`fieldDifference`, `fieldScale`). Step 0 has
an identically zero potential and so a zero field; it is reported in absolute
form and excluded from the worst, as the member does.

The parameters are the gold set's (`--eps 0.025 --source-quadrature vertex`,
blob = eps^2 under `use_matlab_blob=False`) and the reference's treecode
defaults (theta 0.3, order 2, ncrit 64), all set **explicitly** and printed, so
a later change to the reference's defaults cannot silently move the number.

`zmodel3d` is imported from `PYTHONPATH`, never copied. Do not run this under
`python3 -I`, which ignores `PYTHONPATH`.
"""

import argparse
import os
import re
import sys
from typing import List, Optional

import numpy as np

GOLD_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "milestone0-sub4-2000-steps", "gold")
STEP_FILE = re.compile(r"^checkpoint_t(\d+)p(\d+)_step(\d{7})\.npz$")
EXPECTED_STATES = 81

EPS = 0.025
TREE_THETA = 0.3
TREE_ORDER = 2
TREE_NCRIT = 64


def gold_files(gold: str) -> List[str]:
    """The numbered step files, in step order. `checkpoint_latest.npz` (a copy
    of step 2000) and `README.md` do not match and are not states."""
    names = sorted((n for n in os.listdir(gold) if STEP_FILE.match(n)),
                   key=lambda n: int(STEP_FILE.match(n).group(3)))
    if len(names) != EXPECTED_STATES:
        raise SystemExit(f"[t9r] FAIL: {len(names)} step files in {gold}, "
                         f"expected {EXPECTED_STATES}")
    return [os.path.join(gold, n) for n in names]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gold", default=GOLD_DEFAULT)
    args = parser.parse_args(argv)

    try:
        import zmodel3d
        from zmodel3d.mesh_solver import (MeshPotentialZModelState,
                                          MeshZModelParams,
                                          potential_mesh_birkhoff_rott_velocity)
    except ImportError as exc:
        raise SystemExit(f"[t9r] FAIL: cannot import zmodel3d ({exc}); set "
                         "PYTHONPATH to the reference repository root") from exc

    print(f"[t9r] python   = {sys.version.split()[0]}")
    print(f"[t9r] numpy    = {np.__version__}")
    print(f"[t9r] zmodel3d = {zmodel3d.__file__}")
    print(f"[t9r] gold     = {os.path.abspath(args.gold)}")

    common = dict(eps=EPS, use_matlab_blob=False, source_quadrature="vertex")
    p_direct = MeshZModelParams(br_approximation="direct", **common)
    p_tree = MeshZModelParams(br_approximation="treecode",
                              br_treecode_theta=TREE_THETA,
                              br_treecode_order=TREE_ORDER,
                              br_treecode_ncrit=TREE_NCRIT, **common)
    print(f"[t9r] params eps={p_tree.eps!r} "
          f"use_matlab_blob={p_tree.use_matlab_blob!r} "
          f"source_quadrature={p_tree.source_quadrature!r} "
          f"br_sign={p_tree.br_sign!r} forcing_sign={p_tree.forcing_sign!r} "
          f"br_treecode_theta={p_tree.br_treecode_theta!r} "
          f"br_treecode_order={p_tree.br_treecode_order!r} "
          f"br_treecode_ncrit={p_tree.br_treecode_ncrit!r}")

    worst = (-1.0, -1, 0.0)
    rows = 0
    for path in gold_files(args.gold):
        m = STEP_FILE.match(os.path.basename(path))
        with np.load(path) as d:
            step = int(d["step"])
            time = float(d["time"])
            state = MeshPotentialZModelState(
                vertices=np.asarray(d["vertices"], dtype=float),
                faces=np.asarray(d["faces"]),
                potential=np.asarray(d["potential"], dtype=float))
        # The filename's time is %.6f, so it agrees with `time` to 5e-7.
        name_time = float(f"{int(m.group(1))}.{m.group(2)}")
        if step != int(m.group(3)) or abs(time - name_time) > 5.0e-7:
            raise SystemExit(f"[t9r] FAIL: {path}: step/time {step}/{time!r} "
                             "disagree with the filename")
        if state.vertices.shape != (2562, 3) or state.faces.shape != (5120, 3) \
                or state.potential.shape != (2562,):
            raise SystemExit(f"[t9r] FAIL: {path}: unexpected array shapes")

        u_direct = potential_mesh_birkhoff_rott_velocity(state, p_direct)
        u_tree = potential_mesh_birkhoff_rott_velocity(state, p_tree)
        if not (np.all(np.isfinite(u_direct)) and np.all(np.isfinite(u_tree))):
            raise SystemExit(f"[t9r] FAIL: {path}: non-finite velocity")
        max_direct = float(np.max(np.linalg.norm(u_direct, axis=1)))
        max_abs = float(np.max(np.linalg.norm(u_tree - u_direct, axis=1)))
        if max_direct == 0.0:
            print(f"[t9r] row step={step} time={time:.17g} rel=absolute "
                  f"max_abs={max_abs:.17g} max_direct=0", flush=True)
        else:
            rel = max_abs / max_direct
            print(f"[t9r] row step={step} time={time:.17g} rel={rel:.17g} "
                  f"max_direct={max_direct:.17g}", flush=True)
            if rel > worst[0]:
                worst = (rel, step, time)
        rows += 1

    print(f"[t9r] worst rel={worst[0]:.17g} step={worst[1]} "
          f"time={worst[2]:.17g} rows={rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
