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
"""T5 step 7's divergence ladder, for a run whose vertices have MOVED too far
to be paired by position.

    fmm_divergence_ladder.py ladder --run DIR --ref DIR [--label L] [--json OUT]

**A MEASUREMENT TOOL, NOT A TEST.** It is registered in no tier, has no ctest
case and no manifest line, and its exit status says whether the *measurement*
completed -- never whether Beatnik agreed with anything.

WHY THIS EXISTS RATHER THAN `milestone0_ladder.py pair`
------------------------------------------------------
It should not exist, and for a direct-driven run it must not be used: `pair` is
the instrument, its ladder definition is the one on record, and T5 step 7 was
written to reuse it. It cannot be reused **for an FMM-driven run**, and the
reason is a property of the pairing rather than a bug in either tool.

`compare_output.py` recovers the vertex correspondence the files do not record
by quantizing coordinates onto a grid of cell size `--match-eps` (default
`1e-9`) and lexsorting the integer keys. Its own docstring states the
precondition: the cell must be **much larger than the coordinate disagreement
between the two files** and much smaller than the vertex separation. A
direct-driven Beatnik run satisfies it by nine decades -- it tracks the Python
gold set to `8.5e-13`. An FMM-driven run does not: at order 3 the per-evaluation
velocity error is `5.0e-4` of the field, the trajectories separate to `1.5e-7`
by step 25 and to `5.1e-3` by step 2000, so the disagreement is 100x the cell at
step 25 and grows past the vertex spacing later.

When the precondition fails the two files quantize into different cells, the
lexsort orders them differently, and **vertices are paired with the wrong
partners**. The reported `max|e|` is then about half the bubble diameter -- `0.5`
on a radius-0.25 sphere -- where the true disagreement is `1.5e-7`.

**The failure is silent.** `n_ambiguous` stays `0` throughout, because it counts
rows sharing a cell with their predecessor *within one file*, which detects a
within-file collision and not a cross-file mis-pairing. Nothing else in the
report moves: the entity counts match, the load succeeds, the exit status is an
ordinary "compared and disagreed". A session reading that output would conclude
the FMM destroys the trajectory in 25 steps.

**Raising `--match-eps` does not fix it**, which is why this tool pairs
differently rather than passing a larger cell through. Measured on the level-3
FMM run against its gold set, the largest `max|e|` any cell size recovers:

    step    eps 1e-9     1e-7      1e-6      1e-5      1e-4      1e-3
      25    4.95e-1   5.00e-1   1.46e-7   1.46e-7   1.46e-7   1.46e-7
     100    4.97e-1   4.95e-1   4.99e-1   2.36e-6   2.36e-6   2.36e-6
     400    5.28e-1   5.28e-1   5.22e-1   4.87e-1   4.46e-1   5.61e-5
    1000    5.94e-1   5.94e-1   5.68e-1   5.68e-1   2.78e-1   5.41e-1

Each step needs a larger cell than the last, and by step 1000 the window between
"larger than the disagreement" and "smaller than the vertex spacing" has closed.
There is no single value, and a per-step value chosen to make the answer small
would be fitting the instrument to the result.

HOW THIS TOOL PAIRS INSTEAD
---------------------------
Bijective nearest neighbour: each `run` vertex takes the closest `ref` vertex,
and the pairing is **rejected** unless the map is a bijection. That is the
property a quantized lexsort cannot check across two files and this can:
a mis-pairing shows up as two run vertices claiming one ref vertex. A step whose
pairing is not bijective is reported as UNPAIRABLE and contributes no number --
it is not silently tabulated at whatever the mis-pairing produced.

It is `O(N^2)` per step and deliberately so. At milestone-0's 642 and 2562
vertices that is 0.4 M and 6.6 M distance evaluations per checkpoint, which is
nothing, and a tree would add a dependency for no gain. **Do not reach for this
at production vertex counts** -- pair by `/vertices/gid` instead, which is exact
and free. That is not available here: the Python `.npz` gold files carry no
`gid`, which is the whole reason a position-based correspondence has to be
recovered at all.

WHAT IT REPORTS
---------------
The same elementwise criterion `compare_output.py` applies at
[compare_output.py:442], `|e_i| <= atol + rtol*|g_i|`, evaluated directly rather
than bounded from the printed `max|e|` and `max_rel`: the pairing is in hand, so
the first failing step per rung is computed exactly and needs no
derive-then-confirm pass. The rungs are imported from `milestone0_ladder.RUNGS`
so the two tools cannot disagree about what a rung is, and `load_any` is
imported from `compare_output` so `FIELD_MAP` stays the single place a dataset
name lives.

Per-vertex fields are compared under the pairing. Scalars (`time`,
`initial_volume`, `initial_min_edge`) are compared directly and reported, but
are **excluded from the ladder**: `time` diverges because the adaptive dt is a
function of the state, so under `--adaptive-dt` an FMM-driven run and a
direct-driven one are at slightly different physical times at the same step, and
a ladder that failed on `time` would be reporting that rather than the
trajectory. Which fields were compared is printed, because the two possible
right-hand sides do not carry the same set -- `sheet_vector` is absent from the
reference `.npz`, so a Beatnik-vs-Beatnik comparison covers a strictly larger
set than a Beatnik-vs-Python one and the two are not directly comparable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import compare_output as co  # noqa: E402  -- after the path insert
from milestone0_ladder import RUNGS, steps_in  # noqa: E402

# Fields that are one value per vertex and therefore need the pairing applied.
PER_VERTEX = ("vertices", "potential", "sheet_vector", "remesh_material_position")

# Scalars: reported, never laddered. See the module docstring on `time`.
SCALARS = ("time", "initial_volume", "initial_min_edge")


# ===========================================================================
# Pairing
# ===========================================================================
def bijective_nearest(run: np.ndarray, ref: np.ndarray
                      ) -> Tuple[Optional[np.ndarray], str]:
    """Pair each `run` row with its nearest `ref` row; require a bijection.

    :returns: ``(index, note)``. `index` is ``None`` when the pairing is not a
        bijection, in which case `note` says how it failed. A non-bijective
        pairing is a refusal, not a warning: the number it would produce is the
        very artifact this tool exists to avoid.
    """
    if run.shape != ref.shape:
        return None, f"shape mismatch {run.shape} vs {ref.shape}"
    # (N,N) in one allocation. N is 642 or 2562 here; see the docstring on why
    # the quadratic form is deliberate.
    d2 = ((run[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
    idx = d2.argmin(1)
    uniq = np.unique(idx)
    if uniq.size != idx.size:
        return None, (f"NOT BIJECTIVE: {idx.size - uniq.size} of {idx.size} "
                      "run vertices share a ref vertex")
    return idx, ""


# ===========================================================================
# The ladder
# ===========================================================================
def fails(err: np.ndarray, gold: np.ndarray, rtol: float, atol: float) -> bool:
    """`compare_output.py`'s elementwise criterion, evaluated directly."""
    return bool(np.any(np.abs(err) > atol + rtol * np.abs(gold)))


def cmd_ladder(args: argparse.Namespace) -> int:
    lhs = steps_in(args.run)
    rhs = steps_in(args.ref)
    common = sorted(set(lhs) & set(rhs))
    label = args.label or f"{os.path.basename(args.run)} vs {os.path.basename(args.ref)}"

    print(f"### {label}")
    print(f"run  {args.run}   ({len(lhs)} checkpoints)")
    print(f"ref  {args.ref}   ({len(rhs)} checkpoints)")
    print(f"compared steps: {len(common)}")
    print("pairing: bijective nearest neighbour (NOT the comparator's "
          "quantized lexsort -- see this file's docstring)")
    if not common:
        print("NOTHING TO COMPARE -- the two directories share no step.")
        return 1

    rows: List[Dict[str, object]] = []
    fields_seen: set = set()
    unpairable: List[int] = []

    for step in common:
        a = co.load_any(lhs[step])
        b = co.load_any(rhs[step])
        va = np.asarray(a["vertices"], dtype=float)
        vb = np.asarray(b["vertices"], dtype=float)
        idx, note = bijective_nearest(va, vb)
        if idx is None:
            unpairable.append(step)
            rows.append({"step": step, "pairable": False, "note": note})
            continue

        row: Dict[str, object] = {"step": step, "pairable": True, "fields": {}}
        for name in PER_VERTEX:
            if name not in a or name not in b:
                continue
            fields_seen.add(name)
            xa = np.asarray(a[name], dtype=float)
            xb = np.asarray(b[name], dtype=float)
            if xa.shape[0] != va.shape[0] or xb.shape[0] != vb.shape[0]:
                continue
            e = xa - xb[idx]
            g = xb[idx]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = np.abs(e) / np.abs(g)
                rel = rel[np.isfinite(rel)]
            row["fields"][name] = {
                "max_abs": float(np.abs(e).max()),
                "max_rel": float(rel.max()) if rel.size else 0.0,
                "first_failing_rung": [
                    bool(fails(e, g, rt, at)) for rt, at in RUNGS
                ],
            }
        for name in SCALARS:
            if name in a and name in b:
                row[name] = (float(np.asarray(a[name]).reshape(()).item()),
                             float(np.asarray(b[name]).reshape(()).item()))
        rows.append(row)

    print(f"fields compared (per-vertex, laddered): {sorted(fields_seen)}")
    print(f"scalars reported, NOT laddered: {list(SCALARS)}")
    if unpairable:
        print(f"*** UNPAIRABLE STEPS (no bijection; contribute no number): "
              f"{unpairable} ***")

    # -- the growth series, full precision ---------------------------------
    print()
    hdr = f"{'step':>6}"
    for name in sorted(fields_seen):
        hdr += f" {name + ' max|e|':>26}"
    print(hdr)
    for row in rows:
        if not row.get("pairable"):
            print(f"{row['step']:>6} {'UNPAIRABLE: ' + str(row['note']):>26}")
            continue
        line = f"{row['step']:>6}"
        for name in sorted(fields_seen):
            f = row["fields"].get(name)
            line += f" {(f['max_abs'] if f else float('nan')):>26.17g}"
        print(line)

    # -- the ladder --------------------------------------------------------
    print()
    print("tolerance ladder: first failing checkpointed step, EXACT "
          "(elementwise, no derivation)")
    print(f"{'rtol':>10} {'atol':>10} {'first failing step':>19}  per-field")
    ladder: List[Dict[str, object]] = []
    for i, (rt, at) in enumerate(RUNGS):
        first_any: Optional[int] = None
        per_field: Dict[str, Optional[int]] = {}
        for name in sorted(fields_seen):
            first = None
            for row in rows:
                if not row.get("pairable"):
                    continue
                f = row["fields"].get(name)
                if f and f["first_failing_rung"][i]:
                    first = row["step"]
                    break
            per_field[name] = first
            if first is not None and (first_any is None or first < first_any):
                first_any = first
        ladder.append({"rtol": rt, "atol": at, "first_failing_step": first_any,
                       "per_field": per_field})
        detail = ", ".join(f"{k}={v}" for k, v in per_field.items())
        print(f"{rt:>10.0e} {at:>10.0e} {str(first_any):>19}  {detail}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"label": label, "run": args.run, "ref": args.ref,
                       "pairing": "bijective-nearest-neighbour",
                       "fields": sorted(fields_seen),
                       "unpairable_steps": unpairable,
                       "rows": rows, "ladder": ladder}, handle, indent=1)
        print(f"wrote {args.json}")

    # Exit status is about the MEASUREMENT, never about agreement. An
    # unpairable step means part of the series could not be measured at all,
    # which is the one thing that makes the table incomplete.
    return 1 if unpairable else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    lad = sub.add_parser(
        "ladder",
        help="per-step comparison and exact tolerance ladder under a "
             "bijective nearest-neighbour vertex pairing")
    lad.add_argument("--run", required=True,
                     help="directory of Beatnik .h5 checkpoints")
    lad.add_argument("--ref", required=True,
                     help="gold .npz directory, or a second run's .h5 directory")
    lad.add_argument("--label", default="")
    lad.add_argument("--json", default="")
    lad.set_defaults(func=cmd_ladder)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
