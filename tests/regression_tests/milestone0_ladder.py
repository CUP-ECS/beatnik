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
"""M0-D1's tabulation helper: the divergence horizon and the tolerance ladder.

    milestone0_ladder.py pair   --run DIR --ref DIR [--label L] [--json OUT]
    milestone0_ladder.py series --dir DIR [--label L] [--json OUT]

**A MEASUREMENT TOOL, NOT A TEST.** It is registered in no tier, has no ctest
case and no manifest line, and its exit status says whether the *measurement*
completed -- never whether Beatnik agreed with anything. A tool whose exit
status meant agreement would be a test with an unrecorded tolerance, which is
exactly what M0-D1 exists to replace.

WHAT IT DRIVES, AND WHY IT DRIVES IT ONCE
-----------------------------------------
``compare_output.py`` prints, per field, ``max|e|`` and ``max|e|/|g|`` whether
or not the field passes. Those two numbers bound the elementwise criterion at
[compare_output.py:442], ``|e_i| <= atol + rtol*|g_i|``, from both sides:

  * ``max|e| <= atol``   => every entry passes, at any rtol.
  * ``max_rel <= rtol``  => every entry passes, at any atol.

So ONE run of the comparator per step, at the tightest rung, tells us for every
rung whether that step *can* fail -- without re-running it five times per step.
What it does not tell us is whether the step *does* fail: the worst-``|e|``
entry need not be the worst-relative one, so a step where both bounds are
exceeded may still pass elementwise. The derived step is therefore a LOWER
BOUND on the first failing step, and this tool confirms it with real
invocations at that rung before reporting anything (M0-D1 step 3). Both numbers
are printed and both go in the progress log; the CONFIRMED one is the answer.

STEPS ARE FOUND BY THEIR ``_step%07d`` SUFFIX, never by rebuilding a name from
a time -- the time is one of the compared quantities (milestone0.md
Conventions). A step present on one side and missing on the other is reported
by name and skipped, not silently dropped from the denominator.

``pair`` handles both comparisons M0-D1 needs, because they differ only in what
is on the right: a gold directory of ``.npz`` (Beatnik vs Python) or a second
run directory of ``.h5`` (Beatnik rank 1 vs rank 4, the attribution of M0-D1
step 4). ``compare_output.py`` takes either on either side, so nothing here
branches on it.

**THE TWO COMPARISONS ARE NOT OVER THE SAME FIELDS, and that is a trap M0-D1
walked into before it was written down here.** The reference's ``.npz`` carries
nine keys and ``sheet_vector`` is not one of them, so Beatnik-vs-Python cannot
compare that field at all -- while Beatnik-vs-Beatnik, two ``.h5`` files from
the same writer, does. ``compare_output.py`` compares whatever is present in
both, which is right, but it means a Beatnik-vs-Beatnik horizon is measured over
a strictly LARGER field set than a Beatnik-vs-Python one and the two numbers are
not directly comparable. Every subcommand here therefore reports which fields it
actually compared, and ``pair`` prints the derived first-failing step PER FIELD
so the attribution can be read on the shared subset.

``series`` computes, offline in NumPy from one directory of checkpoints, the two
series M0-D1 step 5 records for BOTH codes:

  * the relative volume drift ``V/V0 - 1``, with ``V = (1/6) sum_f a.(b x c)``
    over ``faces`` -- the same convention ``SurfaceOperators::enclosedVolume``
    and T2d's check use, and ``V0`` read from the file's own ``initial_volume``
    so the two are comparable file by file;
  * the minimum triangle quality ``4*sqrt(3)*A / sum(l^2)`` (the project
    convention, ``src/Beatnik_Params.hpp:220``), which is what separates M0-R2
    (the frozen mesh gives out) from M0-R1 (the codes stop agreeing).

Loading goes through ``compare_output.load_any``, imported rather than
reimplemented, so ``FIELD_MAP`` stays the single place a dataset name lives.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compare_output as co  # noqa: E402  (path set above)


# ===========================================================================
# The ladder itself
# ===========================================================================
# M0-D1 step 3's five rungs: `--rtol 1e-12/1e-10/1e-8/1e-6/1e-4`, `--atol` two
# decades below each. Written out rather than generated, so the pair that was
# actually run is readable in the source and in the log.
RUNGS: Tuple[Tuple[float, float], ...] = (
    (1.0e-12, 1.0e-14),
    (1.0e-10, 1.0e-12),
    (1.0e-8, 1.0e-10),
    (1.0e-6, 1.0e-8),
    (1.0e-4, 1.0e-6),
)

# The tightest rung is what the single per-step scan runs at, because the bounds
# above are only useful if the scan's atol is at or below every rung's.
SCAN_RTOL, SCAN_ATOL = RUNGS[0]

# The fields that get their own printed table. M0-D1's exit criterion names the
# first two; `sheet_vector` is here because the measurement showed it is the
# field that actually BINDS -- it is O(1) in magnitude where `vertices` is O(0.1)
# and it carries entries near zero, so at the 1e-12/1e-14 rung it is the first
# field to put an entry outside `atol` and it is what sets the horizon. Every
# field the comparator prints is captured in the JSON regardless.
GROWTH_FIELDS = ("vertices", "potential", "sheet_vector")


# ===========================================================================
# Finding the checkpoints
# ===========================================================================
_STEP_RE = re.compile(r"_step(\d{7})\.(h5|npz)$")


def steps_in(directory: str) -> Dict[int, str]:
    """Map step -> file path for every ``_step%07d.{h5,npz}`` in *directory*.

    Anything without that suffix is ignored, which is deliberate and is what
    makes ``checkpoint_latest.npz`` inert here exactly as it is to
    ``goldForStep`` (M0-G1's log entry). A duplicated step is a hard error: two
    files claiming the same step means the directory holds two runs, and picking
    one would silently measure a mixture.
    """
    found: Dict[int, str] = {}
    for path in sorted(glob.glob(os.path.join(directory, "*"))):
        m = _STEP_RE.search(os.path.basename(path))
        if not m:
            continue
        step = int(m.group(1))
        if step in found:
            raise SystemExit(
                f"milestone0_ladder: {directory} holds two files for step "
                f"{step}: {found[step]} and {path}. Two runs share this "
                f"directory; the measurement would be a mixture of both."
            )
        found[step] = path
    if not found:
        raise SystemExit(
            f"milestone0_ladder: no *_step%07d.h5 or .npz files in {directory}"
        )
    return found


# ===========================================================================
# Driving compare_output.py
# ===========================================================================
_ARRAY_RE = re.compile(
    r"^\s+(\S+)\s+n=(\d+)\s+max\|e\|=(\S+)\s+max\|e\|/\|g\|=(\S+)\s+"
    r"L2=(\S+)\s+outside tol=(\d+)\s*$"
)
_SCALAR_RE = re.compile(
    r"^\s+(\S+)\s+cpp=(\S+)\s+gold=(\S+)\s+\|e\|=(\S+)\s*$"
)
_STRUCT_RE = re.compile(
    r"^structure: vertices cpp=\((\d+), 3\) gold=\((\d+), 3\); "
    r"faces cpp=\((\d+), 3\) gold=\((\d+), 3\)\s*$"
)
_MATCH_RE = re.compile(
    r"^matching \(eps=(\S+)\): (\d+)/(\d+) unambiguous, "
    r"ambiguous cpp=(\d+) gold=(\d+)\s*$"
)


def comparator_path() -> str:
    return os.path.join(_HERE, "compare_output.py")


def python_exe() -> str:
    return os.environ.get("BEATNIK_PYTHON", sys.executable or "python3")


def run_comparator(
    lhs: str, rhs: str, rtol: float, atol: float, quiet: bool
) -> Tuple[int, str]:
    """Run the comparator once. Returns (exit status, stdout).

    Exit status 0 is a match, **exactly 1** is compared-and-disagreed, 2 is a
    load error and anything else a plumbing failure. The three are never
    conflated (milestone0.md Conventions): a load error reported as a mismatch
    would put a broken path into the ladder as a divergence.
    """
    cmd = [python_exe(), comparator_path(), lhs, rhs,
           "--rtol", repr(rtol), "--atol", repr(atol)]
    if quiet:
        cmd.append("--quiet")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(
            f"milestone0_ladder: comparator exit {proc.returncode} on\n"
            f"  {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}\n"
        )
    return proc.returncode, proc.stdout


def parse_report(text: str) -> Dict[str, object]:
    """Pull the per-field numbers out of one non-quiet comparator report."""
    fields: Dict[str, Dict[str, float]] = {}
    out: Dict[str, object] = {"fields": fields}
    for line in text.splitlines():
        m = _ARRAY_RE.match(line)
        if m:
            fields[m.group(1)] = {
                "n": float(m.group(2)),
                "max_abs": float(m.group(3)),
                "max_rel": float(m.group(4)),
                "l2": float(m.group(5)),
                "outside": float(m.group(6)),
            }
            continue
        m = _SCALAR_RE.match(line)
        if m:
            gold = float(m.group(3))
            err = float(m.group(4))
            fields[m.group(1)] = {
                "n": 1.0,
                "max_abs": err,
                "max_rel": err / abs(gold) if gold != 0.0 else 0.0,
                "l2": err,
                "outside": 0.0,
            }
            continue
        m = _STRUCT_RE.match(line)
        if m:
            out["n_vertices_lhs"] = int(m.group(1))
            out["n_vertices_rhs"] = int(m.group(2))
            out["n_faces_lhs"] = int(m.group(3))
            out["n_faces_rhs"] = int(m.group(4))
            continue
        m = _MATCH_RE.match(line)
        if m:
            out["matched"] = int(m.group(2))
            out["n_paired"] = int(m.group(3))
            out["ambiguous_lhs"] = int(m.group(4))
            out["ambiguous_rhs"] = int(m.group(5))
    return out


# ===========================================================================
# Derivation and confirmation
# ===========================================================================
def can_fail(entry: Dict[str, float], rtol: float, atol: float) -> bool:
    """Whether a field's (max|e|, max|e|/|g|) pair permits an elementwise fail.

    Both bounds must be exceeded. Either one alone proves the field passes; see
    the module docstring. This is a NECESSARY condition for failure, not a
    sufficient one, which is why every derived step is confirmed.
    """
    return entry["max_abs"] > atol and entry["max_rel"] > rtol


def derive_first_failing(
    scan: Dict[int, Dict[str, object]], rtol: float, atol: float
) -> Optional[int]:
    """The first scanned step at which SOME field permits a fail at this rung."""
    for step in sorted(scan):
        report = scan[step]
        if report.get("status") not in (0, 1):
            continue
        fields = report["fields"]  # type: ignore[index]
        for entry in fields.values():  # type: ignore[union-attr]
            if can_fail(entry, rtol, atol):
                return step
    return None


def confirm_first_failing(
    lhs: Dict[int, str],
    rhs: Dict[int, str],
    ordered_steps: List[int],
    derived: Optional[int],
    rtol: float,
    atol: float,
) -> Dict[str, object]:
    """Find the true first failing checkpointed step with real invocations.

    Starts at the derived candidate -- which is a lower bound, so the true step
    is at or after it -- and walks forward until the comparator actually exits
    1. Then it re-runs the PREVIOUS checkpointed step, which must exit 0: two
    invocations is the minimum that pins a first-failing step, and the pair is
    what M0-D1 step 3 requires. If the derived candidate itself fails, the
    walk-back checks the step before it, and if THAT fails too the derivation
    was not a lower bound and the discrepancy is reported rather than smoothed
    over -- it would mean the bound reasoning above is wrong, which is worth
    knowing loudly.
    """
    result: Dict[str, object] = {
        "rtol": rtol, "atol": atol, "derived": derived,
        "confirmed": None, "invocations": 0, "note": "",
    }
    if derived is None:
        result["note"] = "no step permits a failure at this rung; nothing to confirm"
        return result

    start = ordered_steps.index(derived)
    confirmed = None
    for step in ordered_steps[start:]:
        status, _ = run_comparator(lhs[step], rhs[step], rtol, atol, quiet=True)
        result["invocations"] = int(result["invocations"]) + 1
        if status == 2 or status < 0:
            result["note"] = f"LOAD ERROR at step {step}; measurement incomplete"
            return result
        if status == 1:
            confirmed = step
            break
    if confirmed is None:
        result["note"] = ("no checkpointed step fails at this rung, though "
                          f"step {derived} permitted it")
        return result
    result["confirmed"] = confirmed

    # The previous checkpointed step must pass, or the confirmed step is not the
    # first one. Only meaningful when the walk did not move: if it did, the step
    # before `confirmed` was already run and passed.
    idx = ordered_steps.index(confirmed)
    if idx == 0:
        result["note"] = "the first checkpointed step already fails at this rung"
        return result
    prev = ordered_steps[idx - 1]
    if confirmed == derived:
        status, _ = run_comparator(lhs[prev], rhs[prev], rtol, atol, quiet=True)
        result["invocations"] = int(result["invocations"]) + 1
        if status == 1:
            result["note"] = (
                f"step {prev} ALSO fails at this rung, so the derivation was "
                f"not a lower bound -- walking back")
            # Walk back to the real first failure rather than reporting a wrong
            # one. Each step is a real invocation and each is counted.
            i = idx - 1
            while i > 0:
                status, _ = run_comparator(lhs[ordered_steps[i - 1]],
                                           rhs[ordered_steps[i - 1]],
                                           rtol, atol, quiet=True)
                result["invocations"] = int(result["invocations"]) + 1
                if status != 1:
                    break
                i -= 1
            result["confirmed"] = ordered_steps[i]
        elif status != 0:
            result["note"] = f"step {prev} comparator exit {status}"
    return result


# ===========================================================================
# `pair`
# ===========================================================================
def cmd_pair(args: argparse.Namespace) -> int:
    lhs = steps_in(args.run)
    rhs = steps_in(args.ref)

    common = sorted(set(lhs) & set(rhs))
    only_lhs = sorted(set(lhs) - set(rhs))
    only_rhs = sorted(set(rhs) - set(lhs))
    label = args.label or f"{os.path.basename(args.run)} vs {os.path.basename(args.ref)}"

    print(f"### {label}")
    print(f"run  {args.run}   ({len(lhs)} checkpoints)")
    print(f"ref  {args.ref}   ({len(rhs)} checkpoints)")
    print(f"compared steps: {len(common)}"
          f"  (run-only {only_lhs or 'none'}, ref-only {only_rhs or 'none'})")
    if not common:
        print("NOTHING TO COMPARE -- the two directories share no step.")
        return 1

    # -- the single scan, at the tightest rung -----------------------------
    scan: Dict[int, Dict[str, object]] = {}
    counts: Dict[Tuple[int, int], List[int]] = {}
    pairing_problems: List[str] = []
    for step in common:
        status, text = run_comparator(lhs[step], rhs[step],
                                      SCAN_RTOL, SCAN_ATOL, quiet=False)
        report = parse_report(text)
        report["status"] = status
        scan[step] = report
        if status not in (0, 1):
            pairing_problems.append(f"step {step}: comparator exit {status}")
            continue
        nv = report.get("n_vertices_lhs")
        nf = report.get("n_faces_lhs")
        if nv is not None and nf is not None:
            counts.setdefault((int(nv), int(nf)), []).append(step)
        if report.get("ambiguous_lhs") or report.get("ambiguous_rhs"):
            pairing_problems.append(
                f"step {step}: AMBIGUOUS vertex pairing "
                f"(cpp={report.get('ambiguous_lhs')}, "
                f"gold={report.get('ambiguous_rhs')}) -- risk M0-R4")

    # -- the structural gate ----------------------------------------------
    # M0-D1's exit criterion FAILS if the vertex or face count changes at any
    # step, and it is checked here rather than inferred from a passing
    # comparison: a count change is a structural failure of the comparator, so
    # it would otherwise arrive as "everything diverged at step N".
    print()
    print("counts (vertices, faces) seen across the compared steps:")
    for key, steps in sorted(counts.items()):
        print(f"  {key}: {len(steps)} step(s)"
              + ("" if len(counts) == 1 else f" -> {steps[:4]}..."))
    structural_ok = len(counts) == 1
    if not structural_ok:
        print("  *** COUNTS CHANGED. Adaptivity leaked into the frozen-mesh "
              "configuration; M0-D1's measurement FAILS. ***")
    for problem in pairing_problems:
        print(f"  ! {problem}")

    # -- the growth series -------------------------------------------------
    print()
    print(f"max|e| and max|e|/|g| per step at --rtol {SCAN_RTOL:g} "
          f"--atol {SCAN_ATOL:g}")
    header = f"{'step':>6}  {'exit':>4}"
    for field in GROWTH_FIELDS:
        header += f"  {field + ' max|e|':>18}  {field + ' max rel':>18}"
    header += f"  {'time |e|':>14}"
    print(header)
    for step in common:
        report = scan[step]
        row = f"{step:>6}  {report['status']:>4}"
        for field in GROWTH_FIELDS:
            entry = report["fields"].get(field)  # type: ignore[union-attr]
            if entry is None:
                row += f"  {'-':>18}  {'-':>18}"
            else:
                row += f"  {entry['max_abs']:>18.6e}  {entry['max_rel']:>18.6e}"
        t = report["fields"].get("time")  # type: ignore[union-attr]
        row += f"  {t['max_abs']:>14.6e}" if t else f"  {'-':>14}"
        print(row)

    # -- the ladder --------------------------------------------------------
    print()
    print("tolerance ladder: first failing checkpointed step, derived then "
          "confirmed")
    print(f"{'rtol':>10} {'atol':>10} {'derived':>10} {'confirmed':>10} "
          f"{'invocations':>12}  note")
    ladder: List[Dict[str, object]] = []
    for rtol, atol in RUNGS:
        derived = derive_first_failing(scan, rtol, atol)
        confirmation = confirm_first_failing(lhs, rhs, common, derived,
                                             rtol, atol)
        ladder.append(confirmation)
        print(f"{rtol:>10.0e} {atol:>10.0e} "
              f"{str(confirmation['derived']):>10} "
              f"{str(confirmation['confirmed']):>10} "
              f"{confirmation['invocations']:>12}  {confirmation['note']}")

    # -- WHICH FIELD BINDS, per rung ---------------------------------------
    # The ladder above is the minimum over fields, so it does not say WHAT gave
    # out. That matters here for a reason specific to M0-D1's two comparisons:
    # the reference's `.npz` carries no `sheet_vector`, so a Beatnik-vs-Python
    # ladder is over a strictly SMALLER field set than a Beatnik-vs-Beatnik one,
    # and comparing the two horizons without this table would attribute to
    # decomposition an effect that is partly just an extra compared field. The
    # per-field derived step is what makes the two comparable: restrict to the
    # fields both ladders share and read the same column.
    all_fields = sorted({f for r in scan.values()
                         for f in r["fields"]})  # type: ignore[union-attr]
    print()
    print("derived first-failing step BY FIELD (derivation only, not confirmed)")
    print(f"{'field':<28}" + "".join(f"{'%.0e' % r[0]:>10}" for r in RUNGS))
    per_field: Dict[str, List[Optional[int]]] = {}
    for field in all_fields:
        cells: List[Optional[int]] = []
        for rtol, atol in RUNGS:
            first = None
            for step in sorted(scan):
                entry = scan[step]["fields"].get(field)  # type: ignore[union-attr]
                if entry is not None and can_fail(entry, rtol, atol):
                    first = step
                    break
            cells.append(first)
        per_field[field] = cells
        if any(c is not None for c in cells):
            print(f"{field:<28}" + "".join(f"{str(c):>10}" for c in cells))

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"label": label, "run": args.run, "ref": args.ref,
                       "scan_rtol": SCAN_RTOL, "scan_atol": SCAN_ATOL,
                       "structural_ok": structural_ok,
                       "counts": {str(k): v for k, v in counts.items()},
                       "pairing_problems": pairing_problems,
                       "fields": all_fields,
                       "derived_by_field": per_field,
                       "scan": {str(k): v for k, v in scan.items()},
                       "ladder": ladder}, handle, indent=1)
        print(f"\nwrote {args.json}")

    # The measurement completed. Disagreement is the RESULT, not a failure of
    # this tool; only a structural change or a plumbing error is.
    return 0 if structural_ok and not any(
        p.startswith("step") and "exit" in p for p in pairing_problems) else 1


# ===========================================================================
# `series`
# ===========================================================================
def enclosed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """``V = (1/6) sum_f a.(b x c)`` -- T2d's convention, and Beatnik's."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    return float(np.sum(np.einsum("ij,ij->i", a, np.cross(b, c))) / 6.0)


def min_quality(vertices: np.ndarray, faces: np.ndarray) -> float:
    """``4*sqrt(3)*A / sum(l^2)`` per face, minimized. src/Beatnik_Params.hpp:220."""
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    l2 = (np.sum((b - a) ** 2, axis=1) + np.sum((c - b) ** 2, axis=1)
          + np.sum((a - c) ** 2, axis=1))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = 4.0 * math.sqrt(3.0) * area / l2
    return float(np.min(q))


def cmd_series(args: argparse.Namespace) -> int:
    files = steps_in(args.dir)
    label = args.label or os.path.basename(os.path.abspath(args.dir))
    print(f"### {label}")
    print(f"dir  {args.dir}   ({len(files)} checkpoints)")
    print()
    print(f"{'step':>6} {'time':>22} {'V/V0 - 1':>24} {'min quality':>22}")
    rows: List[Dict[str, object]] = []
    for step in sorted(files):
        data = co.load_any(files[step])
        vertices = np.asarray(data["vertices"], dtype=float)
        faces = np.asarray(data["faces"]).astype(np.int64)
        v0 = float(np.asarray(data["initial_volume"]).reshape(()).item())
        t = float(np.asarray(data["time"]).reshape(()).item())
        volume = enclosed_volume(vertices, faces)
        drift = volume / v0 - 1.0
        quality = min_quality(vertices, faces)
        rows.append({"step": step, "time": t, "volume": volume,
                     "drift": drift, "min_quality": quality,
                     "n_vertices": int(vertices.shape[0]),
                     "n_faces": int(faces.shape[0])})
        print(f"{step:>6} {t:>22.17g} {drift:>24.17g} {quality:>22.17g}")

    shapes = {(r["n_vertices"], r["n_faces"]) for r in rows}
    print()
    print(f"counts across the series: {sorted(shapes)}")
    if len(shapes) != 1:
        print("  *** COUNTS CHANGED across the series. ***")
    finals = rows[-1]
    print(f"final step {finals['step']}: time {finals['time']!r}, "
          f"drift {finals['drift']!r}, min quality {finals['min_quality']!r}")
    worst = min(rows, key=lambda r: r["min_quality"])
    print(f"global minimum quality {worst['min_quality']!r} at step "
          f"{worst['step']}")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"label": label, "dir": args.dir, "rows": rows}, handle,
                      indent=1)
        print(f"wrote {args.json}")
    return 0 if len(shapes) == 1 else 1


# ===========================================================================
# `growth`
# ===========================================================================
def cmd_growth(args: argparse.Namespace) -> int:
    """``max|e|`` per step on ``vertices`` and ``potential``, at FULL precision.

    ``pair`` harvests these from the comparator's own report, which prints
    ``%.6e`` -- six significant digits, which is enough to build the ladder (the
    rungs are decades apart) but throws away most of a growth series. This
    recomputes them in NumPy at the precision the doubles actually carry, and it
    does so through ``compare_output.quantized_lexsort`` -- IMPORTED, not
    reimplemented, so the pairing is the comparator's own and the two cannot
    disagree about which vertex is which. The ambiguity count it returns is
    reported for the same reason ``pair`` reports it: a degraded pairing (risk
    M0-R4) shows up as a uniform jump across every field at once.
    """
    lhs = steps_in(args.run)
    rhs = steps_in(args.ref)
    common = sorted(set(lhs) & set(rhs))
    label = args.label or "growth"

    # THE FIELD SET IS NOT SYMMETRIC ACROSS THE TWO COMPARISONS M0-D1 MAKES, and
    # this is where it becomes visible. The reference's `.npz` carries nine keys
    # and `sheet_vector` is NOT among them, so a Beatnik-vs-Python comparison
    # cannot see that field at all, while a Beatnik-vs-Beatnik comparison of two
    # `.h5` files does. `compare_output.py` compares whatever is present in both
    # and says so, which is correct behaviour -- but it means the two ladders are
    # over different field sets, and the fields actually compared are reported
    # here rather than assumed.
    probe_a = co.load_any(lhs[common[0]])
    probe_b = co.load_any(rhs[common[0]])
    fields = [f for f in GROWTH_FIELDS if f in probe_a and f in probe_b]
    missing = [f for f in GROWTH_FIELDS if f not in fields]

    print(f"### {label}")
    print(f"run  {args.run}")
    print(f"ref  {args.ref}")
    print(f"pairing: compare_output.quantized_lexsort at eps={args.match_eps:g}")
    print(f"fields compared: {fields}")
    if missing:
        print(f"fields ABSENT from one side and therefore NOT compared: "
              f"{missing}")
    print()
    print(f"{'step':>6} "
          + " ".join(f"{f + ' max|e|':>26}" for f in fields)
          + f" {'time |e|':>26} {'amb':>5}")
    rows: List[Dict[str, object]] = []
    for step in common:
        a = co.load_any(lhs[step])
        b = co.load_any(rhs[step])
        va = np.asarray(a["vertices"], dtype=float)
        vb = np.asarray(b["vertices"], dtype=float)
        if va.shape != vb.shape:
            print(f"{step:>6}  SHAPE MISMATCH {va.shape} vs {vb.shape}")
            return 1
        pa, amb_a = co.quantized_lexsort(va, args.match_eps)
        pb, amb_b = co.quantized_lexsort(vb, args.match_eps)
        row: Dict[str, object] = {"step": step, "ambiguous": amb_a + amb_b}
        for field in fields:
            qa = np.asarray(a[field], dtype=float)[pa]
            qb = np.asarray(b[field], dtype=float)[pb]
            row[field + "_max_abs"] = float(np.max(np.abs(qa - qb)))
        row["time_abs"] = abs(
            float(np.asarray(a["time"]).reshape(()).item())
            - float(np.asarray(b["time"]).reshape(()).item()))
        rows.append(row)
        print(f"{step:>6} "
              + " ".join(f"{row[f + '_max_abs']:>26.17e}" for f in fields)
              + f" {row['time_abs']:>26.17e} {amb_a + amb_b:>5}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"label": label, "run": args.run, "ref": args.ref,
                       "match_eps": args.match_eps, "fields": fields,
                       "fields_absent": missing, "rows": rows}, handle,
                      indent=1)
        print(f"wrote {args.json}")
    return 0


# ===========================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "M0-D1's tabulation helper. A measurement tool in no test tier: "
            "its exit status says whether the measurement completed, never "
            "whether the two sides agreed."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    pair = sub.add_parser(
        "pair",
        help="per-step comparison of a run directory against a gold directory "
             "or a second run directory, plus the derived-and-confirmed "
             "tolerance ladder")
    pair.add_argument("--run", required=True,
                      help="directory of Beatnik .h5 checkpoints")
    pair.add_argument("--ref", required=True,
                      help="gold .npz directory, or a second run's .h5 directory")
    pair.add_argument("--label", default="",
                      help="name for this comparison in the output")
    pair.add_argument("--json", default="",
                      help="write the raw scan and ladder here")
    pair.set_defaults(func=cmd_pair)

    series = sub.add_parser(
        "series",
        help="volume-drift and minimum-quality series for one directory of "
             "checkpoints, .h5 or .npz")
    series.add_argument("--dir", required=True)
    series.add_argument("--label", default="")
    series.add_argument("--json", default="")
    series.set_defaults(func=cmd_series)

    growth = sub.add_parser(
        "growth",
        help="max|e| per step on vertices and potential at FULL precision, "
             "through the comparator's own vertex pairing")
    growth.add_argument("--run", required=True)
    growth.add_argument("--ref", required=True)
    growth.add_argument("--label", default="")
    growth.add_argument("--json", default="")
    growth.add_argument("--match-eps", type=float, default=1.0e-9,
                        help="pairing cell size; the comparator's default")
    growth.set_defaults(func=cmd_growth)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
