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
"""Generate the milestone-1 gold-statistics header from the M1-G3 gold set.

    /usr/tce/bin/python3 -I milestone1_gold_stats.py \\
        /p/lustre5/stewartj/beatnik/M1-G3j/out \\
        tests/regression_tests/milestone1-bubble-1610/README.md \\
        tests/regression_tests/Beatnik_Milestone1GoldStats.hpp

Reads the reference checkpoints ``*_step%07d.npz`` for steps 0, 10, ..., 1610
(162 files; ``tasks/milestone1.md``, M1-G3) and writes one C++ table row per
file, every floating-point value at 17 significant digits:

    step, time, V, F, total area, area-weighted centroid z, volume drift,
    minimum triangle quality, valence 5 / 6 / 7 counts, minimum and mean
    edge length

The definitions are those of M1-D2's
``/p/lustre5/stewartj/beatnik/M1-D2/analysis/structural_stats.py``:

- area: ``sum 1/2 |(b-a) x (c-a)|`` over faces;
- centroid z: the face-area-weighted mean of the face centroids' z;
- volume drift: ``vol / initial_volume - 1`` with
  ``vol = sum a . (b x c) / 6`` and the file's own ``initial_volume``;
- quality: ``4 sqrt(3) A / (l_ab^2 + l_bc^2 + l_ca^2)`` per face, its minimum;
- edges: the unique undirected edges of the face list; valence is the number
  of unique edges at a vertex, counted over every row of ``vertices``.

REFUSES TO EMIT (exit 1, nothing written) if the directory does not hold
exactly one file for each of the 162 steps 0, 10, ..., 1610, if any numeric
field of any of them is nonfinite, if a file's stored ``step`` is not its
name's, or if any row's step / ``time`` (to 6 decimals) / V / F differs from
the gold README's table. Files for steps above 1610 (1620, 1624) are never
read.
"""

import glob
import os
import re
import sys

import numpy as np

FIRST, LAST, EVERY = 0, 1610, 10
STEPS = list(range(FIRST, LAST + 1, EVERY))
KEYS = ("faces", "initial_min_edge", "initial_volume", "potential",
        "remesh_material_position", "state_model", "step", "time", "vertices")


def fail(msg):
    print("milestone1_gold_stats: REFUSING TO EMIT: " + msg, file=sys.stderr)
    sys.exit(1)


def stats(X, F, V0):
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    l2 = ((b - a) ** 2).sum(1) + ((c - b) ** 2).sum(1) + ((a - c) ** 2).sum(1)
    q = 4.0 * np.sqrt(3.0) * A / l2
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]),
                axis=1)
    E = np.unique(E, axis=0)
    L = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    val = np.bincount(E.ravel(), minlength=len(X))
    cen = (A[:, None] * (a + b + c) / 3.0).sum(0) / A.sum()
    return dict(area=float(A.sum()), zc=float(cen[2]),
                vol_drift=float(vol / V0 - 1.0), qmin=float(q.min()),
                v5=int((val == 5).sum()), v6=int((val == 6).sum()),
                v7=int((val == 7).sum()), lmin=float(L.min()),
                lmean=float(L.mean()))


def readme_table(path):
    """The rows of the README's `## Per-checkpoint series` table only; the
    cap section's 1620 / 1624 rows above it are not part of the set."""
    rows = {}
    pat = re.compile(r"^\| (\d+) \| (\d+\.\d{6}) \| (\d+) \| (\d+) \|$")
    in_series = False
    with open(path) as fh:
        for line in fh:
            if line.startswith("## "):
                in_series = line.startswith("## Per-checkpoint series")
                continue
            if not in_series:
                continue
            m = pat.match(line.rstrip("\n"))
            if m:
                s = int(m.group(1))
                if s in rows:
                    fail("README lists step %d twice" % s)
                rows[s] = (m.group(2), int(m.group(3)), int(m.group(4)))
    return rows


def g17(x):
    return "%.17g" % x


def main(argv):
    if len(argv) != 4:
        fail("usage: milestone1_gold_stats.py <npz-dir> <README.md> <out.hpp>")
    npz_dir, readme, out_path = argv[1:4]

    by_step = {}
    for f in glob.glob(os.path.join(npz_dir, "*_step*.npz")):
        m = re.search(r"_step(\d{7})\.npz$", f)
        if not m:
            continue
        s = int(m.group(1))
        if s <= LAST:
            by_step.setdefault(s, []).append(f)
    if sorted(by_step) != STEPS:
        fail("expected exactly the steps %d, %d, ..., %d (%d files); found %d "
             "distinct steps <= %d: extra %s, missing %s" % (
                 FIRST, FIRST + EVERY, LAST, len(STEPS), len(by_step), LAST,
                 sorted(set(by_step) - set(STEPS)),
                 sorted(set(STEPS) - set(by_step))))
    for s, files in by_step.items():
        if len(files) != 1:
            fail("step %d has %d files: %s" % (s, len(files), files))

    table = readme_table(readme)
    if sorted(table) != STEPS:
        fail("the README table does not list exactly the steps 0..1610 every "
             "10 (found %d rows)" % len(table))

    rows = []
    for s in STEPS:
        f = by_step[s][0]
        d = np.load(f)
        if sorted(d.files) != sorted(KEYS):
            fail("%s: keys %s, expected %s" % (f, sorted(d.files),
                                               sorted(KEYS)))
        for k in KEYS:
            if k == "state_model":
                continue
            if not np.all(np.isfinite(d[k])):
                fail("%s: nonfinite value in %s" % (f, k))
        if int(d["step"]) != s:
            fail("%s: stored step %d" % (f, int(d["step"])))
        t = float(d["time"])
        X = np.asarray(d["vertices"], dtype=np.float64)
        F = np.asarray(d["faces"]).astype(np.int64)
        r = dict(step=s, time=t, V=len(X), F=len(F),
                 **stats(X, F, float(d["initial_volume"])))
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                fail("step %d: statistic %s is nonfinite" % (s, k))
        want = table[s]
        got = ("%.6f" % t, r["V"], r["F"])
        if got != want:
            fail("step %d: file gives time/V/F %s, README %s" % (s, got, want))
        rows.append(r)

    lines = []
    lines.append("""/****************************************************************************
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
 * @file Beatnik_Milestone1GoldStats.hpp
 * @brief GENERATED by `tests/regression_tests/milestone1_gold_stats.py` -- do
 *        not edit by hand. The statistics of the M1-G3 gold set, steps 0,
 *        10, ..., 1610 (162 checkpoints), read from
 *        `/p/lustre5/stewartj/beatnik/M1-G3j/out/` and recorded in
 *        `tests/regression_tests/milestone1-bubble-1610/README.md`.
 *
 * One row per checkpoint, at 17 significant digits. The definitions are the
 * generator's docstring: face-area-weighted centroid `z`, total area,
 * `vol / initial_volume - 1`, minimum `4 sqrt(3) A / sum l^2`, valence over
 * the unique edges, and the minimum and mean unique-edge length.
 */

#ifndef BEATNIK_MILESTONE1GOLDSTATS_HPP
#define BEATNIK_MILESTONE1GOLDSTATS_HPP

namespace Beatnik
{
namespace Test
{
namespace Milestone1
{

/// One gold checkpoint's statistics.
struct GoldStats
{
    int step;
    double time;
    long long vertices;
    long long faces;
    double area;
    double centroid_z;
    double volume_drift;
    double min_quality;
    long long valence5;
    long long valence6;
    long long valence7;
    double min_edge;
    double mean_edge;
};

/// Rows in `kGoldStats`: steps 0 through 1610 every 10.
constexpr int kGoldStatsRows = %d;

// step, time, V, F, area, centroid_z, volume_drift, min_quality,
// valence 5 / 6 / 7, min_edge, mean_edge
constexpr GoldStats kGoldStats[kGoldStatsRows] = {""" % len(rows))
    for r in rows:
        lines.append("    { %d, %s, %d, %d, %s, %s, %s, %s, %d, %d, %d, %s, %s },"
                     % (r["step"], g17(r["time"]), r["V"], r["F"],
                        g17(r["area"]), g17(r["zc"]), g17(r["vol_drift"]),
                        g17(r["qmin"]), r["v5"], r["v6"], r["v7"],
                        g17(r["lmin"]), g17(r["lmean"])))
    lines.append("""};

} // namespace Milestone1
} // namespace Test
} // namespace Beatnik

#endif // BEATNIK_MILESTONE1GOLDSTATS_HPP""")
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("milestone1_gold_stats: wrote %d rows (steps %d..%d) to %s"
          % (len(rows), FIRST, LAST, out_path))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
