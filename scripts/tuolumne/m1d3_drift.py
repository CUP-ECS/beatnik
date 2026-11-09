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
"""M1-D3 (`tasks/milestone1.md`): why Beatnik's milestone-1 trajectory drifts
from the gold after about gold step 1000. Analysis of existing files only.

    /usr/tce/bin/python3 -I scripts/tuolumne/m1d3_drift.py \\
        GOLD_DIR BEATNIK_CKPT_DIR BEATNIK_LAUNCH_LOG OUT_DIR

``GOLD_DIR`` holds the reference ``checkpoint_t*_step*.npz``: the M1-G3 gold
(steps 0..1610 are used; 1620 and 1624 are read only for the ``late`` table)
or an ``m1d3_refruns.flux`` run. ``BEATNIK_CKPT_DIR`` holds the compared run's
checkpoints: one M1-A1j launch's ``checkpoint_t*_step*.h5`` (every 10 steps),
or, if it has no ``.h5``, a reference run's ``.npz``. ``BEATNIK_LAUNCH_LOG``
is that launch's ``launch.log`` (its per-step ``[m1traj]`` times give the
compared run's ``dt``) or ``-`` (``dt`` from the checkpoint times). Only
``GOLD_DIR`` checkpoints inside the compared run's time span are compared.
Writes ``OUT_DIR/m1d3_drift.json`` and prints the tables. Refuses to write
into a non-empty ``OUT_DIR``.

Everything is compared at matched physical time: for a gold checkpoint at
``time`` t, the Beatnik quantity is linearly interpolated between the two
Beatnik checkpoints that bracket t. The meshes differ by selection, so shapes
are paired geometrically, never by index:

- **radial function**: from the gold's area-weighted centroid, rays along a
  fixed (polar angle, azimuth) grid are cast against each mesh; ``r`` is the
  farthest hit (the number of hits per ray is recorded, so a surface that is
  not star-shaped from the centroid is visible). ``dr = r_B - r_G``;
- **polar bands**: faces are binned by the polar angle of their centroid about
  the gold's centroid; per band, area, face count and mean edge length.

The statistic definitions (area, centroid ``z``, minimum edge, quality) are
imported from ``tests/regression_tests/milestone1_gold_stats.py`` so they are
the test's.
"""

import glob
import json
import os
import re
import sys

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests", "regression_tests"))
from milestone1_gold_stats import stats  # noqa: E402

NTH, NPH = 36, 24          # 5-degree polar bands, 15-degree azimuths
LAST_GOLD = 1610


def load_npz(f):
    d = np.load(f)
    return (int(d["step"]), float(d["time"]),
            np.asarray(d["vertices"], np.float64),
            np.asarray(d["faces"]).astype(np.int64),
            float(d["initial_volume"]))


def load_h5(f):
    with h5py.File(f, "r") as h:
        return (int(h["beatnik/step"][()]), float(h["beatnik/time"][()]),
                h["vertices/position"][:].astype(np.float64),
                h["faces/verts"][:].astype(np.int64),
                float(h["beatnik/initial_volume"][()]))


def series(pattern, loader):
    out = {}
    for f in glob.glob(pattern):
        if "latest" in f:
            continue
        s, t, X, F, V0 = loader(f)
        out[s] = (t, X, F, V0)
    return out


def directions():
    th = (np.arange(NTH) + 0.5) * np.pi / NTH
    ph = np.arange(NPH) * 2 * np.pi / NPH
    T, P = np.meshgrid(th, ph, indexing="ij")
    D = np.stack([np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)],
                 -1).reshape(-1, 3)
    return th, D


def raycast(X, F, o, D):
    """Möller-Trumbore, every ray against every face. Returns the farthest
    positive hit distance per ray and the hit count."""
    a, b, c = X[F[:, 0]] - o, X[F[:, 1]] - o, X[F[:, 2]] - o
    e1, e2 = b - a, c - a
    p = np.cross(D[:, None, :], e2[None])
    det = np.einsum("nmk,mk->nm", p, e1)
    ok = np.abs(det) > 1e-14
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    s = -a[None]
    u = np.einsum("nmk,nmk->nm", s, p) * inv
    q = np.cross(s, e1[None])
    v = np.einsum("nk,nmk->nm", D, q) * inv
    t = np.einsum("mk,nmk->nm", e2, q) * inv
    hit = ok & (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12) & (t > 0)
    r = np.where(hit, t, -np.inf).max(1)
    return r, hit.sum(1)


def bands(X, F, o):
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    cen = (a + b + c) / 3.0 - o
    th = np.arccos(np.clip(cen[:, 2] / np.linalg.norm(cen, axis=1), -1, 1))
    k = np.minimum((th / np.pi * NTH).astype(int), NTH - 1)
    L = (np.linalg.norm(b - a, axis=1) + np.linalg.norm(c - b, axis=1)
         + np.linalg.norm(a - c, axis=1)) / 3.0
    area = np.bincount(k, A, NTH)
    nf = np.bincount(k, minlength=NTH).astype(float)
    lsum = np.bincount(k, L, NTH)
    zmom = np.bincount(k, A * ((a + b + c) / 3.0)[:, 2], NTH)
    return area, nf, lsum, zmom


def centroid(X, F):
    a, b, c = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    A = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return (A[:, None] * (a + b + c) / 3.0).sum(0) / A.sum()


def main(argv):
    gold_dir, bk_dir, launch_log, out_dir = argv[1:5]
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit("m1d3_drift: refusing to write into non-empty " + out_dir)
    os.makedirs(out_dir, exist_ok=True)

    gold_all = series(os.path.join(gold_dir, "checkpoint_t*_step*.npz"),
                      load_npz)
    bk = series(os.path.join(bk_dir, "checkpoint_t*_step*.h5"), load_h5)
    if not bk:
        bk = series(os.path.join(bk_dir, "checkpoint_t*_step*.npz"), load_npz)
    bsteps = sorted(bk)
    btimes = np.array([bk[s][0] for s in bsteps])
    # Only reference checkpoints inside the compared run's time span.
    gold = {s: v for s, v in gold_all.items()
            if s <= LAST_GOLD and btimes[0] <= v[0] <= btimes[-1]}

    # Per-step time from a Beatnik launch.log's [m1traj]; with "-", the
    # compared run's checkpoint times (a mean dt over each 10-step interval).
    if launch_log != "-":
        traj = {0: 0.0}
        with open(launch_log) as fh:
            for line in fh:
                m = re.match(r"\[m1traj\] step=(\d+) time=(\S+)", line)
                if m:
                    traj[int(m.group(1))] = float(m.group(2))
        ts = np.array([traj[k] for k in sorted(traj)])
        bdt = np.diff(ts)                 # bdt[k-1] = dt of step k
    else:
        ts = btimes
        bdt = np.diff(btimes) / np.diff(np.array(bsteps, float))
    bdt_mid = 0.5 * (ts[1:] + ts[:-1])

    th, D = directions()
    gsteps = sorted(gold)
    gtime = np.array([gold[s][0] for s in gsteps])
    cache = {}

    def bk_at(s, o):
        key = (s, tuple(o))
        if key not in cache:
            t, X, F, V0 = bk[s]
            r, n = raycast(X, F, o, D)
            cache[key] = (r, n, bands(X, F, o), stats(X, F, V0),
                          centroid(X, F))
        return cache[key]

    rows = []
    for gs in gsteps:
        t, X, F, V0 = gold[gs]
        o = centroid(X, F)
        rG, nG = raycast(X, F, o, D)
        bG = bands(X, F, o)
        sG = stats(X, F, V0)
        j = int(np.searchsorted(btimes, t))
        j = min(max(j, 1), len(bsteps) - 1)
        s0, s1 = bsteps[j - 1], bsteps[j]
        w = (t - bk[s0][0]) / (bk[s1][0] - bk[s0][0])
        A0, A1 = bk_at(s0, o), bk_at(s1, o)
        rB = (1 - w) * A0[0] + w * A1[0]
        bB = [(1 - w) * x0 + w * x1 for x0, x1 in zip(A0[2], A1[2])]
        sB = {k: (1 - w) * A0[3][k] + w * A1[3][k] for k in A0[3]}
        cB = (1 - w) * A0[4] + w * A1[4]
        dr = (rB - rG).reshape(NTH, NPH)
        rGm = rG.reshape(NTH, NPH)
        # dt at matched time: gold's mean over the 10 steps ending at gs,
        # Beatnik's per-step dt interpolated to the same mid-time.
        if gs - 10 in gold_all:
            tp = gold_all[gs - 10][0]
            gdt = (t - tp) / 10.0
            bdt_m = float(np.interp(0.5 * (t + tp), bdt_mid, bdt))
        else:
            gdt = bdt_m = float("nan")
        rows.append(dict(
            gold_step=gs, time=t, bk_lo=s0, bk_hi=s1, w=w,
            F_gold=len(F), F_bk=float((1 - w) * len(bk[s0][2])
                                      + w * len(bk[s1][2])),
            area_rel=sB["area"] / sG["area"] - 1.0,
            rise_err=sB["zc"] - sG["zc"],
            cxy_bk=[float(cB[0] - o[0]), float(cB[1] - o[1])],
            lmin_gold=sG["lmin"], lmin_bk=sB["lmin"],
            lmean_gold=sG["lmean"], lmean_bk=sB["lmean"],
            qmin_gold=sG["qmin"], qmin_bk=sB["qmin"],
            dt_gold=gdt, dt_bk=bdt_m,
            multi_hit_gold=int((nG > 1).sum()),
            multi_hit_bk=int(max((A0[1] > 1).sum(), (A1[1] > 1).sum())),
            miss_bk=int(max((A0[1] == 0).sum(), (A1[1] == 0).sum())),
            dr_max=float(np.abs(dr).max()),
            dr_rms=float(np.sqrt((dr ** 2).mean())),
            dr_theta=dr.mean(1).tolist(),
            dr_theta_sd=dr.std(1).tolist(),
            rG_theta=rGm.mean(1).tolist(),
            rG_theta_sd=rGm.std(1).tolist(),
            area_band_gold=bG[0].tolist(), area_band_bk=bB[0].tolist(),
            nf_band_gold=bG[1].tolist(), nf_band_bk=bB[1].tolist(),
            l_band_gold=(bG[2] / np.maximum(bG[1], 1)).tolist(),
            l_band_bk=(bB[2] / np.maximum(bB[1], 1)).tolist(),
            zmom_band_gold=bG[3].tolist(), zmom_band_bk=bB[3].tolist(),
        ))

    # Late gold files beyond the set: shape only.
    late = []
    for gs in sorted(s for s in gold_all if s > LAST_GOLD):
        t, X, F, V0 = gold_all[gs]
        o = centroid(X, F)
        sG = stats(X, F, V0)
        late.append(dict(gold_step=gs, time=t, F=len(F), area=sG["area"],
                         zc=sG["zc"], lmin=sG["lmin"], qmin=sG["qmin"],
                         nf_band=bands(X, F, o)[1].tolist(),
                         l_band=(bands(X, F, o)[2]
                                 / np.maximum(bands(X, F, o)[1], 1)).tolist()))

    out = dict(theta_deg=(th * 180 / np.pi).tolist(), rows=rows, late=late,
               bk_time_final=float(btimes[-1]), bk_steps=bsteps[-1],
               bk_dt=bdt.tolist())
    with open(os.path.join(out_dir, "m1d3_drift.json"), "w") as fh:
        json.dump(out, fh)

    print("gold_step time bk_lo F_g F_b area_rel rise_err dr_max dr_rms "
          "dt_g dt_b dt_ratio lmin_g lmin_b multi_g multi_b miss_b")
    for r in rows:
        if r["gold_step"] % 50 and r["gold_step"] != LAST_GOLD:
            continue
        print("%5d %.6f %5d %5d %7.1f %+.3e %+.3e %.3e %.3e %.3e %.3e %.4f "
              "%.4e %.4e %d %d %d" % (
                  r["gold_step"], r["time"], r["bk_lo"], r["F_gold"],
                  r["F_bk"], r["area_rel"], r["rise_err"], r["dr_max"],
                  r["dr_rms"], r["dt_gold"], r["dt_bk"],
                  r["dt_bk"] / r["dt_gold"], r["lmin_gold"], r["lmin_bk"],
                  r["multi_hit_gold"], r["multi_hit_bk"], r["miss_bk"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
