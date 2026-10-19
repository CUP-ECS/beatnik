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
"""canopy_profiling T0's reducer: one tool for every FMM cost log.

    fmm_profile_reduce.py LOG [LOG ...] [--format auto|milestone|probe]
                          [--bin-steps N] [--json OUT] [--states]
    fmm_profile_reduce.py --self-test

**A MEASUREMENT TOOL, NOT A TEST.** It is registered in no tier and has no
ctest case. Its exit status says whether the *reduction* completed -- every
block parsed and attributed -- never whether a cost was acceptable.

It streams each log line by line (the T9b logs are about 37 MB each) and reads
two input formats.

FORMAT 1: MILESTONE LOGS (`Beatnik_Test_Milestone0Fmm*` under run_milestone.flux)
--------------------------------------------------------------------------------
Canopy's `print_solve_breakdown` (`../canopy/src/Canopy_Profiling.hpp`) prints,
on rank 0, after every `solve()`:

    [Canopy Diagnostics] solve() phase breakdown (N MPI rank[s])
      Phase                  Mean (s)  % of total
    ---------------------------------------------
      Total solve               0.261  100.0%
      Upward sweep              0.150   57.5%
      Downward sweep            0.110   42.1%
      P2P                       0.001    0.4%

Values are rank means printed `%.3f`, so anything near 1 ms -- P2P at levels 3
and 4 -- is at the print resolution and not a measurement.

A **block** is that header followed by the four rows, in that order. Any other
line inside a block is skipped: at np 4 other ranks' `[Canopy diag]` and
`[H2SCR]` lines land between the rows. A block is **truncated** when the next
header, a launch boundary or EOF arrives before its four rows; that is an
error naming the header's line.

Attribution, per launch (`[milestone] === <test> at N ranks ...`):

  claim A  Each block belongs to the state named by the most recent
           `claim A step S comparator exit` note: the test checkpoints step S,
           runs the comparator, then evaluates the standalone FMM at S. One
           block per state; 81 states.
  claim B  Blocks after the first `claim B step 0 relative drift` note. The
           k-th (0-based) is step k//3 + 1, RK3 stage k%3, for k < 3*steps.
           Block 3*steps is the **final probe**: the standalone FMM solver the
           test builds on claim B's final state to read the realized P2P
           fraction. It is not a timestep and is excluded from claim-B sums.
           Each `claim B step S` note is a cross-check: the block count at its
           first occurrence must be 3*S.

At np 4 every rank prints every `[note]`, so a boundary is taken from its first
occurrence per (claim, step). A block that cannot be attributed, a cross-check
that fails, or a block count that is not 3*steps or 3*steps + 1 is an error.

FORMAT 2: PROBE LOGS (`Beatnik_Probe_FmmCost`, canopy_profiling T2)
-------------------------------------------------------------------
T0 owns this grammar; the probe emits it. One row per line:

    [cost] run=<id> step=<n> key=value key=value ...
    [acc] run=<id> step=<n> key=value ...
    [cost-total] run=<id> key=value ...

Tokens are whitespace-separated `key=value` pairs. Keys match
`[A-Za-z_][A-Za-z0-9_.]*` and appear at most once per row. Values are numbers
where they parse as one, else strings (`space=HIP`, `canopy_hash=...`). A value
that is not available is printed as `-1`, never omitted; it is excluded from
statistics and reported as a sentinel count. `run` (and `step`, on `[cost]`
and `[acc]`) are required; `run` groups the rows of one launch, so a script
running several configurations in one log gives each its own `run`. Every
other key is read generically, so the probe can add fields without touching
this tool. Canopy breakdown blocks in a probe log (`canopy_report=1`) are
parsed and counted but not attributed; read such a log with `--format probe`.

**The per-row evaluation-count check** (canopy_profiling T3). Every `[cost]`
row must carry `steps`, `evals` and the four `maint_*` counts as non-negative
integers; a row that does not is malformed and fatal. A well-formed row must
then satisfy

    maint_setup + maint_migrate + maint_rebalance + maint_rebuild
        == evals == 3 * steps

(three RK3 evaluations per step, one maintenance action per evaluation). A row
that fails is **excluded** from every statistic and reported by `run` and
`step`; it is never averaged in, and it does not change the exit status. The
`[acc]` row at the same step is a standalone evaluation outside the window and
is kept.

`--states` prints one line per `[cost]` row of each probe run -- the phase
split and the `[acc]` metric at every sampled state, the view an O task's
before/after comparison needs.
"""

import argparse
import json
import math
import re
import statistics
import sys
from typing import Dict, List, Optional, Tuple

PHASES = ("total", "upward", "downward", "p2p")
ROW_LABELS = ("Total solve", "Upward sweep", "Downward sweep", "P2P")

BLOCK_HEADER = re.compile(
    r"^\[Canopy Diagnostics\] solve\(\) phase breakdown \((\d+) MPI ranks?\)\s*$")
BLOCK_ROW = re.compile(
    r"^\s+(Total solve|Upward sweep|Downward sweep|P2P)\s+(\S+)\s+(\S+)%\s*$")
LAUNCH = re.compile(
    r"^\[milestone\] === (\S+) at (\d+) ranks? / (\d+) node")
CLAIM_A_STATE = re.compile(r"claim A step (\d+) comparator exit")
CLAIM_B_STEP = re.compile(r"claim B step (\d+) relative drift")
STEPS_NOTE = re.compile(r"\[note\] \S+: execution space .* steps (\d+),")
COST_LINE = re.compile(r"^\[t6\] COST (.*)$")
PROBE_ROW = re.compile(r"^\[(cost|acc|cost-total)\](.*)$")
PROBE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")

# Total is computed from unrounded means; each printed value is rounded to
# 0.5e-3, so Total and the sum of the three phases may disagree by 2e-3.
SUM_TOLERANCE = 2.5e-3


class ReduceError(Exception):
    """A log the reducer cannot account for. Always fatal."""


def fail(path: str, lineno: int, msg: str) -> None:
    raise ReduceError(f"{path}:{lineno}: {msg}")


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def p90(xs: List[float]) -> float:
    """Nearest-rank 90th percentile."""
    s = sorted(xs)
    return s[max(0, math.ceil(0.9 * len(s)) - 1)]


def summarize(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "median": statistics.median(xs), "p90": p90(xs),
            "max": max(xs), "mean": statistics.fmean(xs), "sum": math.fsum(xs)}


# --------------------------------------------------------------------------
# Format 1: milestone logs
# --------------------------------------------------------------------------
class Launch:
    def __init__(self, test: str, ranks: int, lineno: int):
        self.test = test
        self.ranks = ranks
        self.lineno = lineno
        self.steps: Optional[int] = None
        self.cost: Dict[str, str] = {}
        self.claim_a_state: Optional[int] = None
        self.claim_a_counts: Dict[int, int] = {}
        self.in_claim_b = False
        self.seen_b_steps: Dict[int, int] = {}
        # Per block: (claim, step, stage, total, upward, downward, p2p)
        self.blocks: List[Tuple[str, int, int, float, float, float, float]] = []
        self.n_b = 0

    def label(self) -> str:
        return f"{self.test} np{self.ranks}"


class MilestoneReducer:
    def __init__(self, path: str, fmt: str):
        self.path = path
        self.fmt = fmt
        self.launches: List[Launch] = []
        self.cur: Optional[Launch] = None
        self.block_line = 0
        self.block_ranks = 0
        self.rows: Dict[str, float] = {}
        self.unattributed_blocks = 0
        self.total_blocks = 0

    # -- block assembly -----------------------------------------------------
    def _close_truncated(self, lineno: int, why: str) -> None:
        if self.block_line:
            got = ", ".join(self.rows) or "no rows"
            fail(self.path, self.block_line,
                 f"breakdown block truncated ({why} at line {lineno} before "
                 f"its four rows; got {got})")

    def _row(self, lineno: int, m: "re.Match") -> None:
        label = m.group(1)
        key = PHASES[ROW_LABELS.index(label)]
        expect = PHASES[len(self.rows)]
        if key != expect:
            fail(self.path, lineno,
                 f"breakdown block at line {self.block_line}: row "
                 f"'{label}' where '{ROW_LABELS[len(self.rows)]}' was expected")
        try:
            self.rows[key] = float(m.group(2))
        except ValueError:
            fail(self.path, lineno, f"unparsable value in row '{label}'")
        if len(self.rows) == 4:
            self._finish_block(lineno)

    def _finish_block(self, lineno: int) -> None:
        r = self.rows
        resid = abs(r["total"] - (r["upward"] + r["downward"] + r["p2p"]))
        if resid > SUM_TOLERANCE:
            fail(self.path, self.block_line,
                 f"breakdown block: Total {r['total']} differs from "
                 f"Upward+Downward+P2P by {resid:.4f} s")
        self.total_blocks += 1
        self._attribute(self.block_line, self.block_ranks,
                        (r["total"], r["upward"], r["downward"], r["p2p"]))
        self.block_line = 0
        self.rows = {}

    def _attribute(self, hdr: int, ranks: int,
                   v: Tuple[float, float, float, float]) -> None:
        L = self.cur
        if self.fmt == "probe":
            self.unattributed_blocks += 1
            return
        if L is None:
            fail(self.path, hdr,
                 "breakdown block outside any '[milestone] ===' launch; "
                 "cannot attribute it (a probe log with canopy_report=1 "
                 "needs --format probe)")
        if ranks != L.ranks:
            fail(self.path, hdr,
                 f"breakdown block reports {ranks} ranks inside a "
                 f"{L.ranks}-rank launch ({L.label()})")
        if L.in_claim_b:
            if L.steps is None:
                fail(self.path, hdr, "claim B block before the step count "
                     "note ('... steps N,') was seen")
            k = L.n_b
            L.n_b += 1
            if k < 3 * L.steps:
                L.blocks.append(("B", k // 3 + 1, k % 3) + v)
            elif k == 3 * L.steps:
                L.blocks.append(("final", L.steps, -1) + v)
            else:
                fail(self.path, hdr,
                     f"claim B block {k + 1} exceeds 3*steps + 1 = "
                     f"{3 * L.steps + 1} ({L.label()})")
        elif L.claim_a_state is not None:
            s = L.claim_a_state
            L.claim_a_counts[s] = L.claim_a_counts.get(s, 0) + 1
            if L.claim_a_counts[s] > 1:
                fail(self.path, hdr,
                     f"second breakdown block for claim A state {s} "
                     f"({L.label()}); expected one evaluation per state")
            L.blocks.append(("A", s, -1) + v)
        else:
            fail(self.path, hdr,
                 f"breakdown block before any claim A or claim B boundary "
                 f"note ({L.label()}); cannot attribute it")

    # -- launch bookkeeping --------------------------------------------------
    def _end_launch(self, lineno: int) -> None:
        L = self.cur
        if L is None or self.fmt == "probe":
            return
        if L.in_claim_b and L.steps is not None and \
                L.n_b not in (3 * L.steps, 3 * L.steps + 1):
            fail(self.path, lineno,
                 f"{L.label()}: {L.n_b} claim B blocks; expected 3*steps "
                 f"({3 * L.steps}) or 3*steps + 1 with the final probe")
        missing = [s for s, c in L.claim_a_counts.items() if c == 0]
        if missing:
            fail(self.path, lineno,
                 f"{L.label()}: claim A states with no breakdown block: "
                 f"{missing[:10]}")

    # -- driver ---------------------------------------------------------------
    def feed(self, lineno: int, line: str) -> None:
        m = BLOCK_HEADER.match(line)
        if m:
            self._close_truncated(lineno, "next block header")
            self.block_line = lineno
            self.block_ranks = int(m.group(1))
            self.rows = {}
            return
        m = LAUNCH.match(line)
        if m:
            self._close_truncated(lineno, "launch boundary")
            self._end_launch(lineno)
            self.cur = Launch(m.group(1), int(m.group(2)), lineno)
            self.launches.append(self.cur)
            return
        if self.block_line:
            m = BLOCK_ROW.match(line)
            if m:
                self._row(lineno, m)
            # Anything else inside a block is another rank's output, or the
            # column header and rule: skip it.
            return
        L = self.cur
        if L is None or self.fmt == "probe":
            return
        if L.steps is None:
            m = STEPS_NOTE.search(line)
            if m:
                L.steps = int(m.group(1))
        m = CLAIM_A_STATE.search(line)
        if m and not L.in_claim_b:
            s = int(m.group(1))
            if s not in L.claim_a_counts:
                L.claim_a_counts[s] = 0
                L.claim_a_state = s
            return
        m = CLAIM_B_STEP.search(line)
        if m:
            s = int(m.group(1))
            if s in L.seen_b_steps:
                return
            if not L.in_claim_b:
                if s != 0:
                    fail(self.path, lineno,
                         f"claim B step {s} note before claim B step 0")
                L.in_claim_b = True
            if L.n_b != 3 * s:
                fail(self.path, lineno,
                     f"{L.label()}: claim B step {s} note after {L.n_b} "
                     f"claim B blocks; expected 3*{s} = {3 * s}")
            L.seen_b_steps[s] = lineno
            return
        m = COST_LINE.match(line)
        if m:
            for tok in m.group(1).split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    L.cost[k] = v

    def finish(self, lineno: int) -> None:
        self._close_truncated(lineno, "end of file")
        self._end_launch(lineno)


# --------------------------------------------------------------------------
# Format 2: probe rows
# --------------------------------------------------------------------------
def parse_probe_row(path: str, lineno: int, tag: str,
                    body: str) -> Dict[str, object]:
    row: Dict[str, object] = {}
    for tok in body.split():
        if "=" not in tok:
            fail(path, lineno, f"[{tag}] token '{tok}' is not key=value")
        k, v = tok.split("=", 1)
        if not PROBE_KEY.match(k):
            fail(path, lineno, f"[{tag}] bad key '{k}'")
        if k in row:
            fail(path, lineno, f"[{tag}] duplicate key '{k}'")
        if v == "":
            fail(path, lineno, f"[{tag}] key '{k}' has no value "
                 "(print -1 for a missing value)")
        try:
            row[k] = float(v)
        except ValueError:
            row[k] = v
    required = ("run",) if tag == "cost-total" else ("run", "step")
    for k in required:
        if k not in row:
            fail(path, lineno, f"[{tag}] row has no '{k}'")
    return row


MAINT_KEYS = ("maint_setup", "maint_migrate", "maint_rebalance",
              "maint_rebuild")


def cost_count_check(path: str, lineno: int,
                     row: Dict[str, object]) -> Optional[str]:
    """The per-row evaluation-count check. Returns None when the row passes
    and the reason when it must be excluded; a row without the counts as
    non-negative integers is malformed and fatal."""
    counts = {}
    for k in ("steps", "evals") + MAINT_KEYS:
        v = row.get(k)
        if not isinstance(v, float) or v < 0 or v != int(v):
            fail(path, lineno, f"[cost] run={row['run']} step={row['step']}: "
                 f"'{k}' is {'missing' if v is None else repr(v)}; the "
                 "count check needs a non-negative integer")
        counts[k] = int(v)
    maint = sum(counts[k] for k in MAINT_KEYS)
    if maint == counts["evals"] == 3 * counts["steps"]:
        return None
    return (f"maint sum {maint}, evals {counts['evals']}, "
            f"3*steps {3 * counts['steps']}")


def reduce_probe(rows: Dict[str, List[Dict[str, object]]]) -> Dict:
    """Per run: each numeric field's statistics over [cost] and [acc] rows,
    with -1 sentinels counted and excluded, plus the [cost-total] row. [cost]
    rows that failed the count check (rows["excluded"]) are not in rows["cost"]
    and are listed per run under "excluded"."""
    out: Dict[str, Dict] = {}
    for row, why in rows.get("excluded", []):
        out.setdefault(str(row["run"]), {}).setdefault("excluded", []).append(
            {"step": row["step"], "reason": why})
    for tag in ("cost", "acc"):
        for row in rows.get(tag, []):
            run = str(row["run"])
            fields = out.setdefault(run, {}).setdefault(tag, {})
            for k, v in row.items():
                if k in ("run", "step") or not isinstance(v, float):
                    continue
                f = fields.setdefault(k, {"values": [], "sentinels": 0})
                if v == -1.0:
                    f["sentinels"] += 1
                else:
                    f["values"].append(v)
    for run, tags in out.items():
        for tag, fields in tags.items():
            if tag == "excluded":
                continue
            for k, f in fields.items():
                fields[k] = dict(summarize(f["values"]),
                                 sentinels=f["sentinels"])
    for row in rows.get("cost-total", []):
        out.setdefault(str(row["run"]), {})["total"] = row
    return out


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------
def read_log(path: str, fmt: str, stream=None) -> Tuple[MilestoneReducer, Dict]:
    ms = MilestoneReducer(path, fmt)
    probe_rows: Dict[str, List[Dict[str, object]]] = {}
    lineno = 0
    fh = stream if stream is not None else open(path, "r", errors="replace")
    try:
        for lineno, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            m = PROBE_ROW.match(line)
            if m and not ms.block_line:
                if fmt == "milestone":
                    fail(path, lineno, f"[{m.group(1)}] row in --format "
                         "milestone")
                tag = m.group(1)
                row = parse_probe_row(path, lineno, tag, m.group(2))
                why = (cost_count_check(path, lineno, row)
                       if tag == "cost" else None)
                if why is not None:
                    probe_rows.setdefault("excluded", []).append((row, why))
                else:
                    probe_rows.setdefault(tag, []).append(row)
                continue
            ms.feed(lineno, line)
        ms.finish(lineno + 1)
    finally:
        if stream is None:
            fh.close()
    return ms, probe_rows


# --------------------------------------------------------------------------
# Milestone report
# --------------------------------------------------------------------------
def launch_report(L: Launch, bin_steps: int) -> Dict:
    rep: Dict = {"test": L.test, "np": L.ranks, "steps": L.steps,
                 "cost": L.cost}
    for claim in ("A", "B", "final"):
        sel = [b for b in L.blocks if b[0] == claim]
        rep[f"n_{claim}"] = len(sel)
        if claim == "final":
            if sel:
                rep["final_probe"] = dict(zip(PHASES, sel[0][3:]))
            continue
        rep[claim] = {p: summarize([b[3 + i] for b in sel])
                      for i, p in enumerate(PHASES)}
        tot = rep[claim]["total"].get("sum", 0.0)
        rep[claim]["downward_share"] = (
            rep[claim]["downward"]["sum"] / tot if tot > 0 else float("nan"))
    rep["n_blocks"] = len(L.blocks)
    bins = []
    b_blocks = [b for b in L.blocks if b[0] == "B"]
    if b_blocks and L.steps:
        for lo in range(1, L.steps + 1, bin_steps):
            hi = min(lo + bin_steps - 1, L.steps)
            sel = [b for b in b_blocks if lo <= b[1] <= hi]
            if not sel:
                continue
            tot = math.fsum(b[3] for b in sel)
            bins.append({
                "steps": f"{lo}-{hi}", "evals": len(sel),
                "total_median": statistics.median(b[3] for b in sel),
                "upward_median": statistics.median(b[4] for b in sel),
                "downward_median": statistics.median(b[5] for b in sel),
                "p2p_median": statistics.median(b[6] for b in sel),
                "downward_share": (math.fsum(b[5] for b in sel) / tot
                                   if tot > 0 else float("nan")),
                "solve_per_step": tot / (hi - lo + 1)})
    rep["claim_b_bins"] = bins
    try:
        wall = float(L.cost["claimB"])
        rep["claimB_wall"] = wall
        rep["solve_share_of_claimB"] = rep["B"]["total"]["sum"] / wall
    except (KeyError, ValueError, ZeroDivisionError):
        rep["claimB_wall"] = None
        rep["solve_share_of_claimB"] = None
    return rep


def fmt_s(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def print_launch(rep: Dict) -> None:
    c = rep["cost"]
    print(f"\n=== {rep['test']}  np{rep['np']}  level {c.get('level', '?')}  "
          f"space {c.get('space', '?')}  steps {rep['steps']}")
    print(f"evaluations: {rep['n_blocks']} blocks = claim A {rep['n_A']} + "
          f"claim B {rep['n_B']} + final probe {rep['n_final']}")
    print("  claim  phase        median       p90       max      mean"
          "        sum")
    for claim in ("A", "B"):
        for p in PHASES:
            s = rep[claim][p]
            if not s["n"]:
                continue
            print(f"  {claim:5}  {p:9} {s['median']:9.3f} {s['p90']:9.3f} "
                  f"{s['max']:9.3f} {s['mean']:9.4f} {s['sum']:10.2f}")
        print(f"  {claim:5}  downward share of summed solve: "
              f"{100 * rep[claim]['downward_share']:.1f}%")
    if "final_probe" in rep:
        fp = rep["final_probe"]
        print("  final probe (standalone FMM on claim B's final state): "
              + "  ".join(f"{p} {fp[p]:.3f}" for p in PHASES))
    print(f"  claim B: summed solve {rep['B']['total'].get('sum', 0):.2f} s / "
          f"[t6] COST claimB {fmt_s(rep['claimB_wall'])} s = "
          f"{fmt_s(rep['solve_share_of_claimB'])}")
    print("  claim B trend (medians, s/eval; dn% of summed solve; "
          "solve s/step)")
    print("    steps        evals   total  upward    down     p2p    dn%"
          "  solve/step")
    for b in rep["claim_b_bins"]:
        print(f"    {b['steps']:11} {b['evals']:6} {b['total_median']:7.3f} "
              f"{b['upward_median']:7.3f} {b['downward_median']:7.3f} "
              f"{b['p2p_median']:7.3f} {100 * b['downward_share']:6.1f}"
              f"  {b['solve_per_step']:10.4f}")
    print("  (P2P is printed by Canopy as %.3f s; values near 0.001 are at "
          "the print resolution.)")


def print_probe(path: str, red: Dict) -> None:
    for run, tags in red.items():
        print(f"\n=== {path}  probe run {run}")
        n_pass = tags.get("cost", {}).get("evals", {}).get("n", 0)
        excl = tags.get("excluded", [])
        print(f"  count check (maint sum == evals == 3*steps): {n_pass} "
              f"[cost] row(s) passed, {len(excl)} excluded")
        for e in excl:
            print(f"  EXCLUDED [cost] run={run} step={e['step']:g}: "
                  f"{e['reason']}")
        for tag in ("cost", "acc"):
            for k, s in tags.get(tag, {}).items():
                if not s["n"]:
                    print(f"  [{tag}] {k:28} all -1 ({s['sentinels']} rows)")
                    continue
                print(f"  [{tag}] {k:28} n {s['n']:5}  median "
                      f"{s['median']:.6g}  p90 {s['p90']:.6g}  max "
                      f"{s['max']:.6g}  sentinels {s['sentinels']}")
        if "total" in tags:
            print("  [cost-total] " + " ".join(
                f"{k}={v}" for k, v in tags["total"].items()))


# The Canopy keys that fire inside ff_maintain under auto_maintain;
# commplan_dtt_traversal is nested in comm_plan_build. The rest of ff_maintain
# is untimed (canopy_profiling-progress-log.md, T2 "What only running
# revealed").
TIMED_MAINT = ("builder_build", "repartition", "sort_by_leaf",
               "comm_plan_build")
# auto_maintain's own timers and its Migrate-path redistribute, added by
# canopy_profiling O5. A log that predates them lacks the keys, and they count
# as 0 there, so its untimed column is unchanged.
TIMED_MAINT_O5 = ("maint_rebuild_check", "maint_key_snapshot",
                  "maint_compare_build", "maint_key_diff", "maint_sweep_setup",
                  "redistribute")
# Stage 3's sub-timers (canopy_profiling O9: ilist_s3_classify_shard,
# ilist_s3_merge, ilist_s3_instrumentation, ilist_s3_remap) are nested inside
# ilist_s3_classify_pairs, which runs under ff_solve, not ff_maintain. They are
# deliberately in neither tuple above; a log that predates them simply lacks
# the keys.
ILIST_S3_SPLIT = ("ilist_s3_classify_shard", "ilist_s3_merge",
                  "ilist_s3_instrumentation", "ilist_s3_remap")
# Everything in ff_evaluate_total outside ff_maintain and ff_solve.
ADAPTER = ("ff_pack", "ff_fwd_distribute", "ff_fwd_migrate", "ff_contract",
           "ff_rev_distribute", "ff_rev_migrate", "ff_scatter",
           "ff_diagnostics")


def state_line(c: Dict[str, object], a: Optional[Dict[str, object]]) -> str:
    def g(k: str) -> Optional[float]:
        v = c.get(k)
        return v if isinstance(v, float) and v != -1.0 else None

    def ratio(n: Optional[float], d: Optional[float]) -> str:
        if n is None or not d:
            return "n/a"
        return f"{n / d:.3f}"

    ffe, solve, maint = g("ff_evaluate_total"), g("ff_solve"), g("ff_maintain")
    timed = [g(k) for k in TIMED_MAINT] + \
        [g(k) or 0.0 for k in TIMED_MAINT_O5]
    untimed = (maint - sum(timed)) if maint is not None and \
        None not in timed else None
    adapter = [g(k) for k in ADAPTER]
    adapter_sum = sum(adapter) if None not in adapter else None
    actions = "/".join(f"{c[k]:g}" for k in MAINT_KEYS)
    acc = (f"{a['acc_rel']:.3e}" if a and isinstance(a.get("acc_rel"), float)
           else "n/a")
    dse = (f"{a['direct_step_est']:.4f}"
           if a and isinstance(a.get("direct_step_est"), float) else "n/a")
    return (f"{c['step']:6g} {c.get('time', -1):7.3f} "
            f"{c.get('step_wall', -1):8.4f} {ratio(3 * ffe if ffe else None, g('step_wall')):>6} "
            f"{solve if solve is not None else -1:8.4f} "
            f"{maint if maint is not None else -1:8.4f} "
            f"{untimed if untimed is not None else -1:8.4f} "
            f"{ratio(g('ilist_s4_op_table_build'), solve):>6} "
            f"{ratio(adapter_sum, ffe):>6} {actions:>11} "
            f"{c.get('p2p_frac', -1):6.3f} {c.get('fb_range_guard', -1):8g} "
            f"{acc:>10} {dse:>8}")


def print_states(path: str, rows: Dict[str, List[Dict[str, object]]]) -> None:
    acc = {(str(r["run"]), r["step"]): r for r in rows.get("acc", [])}
    runs: Dict[str, List[Dict[str, object]]] = {}
    for r in rows.get("cost", []):
        runs.setdefault(str(r["run"]), []).append(r)
    for run, cs in runs.items():
        print(f"\n=== {path}  states of probe run {run}  (per evaluation "
              "unless noted; untimed = ff_maintain - "
              "builder_build - repartition - sort_by_leaf - comm_plan_build "
              "- the maint_* keys - redistribute, where present)")
        print("  step    time   s/step  ffe/wall  solve    maint  untimed "
              "opb/sv adp/ffe s/m/r/R act   p2p  fb_rg     acc_rel  dir/step")
        for c in sorted(cs, key=lambda r: r["step"]):
            print("  " + state_line(c, acc.get((run, c["step"]))))


# --------------------------------------------------------------------------
# Self-test: the probe grammar and the block rules on hand-written fixtures
# --------------------------------------------------------------------------
_MILESTONE_FIXTURE = """\
[milestone] === Beatnik_Test_Milestone0FmmL4_MPI_HIP at 4 ranks / 1 node(s) ===
[note] Beatnik_Test_Milestone0Fmm: execution space HIP, ranks 4, icosphere subdivisions 4 (2562 vertices), steps 2, ncrit 8, order 3; claim B runs 2 FMM-driven steps
[note] Beatnik_Test_Milestone0Fmm: claim A step 0 comparator exit 0 (0 = match)
[note] Beatnik_Test_Milestone0Fmm: claim A step 0 comparator exit 0 (0 = match)
{blk}[note] Beatnik_Test_Milestone0Fmm: claim B step 0 relative drift 0 reference 0
[note] Beatnik_Test_Milestone0Fmm: claim B step 0 relative drift 0 reference 0
{blk}{blk}{blk_foreign}{blk}{blk}{blk}[note] Beatnik_Test_Milestone0Fmm: claim B step 2 relative drift 0 reference 0
{blk}[t6] COST level=4 space=HIP np=4 steps=2 claimA=1.0 claimB=2.0
"""
_BLK = ("[Canopy Diagnostics] solve() phase breakdown (4 MPI ranks)\n"
        "  Phase                  Mean (s)  % of total\n"
        "---------------------------------------------\n"
        "  Total solve               0.200  100.0%\n"
        "  Upward sweep              0.050   25.0%\n"
        "  Downward sweep            0.149   74.5%\n"
        "  P2P                       0.001    0.5%\n\n")
_BLK_FOREIGN = _BLK.replace(
    "  Downward", "[H2SCR] np=4 nleaves=782 lb=1.0000 lpt=1.0008\n  Downward")

_PROBE_FIXTURE = """\
[cost] run=l4_np1 step=1 level=4 np=1 space=HIP steps=1 evals=3 step_wall=0.50 ff_solve=0.40 m2l_kernel=-1 maint_setup=1 maint_migrate=2 maint_rebalance=0 maint_rebuild=0
[acc] run=l4_np1 step=1 claimA=1.1e-3 direct_wall=0.003
[cost] run=l4_np1 step=20 level=4 np=1 space=HIP steps=19 evals=57 step_wall=0.70 ff_solve=0.60 m2l_kernel=-1 maint_setup=0 maint_migrate=50 maint_rebalance=7 maint_rebuild=0
[cost-total] run=l4_np1 canopy_hash=abc123 rebuild=0 migrate=60
"""
# A window whose maintenance counts miss one evaluation: excluded, not fatal.
_PROBE_FAILING_ROW = (
    "[cost] run=l4_np1 step=40 steps=20 evals=60 step_wall=9.99 "
    "ff_solve=9.99 m2l_kernel=-1 maint_setup=0 maint_migrate=52 "
    "maint_rebalance=7 maint_rebuild=0\n")


def self_test() -> int:
    import io

    def run(text: str, fmt: str = "auto"):
        return read_log("<fixture>", fmt, io.StringIO(text))

    good = _MILESTONE_FIXTURE.format(blk=_BLK, blk_foreign=_BLK_FOREIGN)
    ms, _ = run(good)
    L = ms.launches[0]
    rep = launch_report(L, 1)
    assert (rep["n_A"], rep["n_B"], rep["n_final"]) == (1, 6, 1), rep
    assert rep["claimB_wall"] == 2.0
    assert [b[1:3] for b in L.blocks if b[0] == "B"] == \
        [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    # Truncated mid-table: drop the last two rows of the third block.
    trunc_blk = _BLK.split("  Downward")[0]
    bad = _MILESTONE_FIXTURE.format(blk=_BLK, blk_foreign=trunc_blk)
    try:
        run(bad)
        raise AssertionError("truncated block not detected")
    except ReduceError as e:
        assert "truncated" in str(e), e
    # A missing evaluation breaks the step-2 cross-check.
    short = _MILESTONE_FIXTURE.format(blk=_BLK, blk_foreign="")
    try:
        run(short)
        raise AssertionError("missing evaluation not detected")
    except ReduceError as e:
        assert "claim B step 2 note after 5" in str(e), e
    # A block outside any launch.
    try:
        run(_BLK)
        raise AssertionError("unattributed block not detected")
    except ReduceError as e:
        assert "outside any" in str(e), e

    _, rows = run(_PROBE_FIXTURE)
    red = reduce_probe(rows)
    c = red["l4_np1"]["cost"]
    assert c["step_wall"]["n"] == 2 and abs(c["step_wall"]["max"] - 0.7) < 1e-12
    assert c["m2l_kernel"]["n"] == 0 and c["m2l_kernel"]["sentinels"] == 2
    assert red["l4_np1"]["total"]["canopy_hash"] == "abc123"
    assert "excluded" not in red["l4_np1"]

    # The count check: the failing row is reported by run and step and
    # excluded from every statistic; the passing rows are unchanged.
    _, rows = run(_PROBE_FIXTURE + _PROBE_FAILING_ROW)
    red = reduce_probe(rows)
    c = red["l4_np1"]["cost"]
    assert c["step_wall"]["n"] == 2 and abs(c["step_wall"]["max"] - 0.7) < 1e-12
    assert c["ff_solve"]["max"] == 0.6, c["ff_solve"]
    assert red["l4_np1"]["excluded"] == [
        {"step": 40.0, "reason": "maint sum 59, evals 60, 3*steps 60"}], \
        red["l4_np1"]["excluded"]
    # A [cost] row without its counts is malformed, and fatal.
    for line in ("[cost] run=a step=1 steps=1 evals=3 maint_setup=1 "
                 "maint_migrate=2 maint_rebalance=0",
                 "[cost] run=a step=1 steps=1 evals=-1 maint_setup=1 "
                 "maint_migrate=2 maint_rebalance=0 maint_rebuild=0"):
        try:
            run(line + "\n")
            raise AssertionError(f"accepted malformed counts: {line}")
        except ReduceError as e:
            assert "count check" in str(e), e
    for line in ("[cost] run=a step=1 x", "[cost] run=a step=1 x=1 x=2",
                 "[cost] step=1 x=1", "[cost] run=a step=1 x="):
        try:
            run(line + "\n")
            raise AssertionError(f"accepted: {line}")
        except ReduceError:
            pass
    # Stage 3's sub-timer keys (O9) leave a state's untimed maintenance
    # unchanged, and a row without them reduces as before.
    base = ("[cost] run=s step=1 steps=1 evals=3 step_wall=1.0 "
            "ff_evaluate_total=0.3 ff_solve=0.2 ff_maintain=0.05 "
            "builder_build=0.01 repartition=0.01 sort_by_leaf=0.0 "
            "comm_plan_build=0.02 ilist_s3_classify_pairs=0.04 "
            "maint_setup=1 maint_migrate=1 maint_rebalance=1 maint_rebuild=0")
    split = base + "".join(f" {k}=0.01" for k in ILIST_S3_SPLIT)
    lines = []
    for text in (base, split):
        _, rows = run(text + "\n")
        lines.append(state_line(rows["cost"][0], None))
        assert reduce_probe(rows)["s"]["cost"]["step_wall"]["n"] == 1
    assert lines[0] == lines[1], lines
    assert ILIST_S3_SPLIT[0] not in TIMED_MAINT + TIMED_MAINT_O5
    print("self-test: OK")
    return 0


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="*", help="log files to reduce")
    parser.add_argument("--format", choices=("auto", "milestone", "probe"),
                        default="auto")
    parser.add_argument("--bin-steps", type=int, default=250,
                        help="claim B trend bin width in steps (default 250)")
    parser.add_argument("--json", help="also write the reduction as JSON")
    parser.add_argument("--states", action="store_true",
                        help="also print one line per [cost] row of each "
                        "probe run (count-check failures excluded)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the fixture checks and exit")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.logs:
        build_parser().error("no log given")
    if args.bin_steps < 1:
        build_parser().error("--bin-steps must be >= 1")
    out = {}
    try:
        for path in args.logs:
            ms, rows = read_log(path, args.format)
            reps = [launch_report(L, args.bin_steps) for L in ms.launches
                    if L.blocks]
            print(f"\n##### {path}: {ms.total_blocks} breakdown blocks "
                  f"({ms.unattributed_blocks} unattributed under "
                  f"--format probe)")
            for rep in reps:
                print_launch(rep)
            red = reduce_probe(rows)
            print_probe(path, red)
            if args.states:
                print_states(path, rows)
            out[path] = {"breakdown_blocks": ms.total_blocks,
                         "launches": reps, "probe": red}
    except ReduceError as e:
        print(f"fmm_profile_reduce: ERROR: {e}", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"fmm_profile_reduce: ERROR: {e}", file=sys.stderr)
        return 2
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
