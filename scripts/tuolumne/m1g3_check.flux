#!/bin/bash
# flux: --job-name=beatnik_m1g3_check
# flux: --nodes=1
# flux: --exclusive
# flux: -t 10m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
# flux: --bank=unm
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
# M1-G3 (`tasks/milestone1.md`): the verification job for the milestone's
# steps 0-1610 Python reference gold set, which M1-G3j's job wrote to
# /p/lustre5/stewartj/beatnik/M1-G3j/out/. It runs two checks:
#   1. a self-compare of the step-1610 file
#      (checkpoint_t00001p238122_step0001610.npz) with the staged
#      compare_output.py at `--rtol 1e-12 --atol 1e-14`, printing its rc and
#      its `matching` line;
#   2. steps 0..1610 are exactly 162 numbered files, steps 0, 10, ..., 1610,
#      each with the nine keys and every numeric field finite.
# It also prints the step / time / V / F table of those 162 files.
#
# NO GPU, NO MPI, NO BEATNIK BINARY. The comparator is single-process Python,
# run in this script's body on the allocated node. The resolver is sourced
# first, as every batch script here must, and targets the DEVELOPMENT env.
#
# The job reads the stage /p/lustre5/stewartj/beatnik/M1-G3j/stage/ and the
# output directory and writes nothing into either: Python runs with
# PYTHONDONTWRITEBYTECODE=1 and every file is opened read-only.
#
# Usage:  flux batch scripts/tuolumne/m1g3_check.flux
#   Override the stage with M1G3_STAGE=<path> and the output directory with
#   M1G3_OUT=<path>.
# Then read beatnik_m1g3_check.<jobid>.log in the submitting directory. The
# job exits 0 only if both checks pass.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

# Pin the repo root. `flux batch` copies the script into a per-job spool
# directory before running it, so BASH_SOURCE points at /var/tmp/... and cannot
# find the checkout. It does preserve the submitting working directory, so walk
# up from PWD as the fallback; BASH_SOURCE still covers a direct `bash` run.
beatnik_find_repo() {
    local _d
    for _d in "$(dirname "${BASH_SOURCE[0]}")/../.." "${PWD}"; do
        _d="$(cd "${_d}" 2>/dev/null && pwd)" || continue
        while [ -n "${_d}" ] && [ "${_d}" != "/" ]; do
            if [ -f "${_d}/scripts/lib/beatnik_env.sh" ]; then
                printf '%s\n' "${_d}"
                return 0
            fi
            _d="$(dirname "${_d}")"
        done
    done
    return 1
}
export BEATNIK_REPO="${BEATNIK_REPO:-$(beatnik_find_repo)}"
if [ -z "${BEATNIK_REPO}" ]; then
    echo "[m1g3c] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_stage="${M1G3_STAGE:-/p/lustre5/stewartj/beatnik/M1-G3j/stage}"
_out="${M1G3_OUT:-/p/lustre5/stewartj/beatnik/M1-G3j/out}"
_python=/usr/tce/bin/python3
# The stage and the output stay as written: no __pycache__ goes into them.
export PYTHONDONTWRITEBYTECODE=1

##--------------------------------------------------------------------------##
## Provenance
##--------------------------------------------------------------------------##
if [ ! -f "${_stage}/COMMIT" ]; then
    echo "[m1g3c] FAIL: ${_stage}/COMMIT is missing." >&2
    exit 1
fi
beatnik_env_summary
echo "[m1g3c] stage   = ${_stage}"
echo "[m1g3c] stage/COMMIT:"
sed 's/^/[m1g3c]   /' "${_stage}/COMMIT"
echo "[m1g3c] python  = ${_python} ($(${_python} --version 2>&1))"
echo "[m1g3c] numpy   = $(${_python} -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "[m1g3c] host    = $(hostname)"
echo "[m1g3c] output  = ${_out}"
echo "[m1g3c] submit  = flux batch scripts/tuolumne/m1g3_check.flux"

cd /tmp || exit 1

##--------------------------------------------------------------------------##
## Checks
##--------------------------------------------------------------------------##
M1G_OUT="${_out}" M1G_STAGE="${_stage}" "${_python}" - <<'PYEOF'
import glob, os, re, subprocess, sys
import numpy as np

out = os.environ["M1G_OUT"]
stage = os.environ["M1G_STAGE"]
cmpo = os.path.join(stage, "beatnik/compare_output.py")
last = 1610
f1610 = os.path.join(out, "checkpoint_t00001p238122_step0001610.npz")
fail = []

def say(msg):
    print(f"[m1g3c] {msg}", flush=True)

# 1. Self-compare of the step-1610 file.
cmd = [sys.executable, cmpo, f1610, f1610, "--rtol", "1e-12", "--atol", "1e-14"]
say("self-compare command = " + " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout + r.stderr, flush=True)
m = [l for l in (r.stdout + r.stderr).splitlines() if l.startswith("matching")]
say(f"self-compare step 1610: rc={r.returncode}; {m[0].strip() if m else 'no matching line'}")
if r.returncode != 0:
    fail.append(f"step-1610 self-compare exited {r.returncode}")
am = re.search(r"ambiguous cpp=(\d+) gold=(\d+)", m[0] if m else "")
if not am or am.group(1) != "0" or am.group(2) != "0":
    fail.append("step-1610 self-compare is not zero-ambiguous")

# 2. Steps 0..1610: exactly 162 files, nine keys, every numeric field finite.
files = sorted(glob.glob(os.path.join(out, "*_step???????.npz")))
pairs = [(int(re.search(r"_step(\d{7})\.npz$", f).group(1)), f) for f in files]
gold = [(s, f) for s, f in pairs if s <= last]
steps = [s for s, _ in gold]
say(f"numbered checkpoints in out/ = {len(pairs)}; with step <= {last}: {len(gold)}")
if steps != list(range(0, last + 1, 10)):
    fail.append(f"steps <= {last} are not exactly 0,10,...,{last}: {steps}")
keyset = None
say("table | step | time | V | F |")
nonfinite_any = 0
for s, f in gold:
    with np.load(f) as d:
        keys = sorted(d.files)
        if keyset is None:
            keyset = keys
            say(f"key set = {keyset}")
        if keys != keyset or len(keys) != 9:
            fail.append(f"step {s}: key set {keys}")
        if int(d["step"]) != s:
            fail.append(f"step {s}: stored step {int(d['step'])}")
        nonfinite = [k for k in keys
                     if np.issubdtype(d[k].dtype, np.number) and not np.all(np.isfinite(d[k]))]
        if nonfinite:
            nonfinite_any += 1
            fail.append(f"step {s}: nonfinite {nonfinite}")
        say(f"table | {s} | {float(d['time']):.6f} | {d['vertices'].shape[0]} | {d['faces'].shape[0]} |")
        if s == last:
            say(f"step {last} time = {float(d['time']):.17g}")
say(f"finiteness: {len(gold)} files checked, {nonfinite_any} with a nonfinite numeric field")

if fail:
    for x in fail:
        say(f"CHECK FAIL: {x}")
    sys.exit(1)
say("ALL CHECKS PASS")
PYEOF
_crc=$?
echo "[m1g3c] checks rc=${_crc}"
exit "${_crc}"
