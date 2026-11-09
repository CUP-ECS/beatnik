#!/bin/bash
# flux: --job-name=beatnik_m1g1_gold
# flux: --nodes=1
# flux: --exclusive
# flux: -t 40m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
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
# M1-G1 (`tasks/milestone1.md`): the fixed-mesh 600-step Python reference gold
# set. Runs the reference driver with the M1-G1 command exactly as the task
# entry gives it, `--checkpoint-dir` aside, then runs the task's exit-criterion
# checks over the output in the same job.
#
# NO GPU, NO MPI, NO BEATNIK BINARY. The generation is a single-core Python
# run launched with `flux run -N 1 -n 1`, as the reference's own
# scripts/run_bubble_L3.flux launches; the checks run in this script's body,
# which already executes on the allocated node. The resolver is still sourced
# first, as every batch script here must, and targets the DEVELOPMENT env.
#
# The reference repository is read, never edited. Its `zmodel3d/` package
# must be clean; this script refuses to run otherwise. The output directory
# must hold no checkpoints yet; this script refuses to mix runs.
#
# Usage:  flux batch scripts/tuolumne/m1g1_gold.flux
#   Override the reference location with ZMODEL3D_REPO=<path> and the output
#   directory with M1G1_OUT=<path> (a parallel-filesystem path).
# Then read beatnik_m1g1_gold.<jobid>.log in the submitting directory. The
# job exits 0 only if the generation and every check pass.
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
    echo "[m1g1] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_ref="${ZMODEL3D_REPO:-${HOME}/research-bridges/zmodel-steve/zmodel3d-amr}"
_out="${M1G1_OUT:-/p/lustre5/stewartj/beatnik/M1-G1}"
_python=/usr/tce/bin/python3

##--------------------------------------------------------------------------##
## Provenance -- a gold set is only reusable if a later session can tell
## which code produced it.
##--------------------------------------------------------------------------##
beatnik_env_summary
echo "[m1g1] spack env status:"
spack env status 2>&1 | sed 's/^/[m1g1]   /'
echo "[m1g1] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1g1] beatnik porcelain status:"
git -C "${BEATNIK_REPO}" status --porcelain 2>&1 | sed 's/^/[m1g1]   /'
echo "[m1g1] reference repo   = ${_ref}"
echo "[m1g1] reference commit = $(git -C "${_ref}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1g1] reference git diff:"
git -C "${_ref}" diff 2>&1 | sed 's/^/[m1g1]   /'
_ref_dirty="$(git -C "${_ref}" status --porcelain -- zmodel3d 2>&1)"
echo "[m1g1] reference zmodel3d/ status = '${_ref_dirty}'"
echo "[m1g1] python  = ${_python} ($(${_python} --version 2>&1))"
echo "[m1g1] numpy   = $(${_python} -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "[m1g1] scipy   = $(${_python} -c 'import scipy; print(scipy.__version__)' 2>&1)"
echo "[m1g1] host    = $(hostname)"
echo "[m1g1] nproc (flux run -N 1 -n 1) = $(flux run -N 1 -n 1 nproc 2>&1)"
echo "[m1g1] output  = ${_out}"
echo "[m1g1] submit  = flux batch scripts/tuolumne/m1g1_gold.flux"
if [ -n "${_ref_dirty}" ]; then
    echo "[m1g1] FAIL: the reference's zmodel3d/ package is not clean." >&2
    exit 1
fi
mkdir -p "${_out}" || exit 1
if compgen -G "${_out}/*.npz" > /dev/null; then
    echo "[m1g1] FAIL: ${_out} already holds checkpoints; refusing to mix runs." >&2
    exit 1
fi

##--------------------------------------------------------------------------##
## Generate -- the M1-G1 command verbatim, `--checkpoint-dir` aside.
##--------------------------------------------------------------------------##
cd "${_ref}" || exit 1
_cmd=( "${_python}" examples/run_adaptive_mesh_bubble.py --steps 600
       --source-quadrature vertex --br-approximation direct
       --no-dynamic-remesh --refine-every 0
       --checkpoint-every-steps 10 --no-video --checkpoint-dir "${_out}" )
echo "[m1g1] cwd     = $(pwd)"
echo "[m1g1] command = flux run -N 1 -n 1 ${_cmd[*]}"
_t0=$(date +%s)
flux run -N 1 -n 1 "${_cmd[@]}"
_rc=$?
_dt=$(( $(date +%s) - _t0 ))
echo "[m1g1] generation rc=${_rc} wall=${_dt}s"
cd "${BEATNIK_REPO}" || exit 1
if [ "${_rc}" -ne 0 ]; then
    echo "[m1g1] FAIL: the reference driver exited ${_rc}." >&2
    exit "${_rc}"
fi

##--------------------------------------------------------------------------##
## Exit-criterion checks
##--------------------------------------------------------------------------##
M1G_OUT="${_out}" M1G_TAG=m1g1 "${_python}" - <<'PYEOF'
import filecmp, glob, os, re, subprocess, sys
import numpy as np

out = os.environ["M1G_OUT"]
tag = os.environ["M1G_TAG"]
repo = os.environ["BEATNIK_REPO"]
ic = os.path.join(repo, "tests/regression_tests/initial_conditions/gold.npz")
cmpo = os.path.join(repo, "tests/regression_tests/compare_output.py")
fail = []

def say(msg):
    print(f"[{tag}] {msg}", flush=True)

files = sorted(glob.glob(os.path.join(out, "*_step???????.npz")))
steps = [int(re.search(r"_step(\d{7})\.npz$", f).group(1)) for f in files]
say(f"numbered checkpoints = {len(files)}, steps {steps[0] if steps else None}..{steps[-1] if steps else None}")
if steps != list(range(0, 601, 10)):
    fail.append(f"numbered steps are not 0,10,...,600: {steps}")

latest = os.path.join(out, "checkpoint_latest.npz")
same_bytes = filecmp.cmp(latest, files[-1], shallow=False)
say(f"cmp checkpoint_latest.npz {os.path.basename(files[-1])}: {'identical' if same_bytes else 'DIFFER'}")
if not same_bytes:
    fail.append("checkpoint_latest.npz is not byte-identical to the last numbered file")

g = np.load(ic)
s0 = np.load(files[0])
say(f"step-0 keys = {sorted(s0.files)}")
if sorted(s0.files) != sorted(g.files) or len(g.files) != 9:
    fail.append(f"step-0 key set {sorted(s0.files)} != IC key set {sorted(g.files)}")
else:
    neq = [k for k in g.files if not np.array_equal(s0[k], g[k])]
    say(f"step 0 vs initial_conditions/gold.npz: np.array_equal on all 9 keys: {'yes' if not neq else 'NO ' + str(neq)}")
    if neq:
        fail.append(f"step 0 differs from IC gold in {neq}")

def run_compare(a, b, label):
    r = subprocess.run([sys.executable, cmpo, a, b, "--rtol", "1e-12", "--atol", "1e-14"],
                       capture_output=True, text=True)
    m = [l for l in (r.stdout + r.stderr).splitlines() if "matching" in l]
    say(f"{label}: compare_output.py rc={r.returncode}; {m[0].strip() if m else 'no matching line'}")
    if r.returncode != 0:
        print(r.stdout + r.stderr, flush=True)
        fail.append(f"{label} compare_output.py exited {r.returncode}")
    return m[0] if m else ""

run_compare(files[0], ic, "step0 vs IC")
mline = run_compare(files[-1], files[-1], f"self-compare {os.path.basename(files[-1])}")
am = re.search(r"ambiguous cpp=(\d+) gold=(\d+)", mline)
if not am or am.group(1) != "0" or am.group(2) != "0":
    fail.append(f"last-file self-compare is not zero-ambiguous: {mline.strip()}")

keyset = sorted(g.files)
for f, st in zip(files, steps):
    d = np.load(f)
    if sorted(d.files) != keyset:
        fail.append(f"step {st}: key set {sorted(d.files)}")
    if d["vertices"].shape != (162, 3) or d["faces"].shape != (320, 3):
        fail.append(f"step {st}: vertices {d['vertices'].shape} faces {d['faces'].shape}")
    if int(d["step"]) != st:
        fail.append(f"step {st}: stored step {int(d['step'])}")
    for k in ("vertices", "potential", "remesh_material_position"):
        if not np.all(np.isfinite(d[k])):
            fail.append(f"step {st}: nonfinite {k}")
say(f"shapes checked in {len(files)} files: vertices (162, 3), faces (320, 3) expected")
last = np.load(files[-1])
say(f"final time = {float(last['time']):.17g} step = {int(last['step'])}")

if fail:
    for x in fail:
        say(f"CHECK FAIL: {x}")
    sys.exit(1)
say("ALL CHECKS PASS")
PYEOF
_crc=$?
echo "[m1g1] checks rc=${_crc}"
exit "${_crc}"
