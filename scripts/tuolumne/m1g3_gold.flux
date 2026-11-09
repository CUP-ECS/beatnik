#!/bin/bash
# flux: --job-name=beatnik_m1g3_gold
# flux: --nodes=1
# flux: --exclusive
# flux: -t 150m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
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
# M1-G3j (`tasks/milestone1.md`): the milestone's 5000-step Python reference
# gold set. Runs the reference driver with M1-G2's command, `--steps 5000` in
# place of `--steps 600` and `--checkpoint-dir` aside, then runs the in-job
# checks over the output and prints the per-checkpoint step / time / V / F
# table that M1-G3 judges and records.
#
# NO GPU, NO MPI, NO BEATNIK BINARY. The generation is a single-core Python
# run launched with `flux run -N 1 -n 1`; the checks run in this script's body,
# which already executes on the allocated node. The resolver is sourced first,
# as every batch script here must, and targets the DEVELOPMENT env.
#
# Everything the job reads comes from the staged directory
# /p/lustre5/stewartj/beatnik/M1-G3j/stage/, never from the live reference
# checkout or the Beatnik tree:
#   examples/, zmodel3d/   the reference's working trees, uncommitted
#                          `examples/` diff included; the driver puts
#                          `parents[1]` (the stage) on sys.path, so it imports
#                          the staged zmodel3d/
#   beatnik/compare_output.py, beatnik/initial_conditions/gold.npz,
#   beatnik/milestone1-bubble/checkpoint_t00000p557934_step0000600.npz
#                          the comparator, the initial-condition gold and
#                          M1-G2's step-600 checkpoint
#   COMMIT                 the reference's HEAD, `git diff` and
#                          `git status --porcelain -- zmodel3d`, and Beatnik's
#                          HEAD and porcelain count, taken at staging time
# The job refuses to run if the `zmodel3d/` status recorded in COMMIT is not
# empty, and if the output directory already holds checkpoints.
#
# Usage:  flux batch scripts/tuolumne/m1g3_gold.flux
#   Override the stage with M1G3_STAGE=<path> and the output directory with
#   M1G3_OUT=<path> (both parallel-filesystem paths).
# Then read beatnik_m1g3_gold.<jobid>.log in the submitting directory. The
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
    echo "[m1g3] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_stage="${M1G3_STAGE:-/p/lustre5/stewartj/beatnik/M1-G3j/stage}"
_out="${M1G3_OUT:-/p/lustre5/stewartj/beatnik/M1-G3j/out}"
_python=/usr/tce/bin/python3
# The staged trees stay as staged: no __pycache__ is written into them.
export PYTHONDONTWRITEBYTECODE=1

##--------------------------------------------------------------------------##
## Provenance -- read from stage/COMMIT, which was written at staging time.
##--------------------------------------------------------------------------##
if [ ! -f "${_stage}/COMMIT" ]; then
    echo "[m1g3] FAIL: ${_stage}/COMMIT is missing; the stage is incomplete." >&2
    exit 1
fi
beatnik_env_summary
echo "[m1g3] spack env status:"
spack env status 2>&1 | sed 's/^/[m1g3]   /'
echo "[m1g3] stage   = ${_stage}"
echo "[m1g3] stage/COMMIT:"
sed 's/^/[m1g3]   /' "${_stage}/COMMIT"
# The block between the two markers is `git status --porcelain -- zmodel3d`
# of the reference at staging time; it must be empty.
_ref_dirty="$(sed -n '/^reference_zmodel3d_status_begin$/,/^reference_zmodel3d_status_end$/p' \
                  "${_stage}/COMMIT" | sed '1d;$d')"
if ! grep -qx 'reference_zmodel3d_status_end' "${_stage}/COMMIT"; then
    echo "[m1g3] FAIL: stage/COMMIT carries no reference zmodel3d/ status block." >&2
    exit 1
fi
echo "[m1g3] reference zmodel3d/ status (from COMMIT) = '${_ref_dirty}'"
echo "[m1g3] python  = ${_python} ($(${_python} --version 2>&1))"
echo "[m1g3] numpy   = $(${_python} -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "[m1g3] scipy   = $(${_python} -c 'import scipy; print(scipy.__version__)' 2>&1)"
echo "[m1g3] host    = $(hostname)"
echo "[m1g3] nproc (flux run -N 1 -n 1) = $(flux run -N 1 -n 1 nproc 2>&1)"
echo "[m1g3] output  = ${_out}"
echo "[m1g3] submit  = flux batch scripts/tuolumne/m1g3_gold.flux"
if [ -n "${_ref_dirty}" ]; then
    echo "[m1g3] FAIL: the staged reference's zmodel3d/ package was not clean." >&2
    exit 1
fi
mkdir -p "${_out}" || exit 1
if compgen -G "${_out}/*.npz" > /dev/null; then
    echo "[m1g3] FAIL: ${_out} already holds checkpoints; refusing to mix runs." >&2
    exit 1
fi

##--------------------------------------------------------------------------##
## Generate -- M1-G2's command, `--steps 5000` and `--checkpoint-dir` aside.
##--------------------------------------------------------------------------##
cd "${_stage}" || exit 1
_cmd=( "${_python}" examples/run_adaptive_mesh_bubble.py
       --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
       --viscosity-mode laplace-beltrami --br-approximation direct
       --adaptive-dt --dynamic-remesh --isotropic-cleanup
       --source-quadrature vertex
       --steps 5000 --checkpoint-every-steps 10 --no-video --checkpoint-dir "${_out}" )
echo "[m1g3] cwd     = $(pwd)"
echo "[m1g3] command = flux run -N 1 -n 1 ${_cmd[*]}"
_t0=$(date +%s)
flux run -N 1 -n 1 "${_cmd[@]}"
_rc=$?
_dt=$(( $(date +%s) - _t0 ))
echo "[m1g3] generation rc=${_rc} wall=${_dt}s"
if [ "${_rc}" -ne 0 ]; then
    echo "[m1g3] FAIL: the reference driver exited ${_rc}." >&2
    exit "${_rc}"
fi

##--------------------------------------------------------------------------##
## Checks -- what M1-G3's exit criterion judges.
##--------------------------------------------------------------------------##
M1G_OUT="${_out}" M1G_STAGE="${_stage}" M1G_TAG=m1g3 "${_python}" - <<'PYEOF'
import filecmp, glob, os, re, subprocess, sys
import numpy as np

out = os.environ["M1G_OUT"]
stage = os.environ["M1G_STAGE"]
tag = os.environ["M1G_TAG"]
ic = os.path.join(stage, "beatnik/initial_conditions/gold.npz")
cmpo = os.path.join(stage, "beatnik/compare_output.py")
g2_600 = os.path.join(stage, "beatnik/milestone1-bubble/checkpoint_t00000p557934_step0000600.npz")
fail = []

def say(msg):
    print(f"[{tag}] {msg}", flush=True)

files = sorted(glob.glob(os.path.join(out, "*_step???????.npz")))
steps = [int(re.search(r"_step(\d{7})\.npz$", f).group(1)) for f in files]
say(f"numbered checkpoints = {len(files)}, steps {steps[0] if steps else None}..{steps[-1] if steps else None}")
if not files:
    say("CHECK FAIL: no numbered checkpoints")
    sys.exit(1)
# A run that stops early writes the multiples of 10 it reached, then a final
# last-finite checkpoint whose step need not be a multiple of 10.
regular = [s for s in steps if s % 10 == 0]
if regular != list(range(0, 10 * len(regular), 10)) or len(steps) - len(regular) > 1:
    fail.append(f"numbered steps are not a 0,10,... prefix plus at most one final step: {steps}")
if steps[-1] == 5000 and len(steps) == 501:
    say("run reached step 5000: no early stop")
else:
    say(f"EARLY STOP: last numbered checkpoint is step {steps[-1]} ({len(steps)} files); compare-depth ceiling")

latest = os.path.join(out, "checkpoint_latest.npz")
same_bytes = os.path.exists(latest) and filecmp.cmp(latest, files[-1], shallow=False)
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

# The reference is serial and deterministic: step 600 must equal M1-G2's
# step-600 file exactly, or the staged reference is not the one M1-G2 ran.
f600 = [f for f, st in zip(files, steps) if st == 600]
if not f600:
    say("step 600 vs M1-G2 step 600: NO step-600 file (run stopped before step 600)")
    fail.append("no step-600 checkpoint to compare against M1-G2")
else:
    a, b = np.load(f600[0]), np.load(g2_600)
    if sorted(a.files) != sorted(b.files):
        fail.append(f"step-600 key set {sorted(a.files)} != M1-G2 key set {sorted(b.files)}")
        say("step 600 vs M1-G2 step 600: key sets differ")
    else:
        neq = [k for k in b.files if not np.array_equal(a[k], b[k])]
        say(f"step 600 vs M1-G2 {os.path.basename(g2_600)}: np.array_equal on all {len(b.files)} keys: "
            f"{'yes' if not neq else 'NO ' + str(neq)}")
        if neq:
            fail.append(f"step 600 differs from M1-G2's step 600 in {neq}")

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
say("table | step | time | V | F |")
for f, st in zip(files, steps):
    d = np.load(f)
    if sorted(d.files) != keyset:
        fail.append(f"step {st}: key set {sorted(d.files)}")
        continue
    nv, nf = d["vertices"].shape[0], d["faces"].shape[0]
    if d["vertices"].shape[1:] != (3,) or d["faces"].shape[1:] != (3,) or int(d["faces"].max()) >= nv:
        fail.append(f"step {st}: vertices {d['vertices'].shape} faces {d['faces'].shape}")
    if int(d["step"]) != st:
        fail.append(f"step {st}: stored step {int(d['step'])}")
    nonfinite = [k for k in d.files
                 if np.issubdtype(d[k].dtype, np.number) and not np.all(np.isfinite(d[k]))]
    if nonfinite:
        fail.append(f"step {st}: nonfinite {nonfinite}")
    say(f"table | {st} | {float(d['time']):.6f} | {nv} | {nf} |")
last = np.load(files[-1])
say(f"final time = {float(last['time']):.17g} step = {int(last['step'])}")

if fail:
    for x in fail:
        say(f"CHECK FAIL: {x}")
    sys.exit(1)
say("ALL CHECKS PASS")
PYEOF
_crc=$?
echo "[m1g3] checks rc=${_crc}"
exit "${_crc}"
