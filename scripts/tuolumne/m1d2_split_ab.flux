#!/bin/bash
# flux: --job-name=beatnik_m1d2
# flux: --nodes=1
# flux: --exclusive
# flux: -t 20m
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
# M1-D2 (`tasks/milestone1.md`): where, in steps 1-10, Beatnik and the Python
# reference part on the milestone-1 configuration, and whether the parting is
# a selection difference between two valid edit rules (reading a) or a
# difference in the rules themselves (reading b).
#
# Four parts, all in this one job:
#   1. REFERENCE  M1-G2's command with `--steps 10 --checkpoint-every-steps 1`,
#                 run through scripts/tuolumne/m1d2_refhooks.py, which prints
#                 each step's dt and pass counts (candidates before the cap,
#                 accepted edits, cleanup edges changed). 11 checkpoints.
#   2. BEATNIK    Beatnik_Test_Milestone1Run_MPI_{SERIAL,HIP} 10 1 50 at ranks
#                 1 and 4: the same configuration, a checkpoint every step.
#   3. COMPARE    compare_output.py of every Beatnik step 0-10 against the
#                 reference step 0-10 at its default tolerances, and an exact
#                 check that this run's step-10 file equals M1-G2's.
#   4. PROBE      one reference remesh step (dynamic_remesh_arrays +
#                 isotropic_cleanup, the run's exact parameters) applied to
#                 each code's checkpoints, steps 0-10 here and selected steps
#                 of the M1-G2 / M1-T2j 600-step series: how much editing the
#                 REFERENCE'S rules want on each code's mesh.
#
# ENVIRONMENT AND QUEUE: the DEV spack env, `pdebug`. The binaries are the dev
# view's; their sha256 is echoed. The reference repository is read, never
# edited, and its zmodel3d/ package must be clean.
#
# Usage, from the repo root:  flux batch scripts/tuolumne/m1d2_split_ab.flux [all|diag]
#   `all` (the default) runs parts 1-4 into the output root M1D2_OUT (default
#   /p/lustre5/stewartj/beatnik/M1-D2), which must hold no output yet.
#   `diag` runs part 2 alone into <root>/diag, which must not exist yet: the
#   Beatnik launches again, for their per-step `[m1rem]` selection counts.
#   Read beatnik_m1d2.<jobid>.log in the submitting
#   directory; wait with `flux job status`, never `flux job attach`.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source. Tuolumne needs none.

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
    echo "[m1d2] FAIL: cannot locate the Beatnik checkout." >&2
    exit 1
fi
if [ "${BEATNIK_USE_PROD:-}" = "1" ]; then
    echo "[m1d2] FAIL: BEATNIK_USE_PROD=1; M1-D2 measures the DEV env." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

_ref="${ZMODEL3D_REPO:-${HOME}/research-bridges/zmodel-steve/zmodel3d-amr}"
_out="${M1D2_OUT:-/p/lustre5/stewartj/beatnik/M1-D2}"
_python=/usr/tce/bin/python3
_hooks="${BEATNIK_REPO}/scripts/tuolumne/m1d2_refhooks.py"
_cmp="${BEATNIK_REPO}/tests/regression_tests/compare_output.py"
_g2="${BEATNIK_REPO}/tests/regression_tests/milestone1-bubble/gold"
_t2j="/p/lustre5/stewartj/beatnik/M1-T2j/out/Beatnik_Test_Milestone1Bubble_MPI_HIP_np1/beatnik_milestone1/HIP_np1_compare"
_steps=10
_part="${1:-all}"
case "${_part}" in
    all)  _bk_root="${_out}/beatnik" ;;
    diag) _bk_root="${_out}/diag" ;;
    *)    echo "[m1d2] FAIL: unknown part '${_part}' (all|diag)." >&2; exit 1 ;;
esac

##--------------------------------------------------------------------------##
## Provenance
##--------------------------------------------------------------------------##
echo "=========================== PROVENANCE ==========================="
beatnik_env_summary
echo "[m1d2] commit: $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1d2] git status --short:"
git -C "${BEATNIK_REPO}" status --short 2>&1 | sed 's/^/[m1d2]   /'
for _b in SERIAL HIP; do
    _e="$(beatnik_exe "Beatnik_Test_Milestone1Run_MPI_${_b}")" || exit 1
    echo "[m1d2] exe ${_b} = ${_e} sha256=$(sha256sum "${_e}" | cut -c1-64)"
done
echo "[m1d2] reference repo   = ${_ref}"
echo "[m1d2] reference commit = $(git -C "${_ref}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[m1d2] reference git diff:"
git -C "${_ref}" diff 2>&1 | sed 's/^/[m1d2]   /'
_ref_dirty="$(git -C "${_ref}" status --porcelain -- zmodel3d 2>&1)"
echo "[m1d2] reference zmodel3d/ status = '${_ref_dirty}'"
echo "[m1d2] python = ${_python} ($(${_python} --version 2>&1)), numpy $(${_python} -c 'import numpy; print(numpy.__version__)' 2>&1)"
echo "[m1d2] host = $(hostname)  date = $(date -Is)"
echo "[m1d2] output root = ${_out}  part = ${_part}  beatnik root = ${_bk_root}"
echo "[m1d2] submit = flux batch scripts/tuolumne/m1d2_split_ab.flux ${_part}"
echo "=================================================================="
if [ -n "${_ref_dirty}" ]; then
    echo "[m1d2] FAIL: the reference's zmodel3d/ package is not clean." >&2
    exit 1
fi
if [ "${_part}" = all ]; then
    if [ -e "${_out}" ] && [ -n "$(ls -A "${_out}" 2>/dev/null)" ]; then
        echo "[m1d2] FAIL: ${_out} is not empty; refusing to mix runs." >&2
        exit 1
    fi
    mkdir -p "${_out}/gold" "${_out}/beatnik" "${_out}/analysis" || exit 1
else
    if [ -e "${_bk_root}" ]; then
        echo "[m1d2] FAIL: ${_bk_root} exists; refusing to mix runs." >&2
        exit 1
    fi
    mkdir -p "${_bk_root}" || exit 1
fi

_rc=0
_t_job0="$(date +%s)"

##--------------------------------------------------------------------------##
## 1. Reference, a checkpoint every step through step 10
##--------------------------------------------------------------------------##
if [ "${_part}" = all ]; then
_refcmd=( --A 0.3 --g 1.0 --mu 0.002 --eps 0.025
          --viscosity-mode laplace-beltrami --br-approximation direct
          --adaptive-dt --dynamic-remesh --isotropic-cleanup
          --source-quadrature vertex
          --steps "${_steps}" --checkpoint-every-steps 1 --no-video
          --checkpoint-dir "${_out}/gold" )
echo "[m1d2] === reference: flux run -N 1 -n 1 ${_python} ${_hooks} run ${_ref} ${_out}/ref_params.json -- ${_refcmd[*]}"
_t0="$(date +%s)"
( cd "${_ref}" && flux run -N 1 -n 1 "${_python}" "${_hooks}" run "${_ref}" \
      "${_out}/ref_params.json" -- "${_refcmd[@]}" )
_r=$?
echo "[m1d2] reference rc=${_r} wall=$(( $(date +%s) - _t0 ))s checkpoints=$(ls "${_out}"/gold/*_step???????.npz 2>/dev/null | wc -l)"
if [ "${_r}" -ne 0 ]; then
    echo "[m1d2] FAIL: the reference run exited ${_r}." >&2
    exit 1
fi
fi

##--------------------------------------------------------------------------##
## 2. Beatnik, a checkpoint every step through step 10
##--------------------------------------------------------------------------##
SWEEP="
HIP    1
HIP    4
SERIAL 1
SERIAL 4
"
while IFS= read -r _row <&3; do
    [ -n "${_row}" ] || continue
    # shellcheck disable=SC2086
    set -- ${_row}
    _backend="$1"
    _np="$2"
    _tag="${_backend}_np${_np}"
    _exe="$(beatnik_exe "Beatnik_Test_Milestone1Run_MPI_${_backend}")" || { _rc=1; continue; }
    export BEATNIK_TEST_SCRATCH="${_bk_root}/${_tag}"
    mkdir -p "${BEATNIK_TEST_SCRATCH}" || { _rc=1; continue; }
    echo "[m1d2] === beatnik ${_tag}: ${_exe} ${_steps} 1 50"
    _t0="$(date +%s)"
    flux run --ntasks="${_np}" --nodes=1 --exclusive --gpus-per-task=1 \
        --cores-per-task=24 --setopt=mpibind=verbose:1 \
        "${_exe}" "${_steps}" 1 50 > "${BEATNIK_TEST_SCRATCH}/launch.log" 2>&1
    _r=$?
    grep -E '^\[m1run\] (step=.* rank=0 |FINAL rank=0)|^\[m1rem\]|\[(PASS|FAIL)\]' "${BEATNIK_TEST_SCRATCH}/launch.log" \
        | sed "s/^/[m1d2] ${_tag} /"
    echo "[m1d2] LAUNCH_WALL ${_tag} rc=${_r} wall=$(( $(date +%s) - _t0 ))s" \
         "checkpoints=$(find "${BEATNIK_TEST_SCRATCH}" -name '*_step*.h5' | wc -l)"
    [ "${_r}" -eq 0 ] || _rc=1
done 3<<EOF
${SWEEP}
EOF

if [ "${_part}" = diag ]; then
    echo "[m1d2] total=$(( $(date +%s) - _t_job0 ))s rc=${_rc}"
    exit "${_rc}"
fi

##--------------------------------------------------------------------------##
## 3. Field comparison, step by step, and the non-perturbation check
##--------------------------------------------------------------------------##
M1D2_OUT="${_out}" M1D2_CMP="${_cmp}" M1D2_G2="${_g2}" "${_python}" - <<'PYEOF' || _rc=1
import glob, os, re, subprocess, sys
import numpy as np
out, cmpo, g2 = os.environ["M1D2_OUT"], os.environ["M1D2_CMP"], os.environ["M1D2_G2"]
ok = True
def one(pattern):
    f = glob.glob(pattern)
    return f[0] if len(f) == 1 else None
mine = one(f"{out}/gold/*_step0000010.npz"); theirs = one(f"{g2}/*_step0000010.npz")
a, b = np.load(mine), np.load(theirs)
neq = [k for k in b.files if k not in a.files or not np.array_equal(a[k], b[k])]
print(f"[m1d2cmp] instrumented step 10 vs M1-G2 step 10: np.array_equal on {len(b.files)} keys: "
      f"{'yes' if not neq else 'NO ' + str(neq)}", flush=True)
ok &= not neq
for tag in ("HIP_np1", "HIP_np4", "SERIAL_np1", "SERIAL_np4"):
    for k in range(0, 11):
        hits = glob.glob(f"{out}/beatnik/{tag}/**/*_step{k:07d}.h5", recursive=True)
        cpp = hits[0] if len(hits) == 1 else None
        gold = one(f"{out}/gold/*_step{k:07d}.npz")
        if cpp is None or gold is None:
            print(f"[m1d2cmp] {tag} step {k}: MISSING cpp={cpp} gold={gold}", flush=True); ok = False; continue
        r = subprocess.run([sys.executable, cmpo, cpp, gold], capture_output=True, text=True)
        txt = r.stdout + r.stderr
        open(f"{out}/analysis/cmp_{tag}_step{k:02d}.txt", "w").write(txt)
        keep = [l.strip() for l in txt.splitlines()
                if re.search(r"structure:|FAIL|matching|faces:|max\|e\||RESULT", l)]
        print(f"[m1d2cmp] {tag} step {k} rc={r.returncode} | " + " | ".join(keep), flush=True)
        if r.returncode not in (0, 1):
            ok = False
sys.exit(0 if ok else 1)
PYEOF

##--------------------------------------------------------------------------##
## 4. Probe: the reference's rules applied to each code's meshes
##--------------------------------------------------------------------------##
_items=()
for _k in $(seq 0 "${_steps}"); do
    _s="$(printf '%07d' "${_k}")"
    _items+=( "gold10=$(ls "${_out}"/gold/*_step${_s}.npz)" )
    for _tag in HIP_np1 SERIAL_np4; do
        _items+=( "bk10_${_tag}=$(find "${_out}/beatnik/${_tag}" -name "*_step${_s}.h5" | head -1)" )
    done
done
for _k in 10 20 50 100 200 300 400 500 600; do
    _s="$(printf '%07d' "${_k}")"
    _items+=( "gold600=$(ls "${_g2}"/*_step${_s}.npz)" )
    _items+=( "bk600_HIP_np1=$(ls "${_t2j}"/*_step${_s}.h5)" )
done
echo "[m1d2] === probe: ${#_items[@]} states"
_t0="$(date +%s)"
"${_python}" "${_hooks}" probe "${_ref}" "${_out}/ref_params.json" "${_items[@]}" \
    | tee "${_out}/analysis/probe.txt" || _rc=1
echo "[m1d2] probe wall=$(( $(date +%s) - _t0 ))s"

echo "[m1d2] total=$(( $(date +%s) - _t_job0 ))s rc=${_rc}"
exit "${_rc}"
