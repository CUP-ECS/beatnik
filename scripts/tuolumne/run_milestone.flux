#!/bin/bash
# flux: --job-name=beatnik_milestone
# flux: --nodes=1
# flux: --exclusive
# flux: -t 652m
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pbatch
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
#
# THE MILESTONE TIER runner for tuolumne. NOT the ship gate.
#
# Tier definition (single-sourced -- must stay identical to the `milestone`
# rows in CLAUDE.md "Minimum test set", tests/CMakeLists.txt and
# docs/testing.md):
#
#     tier label : milestone
#     backends   : SERIAL + HIP
#     ranks      : 1 4
#
# This tier holds long end-to-end runs against multi-thousand-step reference
# gold sets. It is deliberately OUTSIDE the 60-launch gate -- a 2000-step run
# in front of every change is not a gate, it is a stall -- so a green result
# here is not a substitute for run_regression_minset.flux and a green gate is
# not a substitute for this. Run it on demand.
#
# Run the tier as EIGHT jobs, one per (member, backend), each with its own -t:
#
#     scripts/tuolumne/submit_milestone_split.sh      (T9b; see its header)
#
# and read each beatnik_milestone_<member-short>_<BACKEND>.<jobid>.log it
# writes to the repo root. That is how the tier is run. A bare
# `flux batch scripts/tuolumne/run_milestone.flux` still runs all sixteen
# launches as ONE job (beatnik_milestone.<jobid>.log), but that serializes an
# 8.7 h sum behind one allocation; it is kept only as a fallback.
#
# Filters, all optional, all space-separated, all read from the environment
# (flux batch copies the submitting environment into the job):
#
#   BEATNIK_MILESTONE_MEMBERS   member stems, e.g. Beatnik_Test_Milestone0Fmm.
#                               Matched EXACTLY against manifest field 1 with
#                               its _MPI_<BACKEND> suffix stripped, so that stem
#                               does not also select Beatnik_Test_Milestone0FmmL4.
#                               Unset means every member. A stem that matches
#                               nothing is a FAIL, never a silent skip.
#   BEATNIK_MILESTONE_BACKENDS  default `SERIAL HIP`.
#   BEATNIK_MILESTONE_RANKS     default `1 4`.
#
# Every launch prints `[milestone] <PASS|FAIL> <target> np=<N> in <S>s`, and the
# job ends `[milestone] SUMMARY: <PASS|FAIL> (<passed>/<run> launches)`. A
# job killed at its walltime prints no SUMMARY at all, so its absence is a
# finding.
#
# This file is run_regression_minset.flux's structure with a different label,
# manifest name and rank list. It is a COPY on purpose: the gate script is
# single-sourced against CLAUDE.md's gate definition and must keep saying
# `regression` x ranks 1-6, so generalizing it in place would put this tier in
# a position to change what the gate means.
#
# --nodes=1 covers the 4-rank case at tuolumne's 4-ranks-per-node.
#
# THE WALLTIME IS MEASURED, NOT GUESSED. The frozen pair alone fit pdebug's
# 60m (M0-T3: 37.25 min measured). T6 added the two FMM members, which moved
# the tier to pbatch, and the first run carrying them (T6, job f3azynKNQFCb,
# 1440m = pbatch's ceiling) measured 8.685 h. T9b then ran the tier as eight
# per-(member, backend) jobs through submit_milestone_split.sh, all green
# (2026-10-09, dev env, beatnik 0411c47 + working tree, canopy bd10c8f). The
# measured job walls, each holding its np1 + np4 launches:
#
#   member              SERIAL                     HIP
#   Milestone0Frozen      236 s  (129 + 86)          144 s  (51 + 72)
#   Milestone0FrozenL4   1781 s  (1338 + 419)        168 s  (61 + 86)
#   Milestone0Fmm        9581 s  (2980 + 6578)       613 s  (287 + 304)
#   Milestone0FmmL4     15130 s  (6377 + 8730)      3614 s  (2089 + 1504)
#
# They sum to 31269 s (8.69 h), which is this script's cost as ONE job, and
# -t 652m is about 1.25x that. Split, the tier's wall-clock is the slowest
# job, level-4 FMM SERIAL at 4.2 h (4 h 15 min submit-to-last-exit in T9b),
# and submit_milestone_split.sh carries each job's own -t at about 1.5x its
# wall. The tier is run split; this whole-tier -t exists only so the bare
# one-job fallback is not killed at the wall.
#
# Two things the walls show that per-step extrapolation did not:
#
#   1. SERIAL np4 is SLOWER than np1 for both FMM members (level 3: 2.2x,
#      level 4: 1.37x), while HIP np4 is no slower at level 3 and 1.4x faster
#      at level 4. 642 or 2562 particles over 4 SERIAL ranks is overhead-bound.
#   2. The FMM per-step cost grows along the trajectory (T5: 0.434 -> 1.189
#      s/step), so a rate taken from a short probe under-predicts (T5 measured
#      2.7x). Re-time -t only from a full run, never from a probe.
#
# A job killed at its walltime prints no SUMMARY line, and M0-R8/R9 is exactly
# the mode where a truncated run reads as a shorter pass: check the exit state
# and the 2/2 launch count before reading any green.
############################################################################

set -u

# Any `module load` belongs HERE, before the resolver source, so the resolver
# and the spack env see the final module state. Tuolumne needs none today.

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
    echo "[milestone] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

##--------------------------------------------------------------------------##
## Provenance -- a walltime or a pass in the progress log is only reusable if a
## later session can tell which toolchain and commit produced it. Echoed before
## any work. Copied from t9a_l4_member.flux.
##--------------------------------------------------------------------------##
_canopy_src="${BEATNIK_REPO}/../canopy"
beatnik_env_summary
echo "[milestone] spack env status:"
spack env status 2>&1 | sed 's/^/[milestone]   /'
echo "[milestone] commit  = $(git -C "${BEATNIK_REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[milestone] dirty   = $(git -C "${BEATNIK_REPO}" status --porcelain 2>/dev/null | wc -l) file(s) modified"
echo "[milestone] canopy branch = $(git -C "${_canopy_src}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "[milestone] canopy commit = $(git -C "${_canopy_src}" rev-parse HEAD 2>/dev/null || echo unknown)"
echo "[milestone] canopy dirty  = $(git -C "${_canopy_src}" status --porcelain --untracked-files=no 2>/dev/null | wc -l) file(s) modified"
echo "[milestone] submit  = BEATNIK_MILESTONE_MEMBERS='${BEATNIK_MILESTONE_MEMBERS-<unset>}'" \
     "BEATNIK_MILESTONE_BACKENDS='${BEATNIK_MILESTONE_BACKENDS-<unset>}'" \
     "flux batch scripts/tuolumne/run_milestone.flux" \
     "(job $(flux getattr jobid 2>/dev/null || echo "${FLUX_JOB_ID:-<none>}"))"
echo "[milestone] canopy variants:"
spack find --variants canopy 2>&1 | grep -i canopy | sed 's/^/[milestone]   /'
echo "[milestone] beatnik spec (compiler after the %):"
spack find --variants beatnik 2>&1 | grep -i beatnik | sed 's/^/[milestone]   /'

##--------------------------------------------------------------------------##
## Tier parameters
##--------------------------------------------------------------------------##
BEATNIK_MILESTONE_LABEL="milestone"
BEATNIK_MILESTONE_MEMBERS="${BEATNIK_MILESTONE_MEMBERS:-}"
BEATNIK_MILESTONE_BACKENDS="${BEATNIK_MILESTONE_BACKENDS:-SERIAL HIP}"
BEATNIK_MILESTONE_RANKS="${BEATNIK_MILESTONE_RANKS:-1 4}"

# A stem is an identifier; anything else would be read as a pattern by ctest -R
# in tree mode and could select more than it names.
for _m in ${BEATNIK_MILESTONE_MEMBERS}; do
    case "${_m}" in
        *[!A-Za-z0-9_]*)
            echo "[milestone] FAIL: member stem '${_m}' is not an identifier." >&2
            echo "[milestone] SUMMARY: FAIL (0/0 launches)"
            exit 1 ;;
    esac
done

# Parent of the per-test I/O directories. MUST be on a PARALLEL filesystem: the
# checkpoints go through MPI-IO and a node-local scratch fails every launch that
# spans more than one node (CLAUDE.md "Minimum test set").
BEATNIK_MILESTONE_SCRATCH_ROOT="${BEATNIK_MILESTONE_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/milestone0}"

echo "[milestone] label=${BEATNIK_MILESTONE_LABEL}" \
     "members='${BEATNIK_MILESTONE_MEMBERS:-<all>}'" \
     "backends='${BEATNIK_MILESTONE_BACKENDS}'" \
     "ranks='${BEATNIK_MILESTONE_RANKS}'"

_milestone_rc=0
_launches_run=0
_launches_passed=0

##--------------------------------------------------------------------------##
## manual / tree mode: ctest inside the build directory
##--------------------------------------------------------------------------##
# The harness already registered one ctest case per (backend, rank), so
# `-L <label> -R <backend>` selects the whole rank sweep. A member filter
# becomes an anchored `-R` on the registered name `<stem>_MPI_<BACKEND>_np_<N>`,
# and `--no-tests=error` then makes a stem that matches nothing fail loudly
# rather than run zero tests and pass.
if [ "${BEATNIK_BIN_MODE}" = "tree" ]; then
    if [ ! -d "${BEATNIK_BUILD_DIR}" ]; then
        echo "[milestone] FAIL: build dir ${BEATNIK_BUILD_DIR} does not exist." >&2
        exit 1
    fi
    export BEATNIK_TEST_SCRATCH="${BEATNIK_TEST_SCRATCH:-${BEATNIK_MILESTONE_SCRATCH_ROOT}/ctest}"
    rm -rf "${BEATNIK_TEST_SCRATCH}"
    mkdir -p "${BEATNIK_TEST_SCRATCH}" || {
        echo "[milestone] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
        exit 1
    }
    echo "[milestone] scratch = ${BEATNIK_TEST_SCRATCH}"
    for _backend in ${BEATNIK_MILESTONE_BACKENDS}; do
        if [ -n "${BEATNIK_MILESTONE_MEMBERS}" ]; then
            _re="^($(echo ${BEATNIK_MILESTONE_MEMBERS} | tr ' ' '|'))_MPI_${_backend}_np_"
            _no_tests=error
        else
            _re="${_backend}"
            _no_tests=ignore
        fi
        echo "[milestone] ctest -L ${BEATNIK_MILESTONE_LABEL} -R ${_re}"
        ( cd "${BEATNIK_BUILD_DIR}" &&
          ctest --output-on-failure --no-tests="${_no_tests}" \
                -L "${BEATNIK_MILESTONE_LABEL}" -R "${_re}" ) || _milestone_rc=1
    done

##--------------------------------------------------------------------------##
## spack / installed mode: loop the installed milestone binaries over the ranks
##--------------------------------------------------------------------------##
# No build tree exists, so there is no ctest to drive. The binaries and a
# manifest naming them were installed by `spack install`; walk the manifest and
# launch each one at every required rank count through flux.
else
    # Locate the manifest CMake generated and the package installed. The spack
    # package prepends share/Beatnik/tests to PATH, so scan PATH for it.
    _manifest=""
    _saved_ifs="${IFS}"
    IFS=':'
    for _p in ${PATH}; do
        if [ -f "${_p}/beatnik_milestone_manifest.txt" ]; then
            _manifest="${_p}/beatnik_milestone_manifest.txt"
            break
        fi
    done
    IFS="${_saved_ifs}"

    if [ -z "${_manifest}" ]; then
        echo "[milestone] FAIL: beatnik_milestone_manifest.txt not found on PATH." >&2
        echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing" >&2
        echo "  (and Beatnik_INSTALL_TEST_EXECUTABLES=ON)?" >&2
        exit 1
    fi
    _manifest_dir="$(cd "$(dirname "${_manifest}")" && pwd)"
    echo "[milestone] manifest = ${_manifest}"

    # A manifest line is `<target> [args...]`. Field 1 carries the backend
    # suffix and is what selects; the remaining fields are the binary's
    # arguments and any path among them is MANIFEST-RELATIVE, so the whole
    # invocation runs from the manifest directory -- the same convention, and
    # the same reasoning, as the gate runner's.
    #
    # With BEATNIK_MILESTONE_MEMBERS set, field 1 must ALSO equal one of the
    # stems once its `_MPI_<BACKEND>` suffix is stripped -- equality, not a
    # prefix match, so Beatnik_Test_Milestone0Fmm never selects ...FmmL4.
    _ranks_run=0
    _matched_stems=" "
    for _backend in ${BEATNIK_MILESTONE_BACKENDS}; do
        _lines="$(grep -v '^[[:space:]]*\(#\|$\)' "${_manifest}" |
                  awk -v b="_${_backend}" -v m="${BEATNIK_MILESTONE_MEMBERS}" '
                      BEGIN { n = split(m, a, " ")
                              for (i = 1; i <= n; i++) want[a[i]] = 1 }
                      substr($1, length($1) - length(b) + 1) == b {
                          if (n == 0) { print; next }
                          sfx = "_MPI" b
                          if (substr($1, length($1) - length(sfx) + 1) != sfx) next
                          if (substr($1, 1, length($1) - length(sfx)) in want) print
                      }' || true)"
        if [ -z "${_lines}" ]; then
            echo "[milestone] no ${BEATNIK_MILESTONE_LABEL} binaries for ${_backend}"
            continue
        fi
        # FD 3, NOT STDIN: `flux run` below inherits and CONSUMES the loop's
        # stdin, so with the line list on stdin the first launched binary
        # swallows every remaining line and the tier silently runs only its
        # first member -- reporting PASS while covering less than it claims.
        # Copied in shape from the gate runner, which already paid for this.
        while IFS= read -r _line <&3; do
            [ -n "${_line}" ] || continue
            # shellcheck disable=SC2086
            set -- ${_line}
            _target="$1"
            shift
            _matched_stems="${_matched_stems}${_target%_MPI_${_backend}} "
            _exe="$(beatnik_exe "${_target}")" || { _milestone_rc=1; continue; }

            # One I/O directory PER TEST, on lustre, deleted and recreated
            # before the test runs so a stale checkpoint from an earlier run
            # cannot be read back as this run's output. Exported absolute: the
            # launch below runs from the manifest directory, which lives inside
            # a read-only spack install prefix.
            export BEATNIK_TEST_SCRATCH="${BEATNIK_MILESTONE_SCRATCH_ROOT}/${_target}"
            rm -rf "${BEATNIK_TEST_SCRATCH}"
            mkdir -p "${BEATNIK_TEST_SCRATCH}" || {
                echo "[milestone] FAIL: cannot create ${BEATNIK_TEST_SCRATCH}" >&2
                _milestone_rc=1
                continue
            }
            echo "[milestone] scratch = ${BEATNIK_TEST_SCRATCH}"

            for _np in ${BEATNIK_MILESTONE_RANKS}; do
                # Tuolumne packs 4 ranks per node; round the node count up.
                _nodes=$(( (_np + 3) / 4 ))
                echo "[milestone] === ${_target} at ${_np} ranks / ${_nodes} node(s) ==="
                _t0=$(date +%s)
                _launch_rc=0
                ( cd "${_manifest_dir}" && flux run \
                    --ntasks="${_np}" \
                    --nodes="${_nodes}" \
                    --exclusive \
                    --gpus-per-task=1 \
                    --cores-per-task=24 \
                    --setopt=mpibind=verbose:1 \
                    "${_exe}" "$@" ) || _launch_rc=1
                _ranks_run=$(( _ranks_run + 1 ))
                if [ "${_launch_rc}" -eq 0 ]; then
                    _launches_passed=$(( _launches_passed + 1 ))
                    echo "[milestone] PASS ${_target} np=${_np} in $(( $(date +%s) - _t0 ))s"
                else
                    _milestone_rc=1
                    echo "[milestone] FAIL ${_target} np=${_np} in $(( $(date +%s) - _t0 ))s"
                fi
            done
        done 3<<EOF
${_lines}
EOF
    done

    # A manifest that named nothing runnable is not a pass. The tier has had two
    # members since M0-T3, so hitting this guard now means something is wrong --
    # an install without +testing, or target names without the expected
    # _<BACKEND> suffix. An empty tier reporting PASS is the failure mode the
    # gate runner's identical guard exists to prevent, and it would be worse
    # here, where the tier's whole purpose is a comparison nobody else runs.
    if [ "${_ranks_run}" -eq 0 ]; then
        echo "[milestone] FAIL: the manifest named no runnable" \
             "${BEATNIK_MILESTONE_LABEL} tests for members" \
             "'${BEATNIK_MILESTONE_MEMBERS:-<all>}' and backends" \
             "'${BEATNIK_MILESTONE_BACKENDS}'." >&2
        echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing, do" >&2
        echo "  the manifest's target names carry the expected _<BACKEND> suffix," >&2
        echo "  and is every BEATNIK_MILESTONE_MEMBERS entry a registered stem?" >&2
        _milestone_rc=1
    fi
    # One unknown stem beside a known one runs the known one and must still not
    # pass: a typo would otherwise silently drop a member from the tier.
    for _m in ${BEATNIK_MILESTONE_MEMBERS}; do
        case "${_matched_stems}" in
            *" ${_m} "*) ;;
            *)
                echo "[milestone] FAIL: member '${_m}' matched no manifest line" \
                     "for backends '${BEATNIK_MILESTONE_BACKENDS}'." >&2
                _milestone_rc=1 ;;
        esac
    done
fi

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
# The milestone tier has FOUR members as of T6:
# Beatnik_Test_Milestone0Frozen (2000 steps of the frozen-mesh configuration at
# --icosphere-subdivisions 3 against the M0-G1 gold set, all 81 checkpointed
# steps at --rtol 1e-10 --atol 1e-12) and Beatnik_Test_Milestone0FrozenL4 (the
# same at subdivisions 4 against M0-G2), and T6's FMM pair
# Beatnik_Test_Milestone0Fmm / ...FmmL4 at the same two levels. Four members x
# {SERIAL, HIP} x ranks {1, 4} = SIXTEEN launches. The gate is unaffected and
# stays at five members and 60 launches.
#
# In tree mode ctest reports its own per-test results above, so the launch
# counts here are those of the installed path only.
if [ "${BEATNIK_BIN_MODE}" = "tree" ]; then
    _summary_counts="ctest; per-test results above"
else
    _summary_counts="${_launches_passed}/${_ranks_run} launches"
fi
if [ "${_milestone_rc}" -eq 0 ]; then
    echo "[milestone] PASS (label=${BEATNIK_MILESTONE_LABEL})"
    echo "[milestone] SUMMARY: PASS (${_summary_counts})"
else
    echo "[milestone] FAIL (label=${BEATNIK_MILESTONE_LABEL})" >&2
    echo "[milestone] SUMMARY: FAIL (${_summary_counts})"
fi
exit "${_milestone_rc}"
