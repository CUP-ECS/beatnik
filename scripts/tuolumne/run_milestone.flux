#!/bin/bash
# flux: --job-name=beatnik_milestone
# flux: --nodes=1
# flux: --exclusive
# flux: -t 60m
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
# Submit with:   flux batch scripts/tuolumne/run_milestone.flux
# Then read the beatnik_milestone.<jobid>.log it writes to cwd.
#
# This file is run_regression_minset.flux's structure with a different label,
# manifest name and rank list. It is a COPY on purpose: the gate script is
# single-sourced against CLAUDE.md's gate definition and must keep saying
# `regression` x ranks 1-6, so generalizing it in place would put this tier in
# a position to change what the gate means.
#
# --nodes=1 covers the 4-rank case at tuolumne's 4-ranks-per-node.
#
# THE WALLTIME IS MEASURED, NOT GUESSED (M0-T3, raised from 30m). M0-D1 step 6
# clocked the level-4 member's four tier launches at 22 + 45 + 382 + 1293 s =
# 1742 s (29.0 min) of launch wall, and the level-3 member adds 167 s of solve
# (8.2 + 29.3 + 43.2 + 86.2) plus its own startup and I/O -- about 32 minutes of
# measured work for the two members. On top of that each launch spawns 83
# compare_output.py invocations (81 compared steps plus the negative case),
# measured on-node at ~0.65 s each = ~7 minutes over the eight launches, which
# M0-A1's 40m estimate did not carry. 30m would have killed the second member
# partway and 40m left ~1% of margin -- and M0-R8 is exactly the failure mode
# where a truncated run reads as a shorter pass, so this is 60m -- the same cap
# M0-D1's own sweep ran under in pdebug. Raise it again,
# with the measurement, if a third member lands.
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

beatnik_env_summary

##--------------------------------------------------------------------------##
## Tier parameters
##--------------------------------------------------------------------------##
BEATNIK_MILESTONE_LABEL="milestone"
BEATNIK_MILESTONE_BACKENDS="${BEATNIK_MILESTONE_BACKENDS:-SERIAL HIP}"
BEATNIK_MILESTONE_RANKS="${BEATNIK_MILESTONE_RANKS:-1 4}"

# Parent of the per-test I/O directories. MUST be on a PARALLEL filesystem: the
# checkpoints go through MPI-IO and a node-local scratch fails every launch that
# spans more than one node (CLAUDE.md "Minimum test set").
BEATNIK_MILESTONE_SCRATCH_ROOT="${BEATNIK_MILESTONE_SCRATCH_ROOT:-/p/lustre5/stewartj/beatnik/milestone0}"

echo "[milestone] label=${BEATNIK_MILESTONE_LABEL}" \
     "backends='${BEATNIK_MILESTONE_BACKENDS}'" \
     "ranks='${BEATNIK_MILESTONE_RANKS}'"

_milestone_rc=0

##--------------------------------------------------------------------------##
## manual / tree mode: ctest inside the build directory
##--------------------------------------------------------------------------##
# The harness already registered one ctest case per (backend, rank), so
# `-L <label> -R <backend>` selects the whole rank sweep.
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
        echo "[milestone] ctest -L ${BEATNIK_MILESTONE_LABEL} -R ${_backend}"
        ( cd "${BEATNIK_BUILD_DIR}" &&
          ctest --output-on-failure --no-tests=ignore \
                -L "${BEATNIK_MILESTONE_LABEL}" -R "${_backend}" ) || _milestone_rc=1
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
    _ranks_run=0
    for _backend in ${BEATNIK_MILESTONE_BACKENDS}; do
        _lines="$(grep -v '^[[:space:]]*\(#\|$\)' "${_manifest}" |
                  awk -v b="_${_backend}" \
                      'substr($1, length($1) - length(b) + 1) == b' || true)"
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
                ( cd "${_manifest_dir}" && flux run \
                    --ntasks="${_np}" \
                    --nodes="${_nodes}" \
                    --exclusive \
                    --gpus-per-task=1 \
                    --cores-per-task=24 \
                    --setopt=mpibind=verbose:1 \
                    "${_exe}" "$@" ) || _milestone_rc=1
                _ranks_run=$(( _ranks_run + 1 ))
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
             "${BEATNIK_MILESTONE_LABEL} tests for backends" \
             "'${BEATNIK_MILESTONE_BACKENDS}'." >&2
        echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing, and do" >&2
        echo "  the manifest's target names carry the expected _<BACKEND> suffix?" >&2
        _milestone_rc=1
    fi
fi

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
# The milestone tier has TWO members as of M0-T3:
# Beatnik_Test_Milestone0Frozen (2000 steps of the frozen-mesh configuration at
# --icosphere-subdivisions 3 against the M0-G1 gold set, all 81 checkpointed
# steps at --rtol 1e-10 --atol 1e-12) and Beatnik_Test_Milestone0FrozenL4 (the
# same at subdivisions 4 against M0-G2). Two members x {SERIAL, HIP} x ranks
# {1, 4} = EIGHT launches. milestone1.md's M1-T1 adds the third member. The gate
# is unaffected and stays at five members and 60 launches.
if [ "${_milestone_rc}" -eq 0 ]; then
    echo "[milestone] PASS (label=${BEATNIK_MILESTONE_LABEL})"
else
    echo "[milestone] FAIL (label=${BEATNIK_MILESTONE_LABEL})" >&2
fi
exit "${_milestone_rc}"
