#!/bin/bash
# flux: --job-name=beatnik_unit_tests
# flux: --nodes=1
# flux: --exclusive
# flux: -t 20m
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
# THE WHOLE `unit` TIER on tuolumne.
#
# The unit tier is DIAGNOSTIC, not the ship gate -- it tells you *where* a fault
# is and does not gate a change (docs/testing.md "The tiers"). The gate is
# scripts/tuolumne/run_regression_minset.flux and is a different script on
# purpose.
#
# This script DISCOVERS its tests rather than naming them, so it grows as tasks
# land (T2b and T2c both add unit tests). Two paths, chosen by the checkout's
# build mode:
#
#   tree mode      `ctest -L unit` in ${BEATNIK_BUILD_DIR}. The labels are
#                  already registered, so this needs no list.
#   installed mode no build tree and therefore no ctest. Walk
#                  beatnik_unit_manifest.txt, which tests/CMakeLists.txt
#                  generated from the same registrations that applied the
#                  labels. Found by scanning PATH, since the spack package
#                  prepends share/Beatnik/tests when +testing.
#
# Manifest line kinds (see tests/CMakeLists.txt for the generator):
#
#   exe     <target>                    a binary; run it, expect exit 0
#   py-pass <name> <script> <args...>   python; expect exit 0
#   py-fail <name> <script> <args...>   python; expect a NON-ZERO exit. This is
#                                       where ctest's WILL_FAIL property has to
#                                       live when there is no ctest -- the
#                                       comparator's negative case is CORRECT
#                                       when it fails, and a runner that missed
#                                       that would report the tier red exactly
#                                       when the comparator works.
#
# Every test decides its own verdict and returns non-zero on failure (see
# tests/unit_tests/Beatnik_TestAssert.hpp), so this script only has to aggregate
# exit codes -- and it does, so THE JOB'S OWN STATUS IS MEANINGFUL.
#
# Submit with:   flux batch scripts/tuolumne/unit_tests.flux
# Then read the beatnik_unit_tests.<jobid>.log it writes to cwd.
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
    echo "[unit] FAIL: cannot locate the Beatnik checkout." >&2
    echo "  Submit from inside it, or export BEATNIK_REPO before flux batch." >&2
    exit 1
fi

# shellcheck source=../lib/beatnik_env.sh
source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh" || exit 1

beatnik_env_summary

_rc=0
_pass=0
_fail=0
_names_failed=""

# The unit tier runs at ONE rank; the rank sweep belongs to the gate.
BEATNIK_UNIT_RANKS="${BEATNIK_UNIT_RANKS:-1}"
_nodes=$(( (BEATNIK_UNIT_RANKS + 3) / 4 ))
if [ "${_nodes}" -lt 1 ]; then _nodes=1; fi

_record() {
    # $1 = name, $2 = observed rc, $3 = 0 for "expect success", 1 for
    # "expect failure" (the WILL_FAIL cases).
    local _name="$1" _obs="$2" _expect_fail="$3"
    if [ "${_expect_fail}" -eq 1 ]; then
        if [ "${_obs}" -ne 0 ]; then
            echo "[unit] PASS (expected failure, rc=${_obs}) ${_name}"
            _pass=$(( _pass + 1 ))
            return
        fi
        echo "[unit] FAIL ${_name}: exited 0 but is expected to FAIL" >&2
    else
        if [ "${_obs}" -eq 0 ]; then
            echo "[unit] PASS ${_name}"
            _pass=$(( _pass + 1 ))
            return
        fi
        echo "[unit] FAIL ${_name}: rc=${_obs}" >&2
    fi
    _fail=$(( _fail + 1 ))
    _names_failed="${_names_failed} ${_name}"
    _rc=1
}

##--------------------------------------------------------------------------##
## manual / tree mode: ctest inside the build directory
##--------------------------------------------------------------------------##
if [ "${BEATNIK_BIN_MODE}" = "tree" ]; then
    if [ ! -d "${BEATNIK_BUILD_DIR}" ]; then
        echo "[unit] FAIL: build dir ${BEATNIK_BUILD_DIR} does not exist." >&2
        exit 1
    fi
    echo "[unit] ctest -L unit in ${BEATNIK_BUILD_DIR}"
    ( cd "${BEATNIK_BUILD_DIR}" &&
      ctest --output-on-failure --no-tests=error -L unit ) || _rc=1
    if [ "${_rc}" -eq 0 ]; then
        echo "[unit] PASS (ctest -L unit)"
    else
        echo "[unit] FAIL (ctest -L unit)" >&2
    fi
    exit "${_rc}"
fi

##--------------------------------------------------------------------------##
## spack / installed mode: walk the manifest
##--------------------------------------------------------------------------##
_manifest=""
_saved_ifs="${IFS}"
IFS=':'
for _p in ${PATH}; do
    if [ -f "${_p}/beatnik_unit_manifest.txt" ]; then
        _manifest="${_p}/beatnik_unit_manifest.txt"
        break
    fi
done
IFS="${_saved_ifs}"

if [ -z "${_manifest}" ]; then
    echo "[unit] FAIL: beatnik_unit_manifest.txt not found on PATH." >&2
    echo "  Is ${BEATNIK_ACTIVE_SPACK_ENV} installed with +testing" >&2
    echo "  (and Beatnik_INSTALL_TEST_EXECUTABLES=ON)?" >&2
    exit 1
fi
_manifest_dir="$(cd "$(dirname "${_manifest}")" && pwd)"
echo "[unit] manifest = ${_manifest}"

_python="${BEATNIK_PYTHON:-python3}"

while IFS= read -r _line || [ -n "${_line}" ]; do
    case "${_line}" in
        ''|'#'*) continue ;;
    esac
    # shellcheck disable=SC2086
    set -- ${_line}
    _kind="$1"
    shift
    case "${_kind}" in
    exe)
        _target="$1"
        _exe="$(beatnik_exe "${_target}")" || { _record "${_target}" 127 0; continue; }
        echo "[unit] === ${_target} (${BEATNIK_UNIT_RANKS} rank(s)) ==="
        flux run \
            --ntasks="${BEATNIK_UNIT_RANKS}" \
            --nodes="${_nodes}" \
            --exclusive \
            --gpus-per-task=1 \
            --cores-per-task=24 \
            --setopt=mpibind=verbose:1 \
            "${_exe}"
        _record "${_target}" "$?" 0
        ;;
    py-pass|py-fail)
        _name="$1"
        shift
        echo "[unit] === ${_name} ==="
        # Run from the manifest's directory, NOT from the submitting cwd. EVERY
        # path on the line is manifest-relative -- the script and the data files
        # it is handed -- so prefixing only the script is not enough. Getting
        # this wrong is worse than a plain failure: a missing fixture makes the
        # positive case fail loudly but makes the py-fail case pass for the
        # wrong reason, so the tally reads 2/3 instead of 1/3. Found by running
        # it; recorded in tasks/framework.md.
        #
        # No `flux run`: these are pure Python over committed fixtures, with no
        # MPI and no GPU, so they run directly on the batch node.
        ( cd "${_manifest_dir}" && "${_python}" "$@" )
        _obs=$?
        if [ "${_kind}" = "py-fail" ]; then
            _record "${_name}" "${_obs}" 1
        else
            _record "${_name}" "${_obs}" 0
        fi
        ;;
    *)
        echo "[unit] FAIL: unrecognized manifest line kind '${_kind}'" >&2
        echo "  line: ${_line}" >&2
        _rc=1
        ;;
    esac
done < "${_manifest}"

##--------------------------------------------------------------------------##
## Report
##--------------------------------------------------------------------------##
_total=$(( _pass + _fail ))
if [ "${_total}" -eq 0 ]; then
    echo "[unit] FAIL: the manifest named no tests at all, which is not a pass." >&2
    exit 1
fi
if [ "${_rc}" -eq 0 ]; then
    echo "[unit] SUMMARY: PASS (${_pass}/${_total} tests)"
else
    echo "[unit] SUMMARY: FAIL (${_pass}/${_total} tests);" \
         "failed:${_names_failed}" >&2
fi
exit "${_rc}"
