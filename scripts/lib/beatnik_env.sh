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
# Shared build/run profile resolver. Every batch script and run wrapper under
# scripts/ must source THIS FILE FIRST, before doing anything else:
#
#     source "${BEATNIK_REPO}/scripts/lib/beatnik_env.sh"
#
# It answers three questions, in order, and nothing else:
#
#   1. Where is the repo?          -> BEATNIK_REPO
#   2. What system are we on?      -> BEATNIK_SYSTEM      (from `hostname`)
#   3. How is work being done?     -> BEATNIK_BUILD_MODE / BEATNIK_BIN_MODE
#                                     BEATNIK_SPACK_ENV / BEATNIK_BUILD_DIR
#
# Note the scopes of (2) and (3). The system is a fact about the *machine*; the
# build mode, the environment a checkout is developed into, and the build
# directory are facts about *this instance of the repository* — two clones on one
# machine can legitimately differ. Instance facts live in the gitignored
# scripts/<system>/profile.local.sh; the committed profile.defaults.sh holds only
# machine-wide fallbacks so a fresh clone runs with zero configuration.
#
# Precedence for every knob is: environment > profile.local.sh > profile.defaults.sh
# (implemented with ${VAR:=default}, so an already-exported value always wins).
# BEATNIK_PROFILE_SOURCE reports which of the three actually supplied the build
# mode, so callers can tell a recorded choice from an unconfirmed fallback.
#
# Knobs
# -----
#   BEATNIK_SYSTEM             Override hostname-based system detection.
#   BEATNIK_BUILD_MODE         "spack" | "manual".
#   BEATNIK_BIN_MODE           Derived from BUILD_MODE; "installed" | "tree".
#                              Set explicitly only to override the derivation.
#   BEATNIK_SPACK_ENV          Development spack environment path.
#   BEATNIK_SPACK_PROD_ENV     Production spack environment path.
#   BEATNIK_BUILD_DIR          Out-of-tree cmake build dir (manual mode).
#   BEATNIK_USE_PROD=1         Activate the production env instead of dev.
#   BEATNIK_NO_SPACK_ACTIVATE=1  Skip env activation AND runtime_env.sh. Use
#                              when the caller has already set up the shell.
#   BEATNIK_ENV_DRY_RUN=1      Resolve and print the profile, touch nothing.
#
# Provides
# --------
#   beatnik_exe <relpath|name>   Resolve a binary for the active bin mode.
#   beatnik_env_summary          Print the resolved profile.
#   BEATNIK_PROFILE_SOURCE       "environment" | "profile.local.sh" |
#                                "profile.defaults.sh" — where the build mode
#                                came from. A value of profile.defaults.sh means
#                                this checkout never recorded a choice.
#
############################################################################

##--------------------------------------------------------------------------##
## 0. Make the caller-supplied knobs safe to read
##--------------------------------------------------------------------------##
# Batch scripts run under `set -u`, and this file is sourced INTO them, so an
# unset knob would abort the job on first read rather than fall through to its
# default. Default every input knob to empty here so the `[ -z ... ]` /
# `[ "${...}" = 1 ]` tests below can be written plainly.
#
# Only INPUTS belong in this list. BEATNIK_PROFILE_SOURCE and
# BEATNIK_ACTIVE_SPACK_ENV are outputs and are assigned unconditionally below.
: "${BEATNIK_REPO:=}"
: "${BEATNIK_SYSTEM:=}"
: "${BEATNIK_BUILD_MODE:=}"
: "${BEATNIK_BIN_MODE:=}"
: "${BEATNIK_SPACK_ENV:=}"
: "${BEATNIK_SPACK_PROD_ENV:=}"
: "${BEATNIK_BUILD_DIR:=}"
: "${BEATNIK_USE_PROD:=}"
: "${BEATNIK_NO_SPACK_ACTIVATE:=}"
: "${BEATNIK_ENV_DRY_RUN:=}"

##--------------------------------------------------------------------------##
## 1. Repo root
##--------------------------------------------------------------------------##
# Honor a pinned BEATNIK_REPO (batch scripts pin it, because a scheduler may
# launch them from an arbitrary cwd with BASH_SOURCE pointing into a spool
# copy). Otherwise derive it from this file's location: scripts/lib/ -> ../..
if [ -z "${BEATNIK_REPO}" ]; then
    _beatnik_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    BEATNIK_REPO="$(cd "${_beatnik_lib_dir}/../.." && pwd)"
    unset _beatnik_lib_dir
fi
export BEATNIK_REPO

##--------------------------------------------------------------------------##
## 2. System detection
##--------------------------------------------------------------------------##
# This `case` MUST mirror the "System detection" hostname table in CLAUDE.md.
# Adding a system means updating both, plus scripts/<system>/.
if [ -z "${BEATNIK_SYSTEM}" ]; then
    case "$(hostname)" in
        tuolumne*) BEATNIK_SYSTEM=tuolumne ;;
        *)
            echo "beatnik_env.sh: unrecognized host '$(hostname)'." >&2
            echo "  Add a case here and a row to the CLAUDE.md hostname table," >&2
            echo "  or set BEATNIK_SYSTEM=<system> explicitly." >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
fi
export BEATNIK_SYSTEM

_beatnik_sysdir="${BEATNIK_REPO}/scripts/${BEATNIK_SYSTEM}"

##--------------------------------------------------------------------------##
## 3. Profile: committed defaults, then per-checkout overrides
##--------------------------------------------------------------------------##
# profile.defaults.sh is committed and gives a zero-config profile.
# profile.local.sh is gitignored and is where a checkout records its own choice.
#
# NOTE THE ORDER: local is sourced BEFORE defaults, which looks backwards but is
# what makes the precedence work. Both files assign with `${VAR:=...}`, which is
# a no-op once a variable has a value. So the FIRST assignment wins, and the
# desired precedence — environment > local > defaults — is produced by visiting
# the sources in exactly that order. Sourcing defaults first would silently
# reduce profile.local.sh to a no-op.
#
# Provenance is tracked in BEATNIK_PROFILE_SOURCE and reported by
# beatnik_env_summary, because "which build mode am I in, and who decided that"
# is exactly the question that goes wrong silently. `defaults` means this
# checkout has never recorded a choice and is running on the system fallback.
if [ -n "${BEATNIK_BUILD_MODE}" ]; then
    BEATNIK_PROFILE_SOURCE=environment
fi

if [ -f "${_beatnik_sysdir}/profile.local.sh" ]; then
    # shellcheck source=/dev/null
    source "${_beatnik_sysdir}/profile.local.sh"
    : "${BEATNIK_PROFILE_SOURCE:=profile.local.sh}"
fi

if [ -f "${_beatnik_sysdir}/profile.defaults.sh" ]; then
    # shellcheck source=/dev/null
    source "${_beatnik_sysdir}/profile.defaults.sh"
    : "${BEATNIK_PROFILE_SOURCE:=profile.defaults.sh}"
else
    echo "beatnik_env.sh: missing ${_beatnik_sysdir}/profile.defaults.sh" >&2
    return 1 2>/dev/null || exit 1
fi
export BEATNIK_PROFILE_SOURCE

##--------------------------------------------------------------------------##
## 4. Derive the bin mode from the build mode
##--------------------------------------------------------------------------##
#   manual -> binaries live in the out-of-tree cmake build dir ("tree"), and
#             the gate is ctest run in that dir.
#   spack  -> `spack install` put binaries on PATH ("installed"); there is no
#             build tree, so the gate runs installed binaries via the scheduler.
if [ -z "${BEATNIK_BIN_MODE}" ]; then
    case "${BEATNIK_BUILD_MODE}" in
        manual) BEATNIK_BIN_MODE=tree ;;
        spack)  BEATNIK_BIN_MODE=installed ;;
        *)
            echo "beatnik_env.sh: BEATNIK_BUILD_MODE='${BEATNIK_BUILD_MODE}'" \
                 "is not one of: manual, spack" >&2
            return 1 2>/dev/null || exit 1
            ;;
    esac
fi
export BEATNIK_BUILD_MODE BEATNIK_BIN_MODE

# Which env this session targets. Dev unless BEATNIK_USE_PROD=1.
if [ "${BEATNIK_USE_PROD}" = "1" ]; then
    BEATNIK_ACTIVE_SPACK_ENV="${BEATNIK_SPACK_PROD_ENV}"
else
    BEATNIK_ACTIVE_SPACK_ENV="${BEATNIK_SPACK_ENV}"
fi
export BEATNIK_ACTIVE_SPACK_ENV

##--------------------------------------------------------------------------##
## 5. Summary helper
##--------------------------------------------------------------------------##
beatnik_env_summary() {
    echo "[beatnik_env] repo       = ${BEATNIK_REPO}"
    echo "[beatnik_env] system     = ${BEATNIK_SYSTEM}"
    echo "[beatnik_env] build_mode = ${BEATNIK_BUILD_MODE}" \
         "(from ${BEATNIK_PROFILE_SOURCE})"
    echo "[beatnik_env] bin_mode   = ${BEATNIK_BIN_MODE}"
    echo "[beatnik_env] spack_env  = ${BEATNIK_ACTIVE_SPACK_ENV}" \
         "$([ "${BEATNIK_USE_PROD}" = "1" ] && echo '(PRODUCTION)' || echo '(dev)')"
    echo "[beatnik_env] build_dir  = ${BEATNIK_BUILD_DIR}"
}

##--------------------------------------------------------------------------##
## 6. Dry run stops here — resolve and report, mutate nothing
##--------------------------------------------------------------------------##
if [ "${BEATNIK_ENV_DRY_RUN}" = "1" ]; then
    beatnik_env_summary
    unset _beatnik_sysdir
    return 0 2>/dev/null || exit 0
fi

##--------------------------------------------------------------------------##
## 7. Activate the spack environment
##--------------------------------------------------------------------------##
if [ "${BEATNIK_NO_SPACK_ACTIVATE}" != "1" ]; then
    if [ -z "${BEATNIK_ACTIVE_SPACK_ENV}" ]; then
        echo "beatnik_env.sh: no spack env configured for ${BEATNIK_SYSTEM}" >&2
        return 1 2>/dev/null || exit 1
    fi
    # `spack` is a shell function supplied by setup-env.sh. Under a
    # non-interactive scheduler shell it may not be loaded yet.
    if ! type spack >/dev/null 2>&1; then
        if [ -n "${SPACK_ROOT}" ] && [ -f "${SPACK_ROOT}/share/spack/setup-env.sh" ]; then
            # shellcheck source=/dev/null
            source "${SPACK_ROOT}/share/spack/setup-env.sh"
        elif [ -f "${HOME}/spack/share/spack/setup-env.sh" ]; then
            # shellcheck source=/dev/null
            source "${HOME}/spack/share/spack/setup-env.sh"
        else
            echo "beatnik_env.sh: spack not on PATH and no setup-env.sh found." >&2
            echo "  Set SPACK_ROOT, or BEATNIK_NO_SPACK_ACTIVATE=1 if the" >&2
            echo "  environment is already active." >&2
            return 1 2>/dev/null || exit 1
        fi
    fi
    spack env activate "${BEATNIK_ACTIVE_SPACK_ENV}" || {
        echo "beatnik_env.sh: failed to activate ${BEATNIK_ACTIVE_SPACK_ENV}" >&2
        return 1 2>/dev/null || exit 1
    }
fi

##--------------------------------------------------------------------------##
## 8. Per-system launch-time runtime environment
##--------------------------------------------------------------------------##
# Single source of truth for env vars that must reach scheduler-launched tasks.
# Batch scripts must NOT re-export these inline — that is how the two copies
# drift apart. Skipped alongside env activation, on the assumption that a
# caller managing its own environment is managing all of it.
if [ "${BEATNIK_NO_SPACK_ACTIVATE}" != "1" ] &&
   [ -f "${_beatnik_sysdir}/runtime_env.sh" ]; then
    # shellcheck source=/dev/null
    source "${_beatnik_sysdir}/runtime_env.sh"
fi

unset _beatnik_sysdir

##--------------------------------------------------------------------------##
## 9. Binary resolution
##--------------------------------------------------------------------------##
# beatnik_exe <relpath-within-build-dir | bare-name>
#
#   tree mode      -> ${BEATNIK_BUILD_DIR}/<relpath>, falling back to a
#                     basename search under the build dir.
#   installed mode -> whatever `command -v <basename>` finds on the PATH the
#                     activated spack env set up.
#
# Prints the absolute path on stdout; returns non-zero and explains if not found.
beatnik_exe() {
    if [ $# -ne 1 ]; then
        echo "beatnik_exe: expected exactly one argument" >&2
        return 2
    fi
    local _want="$1"
    local _base="${_want##*/}"

    if [ "${BEATNIK_BIN_MODE}" = "installed" ]; then
        local _found
        _found="$(command -v "${_base}" 2>/dev/null)"
        if [ -n "${_found}" ]; then
            echo "${_found}"
            return 0
        fi
        echo "beatnik_exe: '${_base}' not on PATH." >&2
        echo "  bin_mode=installed; is ${BEATNIK_ACTIVE_SPACK_ENV} installed" \
             "(spack install) and does the package install this binary?" >&2
        return 1
    fi

    if [ -z "${BEATNIK_BUILD_DIR}" ]; then
        echo "beatnik_exe: BEATNIK_BIN_MODE=tree but BEATNIK_BUILD_DIR is unset." >&2
        return 1
    fi
    if [ -x "${BEATNIK_BUILD_DIR}/${_want}" ]; then
        echo "${BEATNIK_BUILD_DIR}/${_want}"
        return 0
    fi
    local _hit
    _hit="$(find "${BEATNIK_BUILD_DIR}" -type f -perm -u+x -name "${_base}" \
            -print -quit 2>/dev/null)"
    if [ -n "${_hit}" ]; then
        echo "${_hit}"
        return 0
    fi
    echo "beatnik_exe: '${_want}' not found under ${BEATNIK_BUILD_DIR}." >&2
    echo "  bin_mode=tree; has the build dir been configured and built?" >&2
    return 1
}
