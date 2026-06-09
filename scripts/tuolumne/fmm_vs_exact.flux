#!/bin/bash
# flux: --job-name=fmm_vs_exact
# flux: --nodes=1
# flux: --exclusive
# flux: --time=10
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Run Beatnik_Test_FmmVsExact at 1 rank and 4 ranks for each Kokkos
# backend installed by `beatnik +testing +canopy`. Submit with:
#
#     flux batch scripts/tuolumne/fmm_vs_exact.flux
#
# After the job finishes, read `fmm_vs_exact.<jobid>.log` for the
# pass/fail of each (DEVICE, np) combination.

set -euo pipefail

# Activate the Beatnik spack environment so test binaries are on PATH
# (setup_run_environment in the COMPASS beatnik package.py prepends
# share/Beatnik/tests/ when +testing).
spack env activate ~/spack_envs/tuolumne_beatnik

# Runtime environment for MPICH + Kokkos on tuolumne.
export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

# Which backends to exercise. HIP is the production target on tuolumne;
# OPENMP and SERIAL are kept here to surface backend-specific regressions
# (e.g. team-policy bugs that only show up on host backends). Trim this
# list if a backend isn't installed.
DEVICES=(HIP OPENMP SERIAL)

run_test() {
    local device="$1"
    local ntasks="$2"
    local nodes=$(( (ntasks + 3) / 4 ))
    local binary="Beatnik_Test_FmmVsExact_MPI_${device}"

    if ! command -v "${binary}" >/dev/null 2>&1; then
        echo "::: SKIP ${binary} (not on PATH)"
        return
    fi

    echo "::: RUN  ${binary}  ntasks=${ntasks}  nodes=${nodes}"
    flux run \
        --ntasks=${ntasks} \
        --nodes=${nodes} \
        --exclusive \
        --gpus-per-task=1 \
        --cores-per-task=24 \
        --setopt=mpibind=verbose:1 \
        "${binary}"
    echo "::: PASS ${binary}  ntasks=${ntasks}"
}

# 1-rank fits one node; 4-rank fills one node (4 ranks / node on tuolumne).
for device in "${DEVICES[@]}"; do
    run_test "${device}" 1
    run_test "${device}" 4
done

echo "::: All FmmVsExact runs completed."
