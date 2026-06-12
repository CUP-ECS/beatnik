#!/bin/bash
# flux: --job-name=fmm_in
# flux: --nodes=1
# flux: --exclusive
# flux: --time=10
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Run rocketrig against the shipped fmm.in (free-boundary sech2
# rollup driven by Canopy's FMM BR solver) at 4 ranks, then re-run
# the FmmVsExact minimum test set so we know the banner-echo +
# fmm.in nudges did not regress the test surface.

set -euo pipefail

spack env activate ~/spack_envs/tuolumne_beatnik

export MPICH_GPU_SUPPORT_ENABLED=1
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096
export FI_CXI_ATS=0
export HSA_XNACK=1
export MPICH_SMP_SINGLE_COPY_MODE=NONE
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE

ROCKETRIG="$(command -v rocketrig)"
FMM_IN="$(spack location -i beatnik)/share/Beatnik/examples/01_rocketrig/fmm.in"

echo "::: rocketrig: ${ROCKETRIG}"
echo "::: fmm.in   : ${FMM_IN}"

echo "::: ---- rocketrig fmm.in at 4 ranks ----"
flux run --ntasks=4 --nodes=1 --exclusive --gpus-per-task=1 \
    --cores-per-task=24 --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${FMM_IN}"

echo "::: ---- FmmVsExact minimum test set ----"
DEVICES=(HIP OPENMP SERIAL)
run_test() {
    local device="$1" ntasks="$2"
    local nodes=$(( (ntasks + 3) / 4 ))
    local binary="Beatnik_Test_FmmVsExact_MPI_${device}"
    if ! command -v "${binary}" >/dev/null 2>&1; then
        echo "::: SKIP ${binary} (not on PATH)"; return
    fi
    echo "::: RUN  ${binary}  ntasks=${ntasks}"
    flux run --ntasks=${ntasks} --nodes=${nodes} --exclusive \
        --gpus-per-task=1 --cores-per-task=24 --setopt=mpibind=verbose:1 \
        "${binary}"
    echo "::: PASS ${binary}  ntasks=${ntasks}"
}
for d in "${DEVICES[@]}"; do
    run_test "${d}" 1
    run_test "${d}" 4
done

echo "::: All checks completed."
