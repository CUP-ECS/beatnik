#!/bin/bash
# flux: --job-name=rocketrig_smoke
# flux: --nodes=1
# flux: --exclusive
# flux: --time=10
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Smoke-run the input-file-driven rocketrig at 4 ranks against the
# shipped rocketrig.in. Also run the FmmVsExact minimum test set at
# 1 and 4 ranks for all installed backends so we know the Canopy
# interface refresh + parser change leaves the test surface clean.

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
INPUT="$(spack location -i beatnik)/share/Beatnik/examples/01_rocketrig/rocketrig.in"

echo ":::"
echo "::: rocketrig binary: ${ROCKETRIG}"
echo "::: rocketrig input : ${INPUT}"
echo ":::"

echo "::: ---- 1. rocketrig --help (rank 0) ----"
flux run --ntasks=1 --nodes=1 --exclusive --gpus-per-task=1 \
    --cores-per-task=24 --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" --help

echo "::: ---- 2. rocketrig rocketrig.in at 4 ranks ----"
flux run --ntasks=4 --nodes=1 --exclusive --gpus-per-task=1 \
    --cores-per-task=24 --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${INPUT}"

echo "::: ---- 3. negative tests (parser errors) ----"
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "${TMPDIR_LOCAL}"' EXIT

cat > "${TMPDIR_LOCAL}/typo.in" <<'EOF'
nods = 64
EOF
cat > "${TMPDIR_LOCAL}/no_eq.in" <<'EOF'
nodes 64
EOF
cat > "${TMPDIR_LOCAL}/bad_enum.in" <<'EOF'
br_solver = magic
EOF

for f in typo no_eq bad_enum; do
    echo "::: negative: ${f}.in"
    if flux run --ntasks=1 --nodes=1 --exclusive --gpus-per-task=1 \
        --cores-per-task=24 --setopt=mpibind=verbose:1 \
        "${ROCKETRIG}" "${TMPDIR_LOCAL}/${f}.in"; then
        echo "::: FAIL — expected non-zero exit from ${f}.in"
        exit 1
    else
        echo "::: ok — ${f}.in errored as expected"
    fi
done

echo "::: ---- 4. minimum test set: FmmVsExact 1+4 ranks ----"
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
