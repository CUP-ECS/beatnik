#!/bin/bash
# flux: --job-name=rocketrig_1024
# flux: --nodes=4
# flux: --exclusive
# flux: --time=60
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug
#
# Physics-verification single-mode rocketrig run (1024x1024 mesh, B=16)
# driven by Canopy's FMM BR solver. Uses
# examples/01_rocketrig/single_mode_1024.in, which is sized to reach a
# visible rollup (1500 steps at delta_t=0.0006) so we can confirm the IC
# + dynamics scale correctly before committing to the production meshes.
#
# This run uses the *development* spack environment
# (~/spack_envs/tuolumne_beatnik) -- it is an iterative verification run,
# not a long-queued production job. (Production-scale runs use
# ~/spack_envs/tuolumne_beatnik_production; see rocketrig_large.flux.)

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

NUM_PROCS=16
MESH_SIZE=1024
RUN_DIR="/p/lustre5/stewartj/beatnik/fmm/${NUM_PROCS}p_${MESH_SIZE}mesh"

ROCKETRIG="$(command -v rocketrig)"
INPUT_SRC="~/spack_envs/tuolumne_beatnik/beatnik/examples/01_rocketrig/single_mode_1024.in"

mkdir -p "${RUN_DIR}"
cp "${INPUT_SRC}" "${RUN_DIR}/single_mode_1024.in"
cd "${RUN_DIR}"

INPUT="${RUN_DIR}/single_mode_1024.in"

echo ":::"
echo "::: rocketrig binary: ${ROCKETRIG}"
echo "::: rocketrig input : ${INPUT} (copied from ${INPUT_SRC})"
echo "::: run directory   : ${RUN_DIR}"
echo ":::"

flux run \
    --ntasks=${NUM_PROCS} \
    --nodes=4 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    "${ROCKETRIG}" "${INPUT}"
