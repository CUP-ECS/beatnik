#!/bin/bash
# flux: --job-name=[TEST_NAME]
# flux: --nodes=1
# flux: --exclusive
# flux: --time=5
# flux: --output={{name}}.{{jobid}}.log
# flux: -q pdebug

# Activate the Beatnik spack environment.
spack env activate ~/spack_envs/beatnik-canopy

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

# NOTE: --ntasks must equal --nodes * 4. Update both together if you change
# --nodes in the flux header above.
flux run \
    --ntasks=4 \
    --nodes=1 \
    --exclusive \
    --gpus-per-task=1 \
    --cores-per-task=24 \
    --setopt=mpibind=verbose:1 \
    [EXECUTABLE_AND_ARGS]
