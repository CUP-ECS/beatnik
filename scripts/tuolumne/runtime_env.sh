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
# Launch-time environment for tuolumne (Cray MPICH + ROCm/HIP + Kokkos).
#
# THIS IS THE ONLY PLACE these variables are set. Batch scripts must source
# scripts/lib/beatnik_env.sh (which sources this file) rather than re-exporting
# them inline — inline copies drift, and a stale copy in one script is
# effectively an unreproducible run.
#
# Sourced automatically by the resolver, and skipped under
# BEATNIK_NO_SPACK_ACTIVATE=1 or BEATNIK_ENV_DRY_RUN=1.
############################################################################

# GPU-aware Cray MPICH. Without this, device pointers handed to MPI fail.
export MPICH_GPU_SUPPORT_ENABLED=1

# GTL eager/rendezvous cutoff for HSA (device) messages.
export GTL_HSA_VSMSG_CUTOFF_SIZE=4096

# Disable libfabric CXI address translation services.
export FI_CXI_ATS=0

# Allow page migration between host and MI300A device memory.
export HSA_XNACK=1

# Cray MPICH intra-node single-copy transport is disabled deliberately: it
# interacts badly with GPU-aware transfers on this platform.
export MPICH_SMP_SINGLE_COPY_MODE=NONE

# GTL registration-cache capacity. The default (10000 simultaneous regions) is
# exhausted by many-way device-buffer scatters at high rank counts, which
# manifests as `dreg_evict returned NO_SPACE` and a hung job. See README
# "Known Issues" — this is out-provisioning, not a fix.
export GTL_DREG_CACHE_SIZE=262144

# OpenMP: 24 cores per task, 4 tasks per node on tuolumne.
export OMP_NUM_THREADS=24
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_WAIT_POLICY=PASSIVE
