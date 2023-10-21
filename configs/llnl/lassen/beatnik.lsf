#!/bin/bash

### LSF Syntax
#BSUB -nnodes 2
#BSUB -W 120
#BSUB -e rocketrig.%J.err
#BSUB -o rocketrig.%J.out
#BSUB -J beatnik-rocketrig-8
#BSUB -q pdebug

# We create a scratch directory to run in in case I/O is requested. However,
# we default we run with -F0 here to disable I/O as beatnik is primarily as a 
# compute and communication benchmark.
SCRATCHDIR=/p/gpfs1/${USER}/beatnik-rocketrig
SPACKENVDIR=${HOME}/beatnik-env-lassen 

mkdir -p ${SCRATCHDIR}/data/raw

# Then we load spack, which we use to build or develop beatnik
SPACKDIR=${HOME}/spack
source ${SPACKDIR}/share/spack/setup-env.sh

# Since we're running from a beatnik built in a spack environment, activate it.
spack env activate ${SPACKENVDIR}

ROCKETRIG=`which rocketrig`
cd ${SCRATCHDIR}
lrun -M -gpu -n 8 -N 2 -T 4 -g 1 ${ROCKETRIG} -x cuda -n 4096 -w 8 -F 0
