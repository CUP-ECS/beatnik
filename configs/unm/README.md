# UNM Beatnik Configurations
This directory contains run scripts for the UNM hopper cluster (using the CUP-ECS partition with A100 GPU nodes) and the UNM wheeler cluster. In general, beatnik should be installed on these systems using spack as described in the documentation in the parent directory of this directory.

  1. Hopper GPU systems - `hopper.slurm`
  2. Wheeler CPU systems - `wheeler.slurm`

Please read the comments at the top of each file to understand their requirements and how to use the files on these systems (e.g. which system modules to load along with the spack to make them work).
