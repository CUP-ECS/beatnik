# UNM Beatnik Configurations
This directory contains spack configurations and test scripts for the UNM Wheeler, Xnea, and Hopper systems. UNM 
configurations generally rely on the system Spack install, though in many cases it helps if the user creates their
own spack installation that upstreams to the UNM system spack installation.

## Setting up your own spack setup on UNM systems
  1. Download spack and set it up, e.g. `git clone https://github.com/spack/spack ~/spack && git checkout develop && source spack/share/spack/setup-env.sh`
  1. Configure your spack installation to upstream to the system spack configuration: `cp upstreams.yaml ~/.spack/upstreams.yaml`
  1. Find the compilers you want to use with spack: `spack compiler find`

Once this done, you can use your own spack installation to install the packages you need on different system, while still leveraging system-installed spack packages.

## Available spack configurations and tests

  1. Hopper GPU systems - `spack-hopper.yaml` and `test-hopper.slurm`
  2. Xena GPU systems - `spack-xena.yaml` and `test-xena.slurm`
  3. Wheeler CPU systems - `spack-wheeler.yaml` and `test-wheeler.slurm`

Please read the comments at the top of each file to understand their requirements and how to use the files on these systems (e.g. which system modules to load along with the spack to make them work).
