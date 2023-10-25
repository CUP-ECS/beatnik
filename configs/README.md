# Build Configurations for different HPC systems

This directory contains build information and scripts for a variety of systems, mainly using the spack build system.  In general, these builds are designed to work either through a direct spack install/load or in a spack environment with appropriate external compilers. Each subdirectory has a run script for the relevant system, and the relevant spack installation information. In particular, on systems where spack is supported by system maintainers or works out of the box with minimal configuration, this file describes the spack install command line to use. For systems where spack cannot build beatnik out of the box, the directory also contains a `spack.yaml` environment specification that can fully configure and build beatnik on that system. 

## General build requirements
Beatnik depends on the following packages to build in all configurations:
  1. Cabana version 0.6.0 or newer
  1. A heffte version compatible with Cabana (2.3.0 is needed as of Cabana 0.6.0)
  1. A GPU-aware MPI implementation
  1. Kokkos 4.0 or newer
  1. LLNL Silo 4.11 or newer configured with MPI support

## Installing and Building Beatnik with Spack

The beatnik spack package should enforce its build requirements appropriately; we strongly suggest that you use the spack package for both installation (via `spack install` or `spack env create`) and development (via `spack develop` in a created environment). We strive to keep the spack package spec up to date to enable this. Note that to do so, you will often need your own spack installation, information about which can be found at [the spack documentation installation tutorial](https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html). On systems with a system spack install, we also advise upstreaming to the system spack installation. When appropriate, we note this in the list of systems below and provide a relevant `upstreams.yaml` file.

### Current list of tested systems targets and suggested installation method

We have tested beatnik installation on the following systems via either spack install with the provided spack install flags or a spack environment. An example run script is also provided for each of these systems in the appropriate subdirectory.
  * University of New Mexico - These systems simply use `spack install` as the UNM machines have a full global spack packages.yaml already set up. Because thse systems use spack for package maintenance, we recommend adding the provided `upstreams.yaml` file in the `unm` directory to the `.spack` directory in your home directory.
    * Hopper V100/A100 GPU cluster system - `spack install beatnik +cuda cuda_arch=80 %gcc ^cuda@11` (or `cuda_arch=70` for the V100 nodes); ^cuda@11 is needed to avoid using CUDA 12 which the UNM cuda drivers aren't updated to support as of 10/1/23. Only tested with gcc as the compiler.
    * General UNM (Wheeler/Hopper) CPU systems - `spack install beatnik` is sufficient.
    * If you prefer to use a spack environment to run beatnik instead of installing beatnik via spack, a `spack.yaml` file is provided in the `unm` directory.
  * Lawrence Livermore National Laboratory - These systems need a spack environment (provided) to set up compilers and external packages to use spack effectively on these systems. Use `spack env create` with the provided spack.yaml to build beatnik in an environment on these systems. Simple test run scripts are also provided.
    * Lassen V100 GPU system - Build using the environment specification in llnl/lassen/spack.yaml. Other compilers besides gcc untested. Note that cuda-aware MPI support is broken in spectrum MPI with cuda versions later than 11.2; the spack configuration avoids this, but if you're building beatnik by hand on Lassen or another IBM CORAL system, be aware of this limitation.
    * Tioga MX250X GPU system - Build using the environment specification in llnl/tioga/spack.yaml; gcc should also work. Other compilers besides cce 16.0.1 untested. Note that you must run with the environment variable to enable gpu-aware cray-mpich, i.e. `export MPICH_GPU_SUPPORT_ENABLED=1`. The provided flux script (beatnik.flux) does this.
    * Quartz CPU system - Build using the environment specification in llnl/quartz/spack.yaml. Othe rcompilers besides gcc@10.3.1 untested.
  * Los Alamos National Laboratory
    * Chicoma Cray A100 GPU system - not yet complete. An environment still needs to be developed that uses cray-mpich with the appropriate flags to properly compile beatnik and its dependenvies to use GPU-aware MPI.

If you use `spack install` to build beatnik (e.g., on the UNM systems), you'll then need to run `spack load beatnik` to get access to the test executables to run, for example the `rocketrig` benchmark. If you're running in an environment, the installed environment will include the benchmark executable.

### Developing Beatnik and its dependencies using a Spack package description

If you want to develop Beatnik, we recommend using an environment along with the `spack develop` command to setup the development environment. In addition to allowing you to use spack to install dependecies, this will also let you tweak the package specification to control the details of the build environment, directly modify packages that you're developing (e.g. beatnik //and// its dependencies if you want!) and still use spack to build it. 

For example, to work on the current development branch of beatnik on the LLNL lassen system, you might do the following:
```
# Create a local spack environment in which to develop
[bridges7@lassen708:~]$ mkdir devel-env
[bridges7@lassen708:~]$ cd devel-env
[bridges7@lassen708:devel-env]$ spack env create -d . beatnik/configs/llnl/lassen/spack.yaml
==> Created environment in /g/g16/bridges7/devel-env
==> You can activate this environment with:
==>   spack env activate /g/g16/bridges7/devel-env
[bridges7@lassen708:devel-env]$ spack env activate .
[bridges7@lassen708:devel-env]$ spack develop beatnik @develop +cuda cuda_arch=70 %gcc@8.3.1
==> Configuring spec beatnik@=develop%gcc@8.3.1+cuda cuda_arch=70 for development at path beatnik
[bridges7@lassen708:devel-env]$ spack concretize
==> Concretized beatnik@develop%gcc@8.3.1+cuda cuda_arch=70
...
# Now start developing in a git subbranch
[bridges7@lassen708:devel-env] cd beatnik
[bridges7@lassen708:beatnik]$ git checkout release-1.0-cleanup
Branch 'release-1.0-cleanup' set up to track remote branch 'release-1.0-cleanup' from 'origin'.
Switched to a new branch 'release-1.0-cleanup'
[bridges7@lassen708:beatnik]$ spack install
[+] /usr/tce/packages/cmake/cmake-3.23.1 (external cmake-3.23.1-gg5jrvmou665ssxt4wucmgij3b7nknou)
[+] /usr/tce/packages/cuda/cuda-11.7.0 (external cuda-11.7.0-u5mvzngv5t3r5s7em4bqmsupkkkrt7qu)
[+] /usr/tce/packages/spectrum-mpi/spectrum-mpi-rolling-release-gcc-8.3.1 (external spectrum-mpi-rolling-release-cms5gg7vjfr6ez4i6g6v2d72yxpd4apx)
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/pkgconf-1.9.5-2c2v5nm34o53bqmfesem4p7oyfqvisra
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/zlib-1.2.13-ibqzhjkqlgbzuer47g26qazsmqkete5y
[+] /usr (external ncurses-5.9-umdjlknnhlhps46qxuwkuvv2mqfotyhb)
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/gmake-4.4.1-bt4sn3lkoi26rkpzpnhk3suoeevhlgrf
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/blt-0.5.3-yd22z7pcyvdwzjd6de4fpgegj5gdtnck
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/kokkos-nvcc-wrapper-4.0.01-taoq622vaz2tem3e2afagwic5a7zhgfw
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/fftw-3.3.10-2rwicdyovmekxe4pwf4hgivtkttxozoh
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/readline-8.2-nlvfz33jvxohsdw5b3223xhwe44fjjot
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/hdf5-1.14.2-fjtpyalmdxaoyh4ea7yxy66ka2okvtoi
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/kokkos-4.1.00-kvn3hvnouc545lvxet73z3xhpdyhsxcn
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/heffte-2.1.0-rzqlejfwrnp34ihemu7tfzt5a2nkdqsh
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/silo-4.11-gaci6mdypwmuhepf6zsqqnupilug2jsl
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/cabana-0.5.0-fybm456ddrcnchkxcof53q7j427ef2to
==> Installing beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa [17/17]
==> No binary for beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa found: installing from source
==> No patches needed for beatnik
==> beatnik: Executing phase: 'cmake'
==> beatnik: Executing phase: 'build'
==> beatnik: Executing phase: 'install'
==> beatnik: Successfully installed beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa
  Stage: 0.00s.  Cmake: 11.20s.  Build: 1m 21.66s.  Install: 0.86s.  Post-install: 0.34s.  Total: 1m 35.20s
[+] /g/g16/bridges7/spack/opt/spack/linux-rhel7-power9le/gcc-8.3.1/beatnik-develop-vnxen3li2npx3muhwmxtabj2ycmxp2qa
```

Importantly, you can also develop beatnik's //dependencies// this way. For example, if you want to modify cabana to better support beatnik, you would also run `spack develop cabana && spack concretize -f` to add cabana to the list of packages to develop locally. You can find more information on the spack develop workflow at the [Spack documentation webpages](https://spack-tutorial.readthedocs.io/en/latest/tutorial_developer_workflows.html).
