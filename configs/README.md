# Build Configuraitons for diffrent HPC systems

This directory contains build information and scripts for a variety of systems, mainly using the spack build system.  In general, these builds are designed to work in a self-contained spack environment (spack documentation goes here XXX) with appropriate external compilers. Each directory describes the compiler used on each system and how to access/configure it, along with the spack environment to create in which to build beatnik.  For systems where spack is not pre-configured by the system maintainers, the build directory also contains a link to a repository with a relevant spack packages.yaml file, as well as a spack environment specification the can fully configure and build on that system. 

## Current list of tested build targets

  * University of New Mexico
    * Hopper V100/A100 GPU cluster system
    * Xena K40m GPU cluster system
    * General UNM (Wheeler/Hopper) CPU systems
  * Lawrence Livermore National Laboratory
    * Lassen V100 GPU system
    * Quartz CPU system
    * Tioga MX250X GPU system - WORK IN PROGRESS
  * Los ALamos National Laboratoru
    * Chicoma Cray A100 GPU system
    * Darwin GPU system

## Known build problems

  * The current HeFFTe spack specification tries to install testing code and does not create the directory needed for the install, resulting in a failed HeFFTe spack build.
