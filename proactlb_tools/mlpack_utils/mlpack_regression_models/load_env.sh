echo "Loading intel-compiler & mlpack, chameleon-lib, ..."
source /opt/intel/oneapi/setvars.sh
module use ~/projects/loc-libs/local-modulefiles
module use ~/projects/loc-libs/spack/share/spack/modules/linux-ubuntu20.04-skylake
module load hwloc-2.3.0
module load mlpack-3.4.2
# module load mlpack-4.0.0
module load cereal-1.3.0-gcc-9.3.0-ebju5lb
