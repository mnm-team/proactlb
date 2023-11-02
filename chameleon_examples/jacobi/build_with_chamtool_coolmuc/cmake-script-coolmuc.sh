# remove old compiled fiels
echo "Removing old-compiled files..."
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

# load intel_compiler & itac, ...
echo "Loading intel-compiler & mlpack, chameleon-lib, ..."
module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load boost-1.76.0-gcc-7.5.0-fromrfo # built with local-spack
module load chamtool_pred_mig
module load hwloc/2.0
module load set-hwloc-inc
module load libffi-3.3
# below is something needed for cham-tool
module load hdf5-1.10.4         # built with oneapi/2021
module load armadillo-10.4.0    # built with oneapi/2021
module load ensmallen-2.16.2    # built with oneapi/2021
module load cereal
module load mlpack-3.4.2        # built with oneapi/2021

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# run cmake
cmake -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        ..
