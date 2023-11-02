# remove old compiled fiels
echo "Removing old-compiled files..."
rm -r CMakeCache.txt CMakeFiles cmake_install.cmake Makefile

# load intel_compiler & itac, ...
echo "Loading intel-compiler & mlpack, chameleon-lib, ..."
module use ~/local-libs/spack/share/spack/modules/linux-centos7-ivybridge
module use ~/.modules
module load cmake-3.20.2-intel-2021.2.0-cpqchac
module load hwloc-2.4.1-intel-2021.2.0-ct77noz
module load libffi-3.3-intel-2021.2.0-wrygxlw
module load chamtool_pred_mig
module load boost-1.69.0-intel-2021.2.0-twv4dyn
module load cereal-1.3.0-intel-2021.2.0-o5uqirz
module load armadillo-10.4.1
module load ensmallen-2.16.2
module load hdf5-1.10.4
module load load-intelmkl-2019
module load mlpack-3.4.2

# export itac if we need
# but, maybe it's already loaded by intel oneapi/2021

# run cmake
cmake -DCMAKE_C_COMPILER=mpiicc -DCMAKE_CXX_COMPILER=mpiicpc \
        ..
