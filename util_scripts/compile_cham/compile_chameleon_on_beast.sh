#!/bin/bash
echo "!!!!!!! Compiling Chameleon on BEAST !!!!!!!"

# get the current date
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")"

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon}
echo "1. Chameleon source in: ${DIR_CH_SRC}"

# check the current directory
CUR_DIR=$(pwd)

# load given-dependencies on CoolMUC2, e.g., hwloc, cmake
echo "2. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/.module
module use ~/loc-libs/spack/share/spack/lmod/linux-sles15-zen2
module load loc-util-loggers
# module load hdf5-1.10.4-intel-2021.2.0-akmloob
module load hdf5-1.10.7-gcc-10.2.1-3c4s4ai
module load intel-oneapi-compilers-2021.2.0-gcc-10.2.1-rrubbme
module load intel-oneapi-mpi-2021.2.0-gcc-10.2.1-7hbelev
module load netcdf-c-4.6.1-gcc-10.2.1-s5bm5ra
module load cmake-3.20.3-gcc-10.2.1-crj7py7
module load hwloc-2.4.1-gcc-10.2.1-bu232a3
module load libffi-3.3-gcc-10.2.1-pp5dq3s

# setting compilers and flags
export C_COMPILER=mpiicc
export CXX_COMPILER=mpiicpc
export FORT_COMPILER=mpiifort
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KMP_AFFINITY=verbose
# export I_MPI_PIN=1
# export I_MPI_PIN_DOMAIN=auto
# export I_MPI_FABRICS="shm:tmi"
# export I_MPI_DEBUG=5


# move to Chameleon folder
cd ${DIR_CH_SRC}

# remove some old compiled files
rm -r cmake_install.cmake CMakeCache.txt Makefile CMakeFiles install_manifest.txt

# compile chameleon
if [ "${BUILD_CHAM}" = "1" ]
then

###################### WITH ITAC ############################


###################### WITHOUT ITAC #########################

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_commthread" \
        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_mig" \
        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=0 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_rep3_mig" \
        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
make install

make clean
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_rep3_nomig" \
        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
make install

fi

# move back to the current folder
cd ${CUR_DIR}