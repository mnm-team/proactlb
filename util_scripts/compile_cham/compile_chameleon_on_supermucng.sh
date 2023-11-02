#!/bin/bash
echo "!!!!!!! Compiling Chameleon on SuperMUC-NG !!!!!!!"

# get the current date
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")" 

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon/}
echo "1. Chameleon source: ${DIR_CH_SRC}"

# check the current directory
CUR_DIR=$(pwd)

# load hwloc/2.0.1, itac, cmake
echo "2. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/.modules
module load cmake/3.16.5
module load hwloc/2.2.0-gcc8
module load libffi-3.3-lrz-spack

# SET PKG CONFIG with FFI
echo "3. Exporting some env-variables: C, CXX & Settings ... compilers"
export C_COMPILER=/dss/dsshome1/lrz/sys/intel/oneapi/2021.2/mpi/2021.2.0/bin/mpiicc
export CXX_COMPILER=/dss/dsshome1/lrz/sys/intel/oneapi/2021.2/mpi/2021.2.0/bin/mpiicpc
export FORT_COMPILER=/dss/dsshome1/lrz/sys/intel/oneapi/2021.2/mpi/2021.2.0/bin/mpiifort

export OMP_PLACES=cores
export OMP_PROC_BIND=close
export KMP_AFFINITY=verbose
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto
# export I_MPI_FABRICS="shm:tmi"
# export I_MPI_DEBUG=5

# move to Chameleon folder
echo "4. Moving to Chameleon-src folder..."
cd ${DIR_CH_SRC}

# remove some old compiled files
echo "5. Removing old-compiled stuffs..."
rm -r cmake_install.cmake CMakeCache.txt Makefile CMakeFiles install_manifest.txt

# compile chameleon
if [ "${BUILD_CHAM}" = "1" ]
then

# ##################### WITH ITAC #########################


# ##################### WITHOUT ITAC ######################
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

# ##################### WITH DEBUG ########################



else

echo "Turn BUILD_CHAM to 1 for compiling the src (e.g., export BUILD_CHAM=1)..."

fi

# move back to the current folder
echo "Moving back to the curent folder..."
echo "!!!!!!! END COMPILING !!!!!!!"
cd ${CUR_DIR}