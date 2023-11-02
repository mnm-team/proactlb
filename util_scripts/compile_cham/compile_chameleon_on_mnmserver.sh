#!/bin/bash

# get the current date
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")" 
# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon/}
# check the current directory
CUR_DIR=$(pwd)

# Load itac
source /opt/intel/itac/2019.5.041/bin/itacvars.sh intel64

# Load intel-compiler
source /opt/intel/compilers_and_libraries/linux/bin/compilervars.sh intel64

# SET PKG CONFIG with FFI
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/home/chungmi/local-libs/libffi-3.3/build/lib/pkgconfig
export LD_LIBRARY_PATH=/home/chungmi/local-libs/libffi-3.3/build/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/home/chungmi/local-libs/libffi-3.3/build/lib64:$LIBRARY_PATH
export INCLUDE=/home/chungmi/local-libs/libffi-3.3/build/include:$INCLUDE
export CPATH=/home/chungmi/local-libs/libffi-3.3/build/include:$CPATH
# export FFI_INCLUDE_DIR=/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/include
# export FFI_LIBRARY_DIR=/dss/dsshome1/0A/di49mew/loc-libs/libffi-3.3/build/lib64

export C_COMPILER=/opt/intel/compilers_and_libraries_2019.5.281/linux/bin/intel64/icc
export CXX_COMPILER=/opt/intel/compilers_and_libraries_2019.5.281/linux/bin/intel64/icpc
export Fortran_COMPILER=/opt/intel/compilers_and_libraries_2019.5.281/linux/bin/intel64/ifort

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

# ##################### WITH ITAC #########################
# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_no_commthread_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_COMM_THREAD=0 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_mig_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

make clean 
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_rep_3_nomig_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${Fortran_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_rep_3_mig_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

# ##################### WITHOUT ITAC #########################
# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_no_commthread" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=0 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_mig" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_rep_3_nomig" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon/install/intel_rep_3_mig" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

fi

# move back to the current folder
cd ${CUR_DIR}