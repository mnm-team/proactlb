#!/bin/bash
echo "!!!!!!! Compiling Chameleon-with-Tool on CoolMUC2 !!!!!!!"

# get the current date
export CUR_DATE_STR="$(date +"%Y%m%d_%H%M%S")"

# path to chameleon_folder
DIR_CH_SRC=${DIR_CH_SRC:-~/chameleon_tool_dev/}
echo "1. Chameleon source in: ${DIR_CH_SRC}"

# copy chameleon_patched files to merge with the tool
echo "2. Copy/Replace chameleon_patched files to integrate with the tool..."
cp ~/chameleon-scripts/cham_tools/load_pred_tool/chameleon_patch/* ~/chameleon_tool_dev/src/

# turn on CHAMELEON_TOOL flag and declare where is the tool
echo "3. Turn on CHAMELEON_TOOL flag, and declare the tool path..."
export CHAMELEON_TOOL=1
export CHAMELEON_TOOL_LIBRARIES=/home/chungmi/chameleon-scripts/cham_tools/load_pred_tool/build/libtool.so

# check the current directory
CUR_DIR=$(pwd)

# load given-dependencies on Server Projekt03, e.g., hwloc, cmake
echo "4. Load dependencies: hwloc, itac, cmake, libffi-3.3(local) ..."
module use ~/local-libs/spack/share/spack/modules/linux-centos7-ivybridge
module use ~/.modules
module load cmake-3.20.2-intel-2021.2.0-cpqchac
module load hwloc-2.4.1-intel-2021.2.0-ct77noz
module load libffi-3.3-intel-2021.2.0-wrygxlw

# option: load itac packages on CoolMUC2
# module load itac
# source /lrz/sys/intel/studio2019_u5/itac/2019.5.041/bin/itacvars.sh intel64

# setting compilers and flags
# note: intel oneapi/2021 is already loaded on Projekt03
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

# for Sam(oa)2 settings about the estimated max tasks per rank
#       + on CoolMUC2, 13 physical-threads/rank, with num_sections=16
#       + then, total num tasks/rank = 16 * 13 = 208 tasks
export MAX_EST_NUM_ITERS=100
export MAX_TASKS_PER_RANK=208

# move to Chameleon folder
cd ${DIR_CH_SRC}

# remove some old compiled files
rm -r cmake_install.cmake CMakeCache.txt Makefile CMakeFiles install_manifest.txt

# compile chameleon
if [ "${BUILD_CHAM}" = "1" ]
then

# ##################### WITH ITAC #########################
# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_no_commthread_itac" \
#        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
#        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_COMM_THREAD=0 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_mig_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_rep_3_nomig_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${Fortran_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_rep_3_mig_itac" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DTRACE -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

# ##################### WITHOUT ITAC #########################
# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_no_commthread" \
#        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
#        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=0 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

make clean 
cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_pred_mig" \
        -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_Fortran_COMPILER=${FORT_COMPILER} \
        -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=0 -DCHAMELEON_TOOL_SUPPORT -DCHAM_PROACT_MIGRATION=1 -DCHAM_PREDICTION_MODE=1 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_mig" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=0" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/chamtool_rep_3_nomig" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=0 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

# make clean 
# cmake -DCMAKE_INSTALL_PREFIX="~/chameleon_tool_dev/install/intel_rep_3_mig" -DCMAKE_C_COMPILER=${C_COMPILER} -DCMAKE_CXX_COMPILER=${CXX_COMPILER} -DCMAKE_CXX_FLAGS="-DCHAM_STATS_RECORD -DCHAM_STATS_PRINT -DENABLE_COMM_THREAD=1 -DENABLE_TASK_MIGRATION=1 -DCHAM_REPLICATION_MODE=3 -DCHAM_STATS_PER_SYNC_INTERVAL" cmake .
# make install

fi

# move back to the current folder
cd ${CUR_DIR}