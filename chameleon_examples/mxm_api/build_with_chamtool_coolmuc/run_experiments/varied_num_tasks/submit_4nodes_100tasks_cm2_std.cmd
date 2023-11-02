#!/bin/bash
#SBATCH -J chamtool_mxm_api
#SBATCH -o job_output/out_%j_mxm_100_tasks_chamtool_pred3_mig0.txt
#SBATCH -e job_output/err_%j_mxm_100_tasks_chamtool_pred3_mig0.txt
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2
#SBATCH --partition=cm2_std
#SBATCH --qos=cm2_std
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=13
#SBATCH --mem=MaxMemPerNode

source /etc/profile.d/modules.sh
module load slurm_setup

module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load libffi-3.3-oneapi-2021.1-3vomiwb
module load cereal-1.3.0-gcc-7.5.0-jwb3bux # built by spack
module load boost-1.76.0-gcc-7.5.0-fromrfo ## built with local-spack
module load hdf5-1.10.4         ## built with oneapi/2021
module load armadillo-10.4.0    ## built with oneapi/2021
module load ensmallen-2.16.2    ## built with oneapi/2021
module load mlpack-3.4.2        ## built with oneapi/2021
module load libtorch
module load chamtool_pred3_mig0

## do not have permission
## module use -a /lrz/sys/share/modules/extfiles
## module add likwid/modified

# export KMP_AFFINITY=verbose
unset KMP_AFFINITY
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export OMP_PROC_BIND=true
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto


ulimit -s unlimited

## for new mpi version which does not support a sufficient number of tags
export MPIR_CVAR_CH4_OFI_TAG_BITS=30
export MPIR_CVAR_CH4_OFI_RANK_BITS=8

## for chameleon configs
export OMP_NUM_THREADS=13
export MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=0.1
export MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE=1
export PERCENTAGE_DIFF_TASKS_TO_MIGRATE=0.3
export MXM_EXAMPLE=1
export CHAMELEON_TOOL=1
export CHAMELEON_TOOL_SUPPORT=1
export CHAMELEON_TOOL_LIBRARIES="/dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_tools/load_pred_tool/build/libtool.so"
export LOG_DIR="/dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_chamtool_coolmuc/run_experiments/varied_num_tasks/logs/100_tasks"

NUM_TASKS=100
MIN_SIZE=128
MAX_SIZE=1024

## for mxm, we config
export TASKS_PER_RANK=${NUM_TASKS}
export EST_NUM_ITERS=5
export TIME2TRAIN=3

echo "mpirun -n ${SLURM_NTASKS} /dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_chamtool_coolmuc/mxm_api random-size ${NUM_TASKS} ${MIN_SIZE} ${MAX_SIZE}"
mpirun -n ${SLURM_NTASKS} /dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_chamtool_coolmuc/mxm_api random-size ${NUM_TASKS} ${MIN_SIZE} ${MAX_SIZE}
