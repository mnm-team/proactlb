#!/bin/bash
#SBATCH -J test_bw
#SBATCH -o job_output/out_%j_varied_tsizes_512.txt
#SBATCH -e job_output/err_%j_varied_tsizes_512.txt
#SBATCH -D ./
#SBATCH --get-user-env
#SBATCH --clusters=cm2_tiny
#SBATCH --partition=cm2_tiny
#SBATCH --qos=cm2_tiny
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=MaxMemPerNode

source /etc/profile.d/modules.sh
module load slurm_setup

module use ~/.modules
module use ~/local_libs/spack/share/spack/modules/linux-sles15-haswell
module load hwloc-2.4.1-oneapi-2021.1-rrkvihp
module load libffi-3.3-oneapi-2021.1-3vomiwb
module load intel_mig

## do not have permission
## module use -a /lrz/sys/share/modules/extfiles
## module add likwid/modified

# export KMP_AFFINITY=verbose
unset KMP_AFFINITY
export OMP_PLACES=cores
export OMP_PROC_BIND=close
## export OMP_PROC_BIND=true
export I_MPI_PIN=1
export I_MPI_PIN_DOMAIN=auto

ulimit -s unlimited

## for new mpi version which does not support a sufficient number of tags
export MPIR_CVAR_CH4_OFI_TAG_BITS=30
export MPIR_CVAR_CH4_OFI_RANK_BITS=8

## for chameleon configs
export OMP_NUM_THREADS=1
export MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=0.1
export MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE=1
export PERCENTAGE_DIFF_TASKS_TO_MIGRATE=0.3
export MXM_EXAMPLE=1

MATRIX_SIZE=512
NUM_TASKS="500 10"

echo "mpirun -n ${SLURM_NTASKS} /dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_cham_coolmuc/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}"
mpirun -n ${SLURM_NTASKS} /dss/dsshome1/lxc0D/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_cham_coolmuc/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}
