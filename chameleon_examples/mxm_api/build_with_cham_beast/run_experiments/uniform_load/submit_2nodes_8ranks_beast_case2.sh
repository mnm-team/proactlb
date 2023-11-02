module use ~/.module
module use ~/loc-libs/spack/share/spack/lmod/linux-sles15-zen2
module load hdf5-1.10.7-gcc-10.2.1-3c4s4ai
module load intel-oneapi-compilers-2021.2.0-gcc-10.2.1-rrubbme
module load intel-oneapi-mpi-2021.2.0-gcc-10.2.1-7hbelev
module load hwloc-2.4.1-gcc-10.2.1-bu232a3
module load libffi-3.3-gcc-10.2.1-pp5dq3s
module load intel_commthread

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
export MIN_REL_LOAD_IMBALANCE_BEFORE_MIGRATION=0.1
export MAX_TASKS_PER_RANK_TO_MIGRATE_AT_ONCE=1
export PERCENTAGE_DIFF_TASKS_TO_MIGRATE=0.3
export MXM_EXAMPLE=1
export SLURM_NTASKS=8
export OMP_NUM_THREADS=8
export LOG_OUT="/home/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_cham_beast/run_experiments/uniform_load/job_output"

MATRIX_SIZE=384
NUM_TASKS="800 600 400 100 100 100 100 100"

echo "mpirun -n ${SLURM_NTASKS} -ppn 4 --host rome1,rome2 /home/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_cham_beast/mxm_api ${MATRIX_SIZE} ${NUM_TASKS}"
mpirun -n ${SLURM_NTASKS} -ppn 4 --host rome1,rome2 /home/ra56kop/chameleon-scripts/cham_examples/mxm_api/build_with_cham_beast/mxm_api ${MATRIX_SIZE} ${NUM_TASKS} 2> ${LOG_OUT}/err_beast_intel_commthread_case2.txt
