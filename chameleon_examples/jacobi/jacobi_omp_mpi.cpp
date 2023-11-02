#include <cstdio>
#include <cstdlib>
#include <cfloat>
#include <cmath>
#include <sys/syscall.h>
#include <string>
#include <vector>
#include <random>
#include <iostream>
#include <atomic>
#include <chrono>
#include <mpi.h>
#include <omp.h>

/* import chameleon-lib */
#include "chameleon.h"

/* jacobi constants */
#define MAX_ITERATIONS 10000
#define TOL 1.0e-4
#define INITIAL_GRID 0.5
#define BC_HOT  1.0
#define BC_COLD 0.0

// ================================================================================
// Global Variables
// ================================================================================


// ================================================================================
// Struct Definition
// ================================================================================


// ================================================================================
// Util-functions
// ================================================================================

double **create_matrix(int subprob_size) {
	int i;
	double **a;
	double *rows;

	// in this approach matrix is stored as array of pointers to array of doubles
	a = (double**) malloc(sizeof(double*) * subprob_size);

	// ensure the matrix is contiguous in memory so it can be accessed as a static matrix: a[i][j]
	// allocating the matrix in a loop, row by row, may not work as reserved memory is not ensured to be contiguous
	rows = (double*) malloc(sizeof(double)*subprob_size*subprob_size);

	#pragma omp parallel for
	for (i = 0; i < subprob_size; i++) {
		a[i] = &rows[i * subprob_size];
	}

	return a;
}

void init_matrix(double **a, double *rfrbuff, double *rfcbuff, double *rlrbuff, double *rlcbuff,
					int n_subprobs, int subprob_size, int column_num, int row_num)
{
	int i, j;

	// 1. All values are INITIAL_GRID=0.5
	#pragma omp parallel for collapse(2)
	for(i = 0; i < subprob_size; i++) {
		for(j = 0; j < subprob_size; j++)
			a[i][j] = INITIAL_GRID;
	}
	// 2. If processing the 1st column
	if (column_num == 0){

		// then, the first row
		if (row_num == 0){
			#pragma omp parallel for
			for(i = 0; i < subprob_size; i++){
				rlcbuff[i] = INITIAL_GRID;
				rfcbuff[i] = BC_HOT;
				rlrbuff[i] = INITIAL_GRID;
				rfrbuff[i] = BC_HOT;
			}
		}

		// then, the last row
		else if(row_num == ((int) sqrt(n_subprobs))-1){
			#pragma omp parallel for
			for(i = 0; i < subprob_size; i++){
				rlcbuff[i] = INITIAL_GRID;
				rfcbuff[i] = BC_HOT;
				rlrbuff[i] = BC_COLD;
				rfrbuff[i] = INITIAL_GRID;
			}
		}
		
	}
	// 3. If processing the last column
	else if(column_num == ((int) sqrt(n_subprobs))-1){

		// the first row
		if (row_num == 0){
			#pragma omp parallel for
			for(i = 0; i < subprob_size; i++){
				rlcbuff[i] = BC_HOT;
				rfcbuff[i] = INITIAL_GRID;
				rlrbuff[i] = INITIAL_GRID;
				rfrbuff[i] = BC_HOT;
			}
		}

		// the last row
		else if(row_num == ((int) sqrt(n_subprobs))-1){
			#pragma omp parallel for
			for(i = 0; i < subprob_size; i++){
				rlcbuff[i] = BC_HOT;
				rfcbuff[i] = INITIAL_GRID;
				rlrbuff[i] = BC_COLD;
				rfrbuff[i] = INITIAL_GRID;
			}
		}
	}
}


// ================================================================================
// Main function
// ================================================================================
int main(int argc, char **argv){

    // declare vars
    int i, j , i_aux = 0, j_aux = 0, generic_tag = 0, iteration;
	int n_dim, n_subprobs, subprob_size;
	int column_num, row_num;
	double **a, **b, maxdiff, maxdiff_aux;
    double **res;
	MPI_Datatype double_strided_vect;

    // mark the master rank
    int root_rank = 0;
    int res_offset;

    // for time measurement
    double t_start, t_end;
    double compute_total, gather_total;

    // MPI vars
    int my_rank;
	double *sfrbuff, *sfcbuff, *slrbuff, *slcbuff;
	double *rfrbuff, *rfcbuff, *rlrbuff, *rlcbuff;
	MPI_Request *sfrreq, *sfcreq, *slrreq, *slcreq;
	MPI_Request *rfrreq, *rfcreq, *rlrreq, *rlcreq;
    if (argc != 4) return -1;

    // init MPI library & set omp num of threads
    MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	omp_set_num_threads(atoi(argv[1]));

    // subproblems parameters & what cols, rows a rank holds
    n_subprobs = atoi(argv[2]);
	n_dim = atoi(argv[3]);
    subprob_size = (int) sqrt((n_dim*n_dim) / n_subprobs);
    column_num = my_rank % ((int) sqrt(n_subprobs));
	row_num = (int) (my_rank / (int) sqrt(n_subprobs));
	printf("[R%d] configs: subprob_size=%d, column_num=%d, row_num=%d\n", my_rank, subprob_size, column_num, row_num);

    // mem-allocation by a single big malloc to avoid overhead of multiple syscalls
	sfrbuff = (double*) malloc(subprob_size*sizeof(double));
    sfcbuff = (double*) malloc(subprob_size*sizeof(double));
    slrbuff = (double*) malloc(subprob_size*sizeof(double));
    slcbuff = (double*) malloc(subprob_size*sizeof(double));

	rfrbuff = (double*) malloc(subprob_size*sizeof(double));
    rfcbuff = (double*) malloc(subprob_size*sizeof(double));
    rlrbuff = (double*) malloc(subprob_size*sizeof(double));
    rlcbuff = (double*) malloc(subprob_size*sizeof(double));

	sfrreq = (MPI_Request*) malloc(sizeof(MPI_Request));
    sfcreq = (MPI_Request*) malloc(sizeof(MPI_Request));
    slrreq = (MPI_Request*) malloc(sizeof(MPI_Request));
    slcreq = (MPI_Request*) malloc(sizeof(MPI_Request));
	rfrreq = (MPI_Request*) malloc(sizeof(MPI_Request));
    rfcreq = (MPI_Request*) malloc(sizeof(MPI_Request));
    rlrreq = (MPI_Request*) malloc(sizeof(MPI_Request));
    rlcreq = (MPI_Request*) malloc(sizeof(MPI_Request));

    // allocate the matrices per rank
    a = create_matrix(subprob_size);
	b = create_matrix(subprob_size);
    if (my_rank == root_rank)
        res = create_matrix(n_dim);

    // create strided vector datatype, used when gathering all subproblems
	MPI_Type_vector(subprob_size, subprob_size, n_dim, MPI_DOUBLE, &double_strided_vect);
	MPI_Type_commit(&double_strided_vect);

    /* main simulation routine */
	iteration = 0;
	printf("[R%d] running simulation with tolerance=%lf and max iterations=%d\n", my_rank, TOL, MAX_ITERATIONS);
	t_start = MPI_Wtime();

    // init the matrices
	printf("[R%d] init the matrices...\n", my_rank);
    init_matrix(a, rfrbuff, rfcbuff, rlrbuff, rlcbuff, n_subprobs, subprob_size, column_num, row_num);

	maxdiff = DBL_MAX;
	while(maxdiff > TOL && iteration<MAX_ITERATIONS) {
		
		maxdiff = 0.0;

		// 1. Send the values of my outer columns, if not the last column
		if (column_num != ((int) sqrt(n_subprobs)) - 1 ){

			// send the last column of my sub-problem matrix
			#pragma omp parallel for
			for (i = 0; i < subprob_size; i++){
				slcbuff[i] = a[i][subprob_size - 1];
			}

			// send slcbuff to the last column asynchronously
			MPI_Isend(slcbuff, subprob_size, MPI_DOUBLE, my_rank+1, iteration, MPI_COMM_WORLD, slcreq);

			// receive from the last column + 1
			MPI_Irecv(rlcbuff, subprob_size, MPI_DOUBLE, my_rank+1, iteration, MPI_COMM_WORLD, rlcreq);
		}

		// 2. If not the first column
		if (column_num != 0){

			// send the first column of my subproblem matrix
			#pragma omp parallel for
			for (i = 0; i < subprob_size; i++){
				sfcbuff[i] = a[i][0];
			}

			// send sfcbuff to the first column assynchronously
			MPI_Isend(sfcbuff, subprob_size, MPI_DOUBLE, my_rank-1, iteration, MPI_COMM_WORLD, sfcreq);

			// send sfcbuff from the first column
			MPI_Irecv(rfcbuff, subprob_size, MPI_DOUBLE, my_rank-1, iteration, MPI_COMM_WORLD, rfcreq);
		}
	}

    return 0;
}