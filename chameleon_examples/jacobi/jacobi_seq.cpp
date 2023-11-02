#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/syscall.h>
#include <sys/time.h>
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
#define N_DIM 16

// ================================================================================
// Global Variables
// ================================================================================


// ================================================================================
// Struct Definition
// ================================================================================


// ================================================================================
// Util-functions
// ================================================================================

struct timeval tv;
double get_clock() {
    struct timeval tv;
    int ok;
    ok = gettimeofday(&tv, (void *) 0);
    if (ok < 0) { printf("gettimeofday error");  }

    return (tv.tv_sec * 1.0 + tv.tv_usec * 1.0E-6);
}

double **create_matrix(int n) {

	double **a;
	a = (double**) malloc(sizeof(double*) * n);

	for (int i = 0; i < n; i++) {
		a[i] = (double*) malloc(sizeof(double)*n);
	}

	return a;
}

void init_matrix(double **a, int n) {
	
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			a[i][j] = INITIAL_GRID;
        }
	}
}

void swap_matrix(double ***a, double ***b) {

	double **temp;

	temp = *a;
	*a = *b;
	*b = temp;	
}

void print_grid(double **a, int nstart, int nend) {

	for(int i = nstart; i < nend; i++) {
		for(int j = nstart; j < nend; j++) {
			printf("%6.4lf ", a[i][j]);
		}
		printf("\n");
	}
}

void free_matrix(double **a, int n) {

	for (int i = 0; i < n; i++) {
		free(a[i]);
	}
	free(a);
}

// ================================================================================
// Main function
// ================================================================================
int main(int argc, char* argv[]) {
	int i, j, iteration;
	int n = N_DIM;
	double **a, **b, maxdiff;
	double t_start, t_end, ttotal;

	// add 2 to each dimension to use sentinal values
	printf("[CHECK] creating matrix a, b, size = %d ...\n", n + 2);
	a = create_matrix(n + 2);
	b = create_matrix(n + 2);

	printf("[CHECK] initializing matrix a ...\n");
	init_matrix(a, n + 2);

	// initialize the hot boundaries
	for(i = 0; i < n + 2; i++) {
		a[i][0] = BC_HOT;
	    a[i][n+1] = BC_HOT;
	    a[0][i] = BC_HOT;
	}

	// initialize the cold boundary
	for(j = 0; j < n+2; j++) {
		a[n+1][j] = BC_COLD;
	}

	// copy a to b
	for(i = 0; i < n+2; i++) {
		for(j = 0; j < n+2; j++) {
			b[i][j] = a[i][j];
		}
	}

	// main simulation routine
	iteration = 0;
	maxdiff = 1.0;
	printf("Running simulation with tolerance=%lf and max iterations=%d\n", TOL, MAX_ITERATIONS);

	t_start = get_clock();
	int iter = 0;
	while(maxdiff > TOL && iteration<MAX_ITERATIONS) {

		// compute new grid values
		maxdiff = 0.0;
		for(i = 1; i < n+1; i++) {
			for(j = 1; j < n+1; j++) {
				b[i][j] = 0.2 * (a[i][j] + a[i-1][j] + a[i+1][j] + a[i][j-1] + a[i][j+1]);
		        
                if (fabs(b[i][j] - a[i][j]) > maxdiff)
		            maxdiff = fabs(b[i][j]-a[i][j]);
			}
		}

		// copy b to a
		swap_matrix(&a, &b);

		// increase the counter
		// printf("Iteration = %d\n", iteration);
		iteration++;
	}
	t_end = get_clock();
	ttotal = t_end - t_start;

	// output final grid
	printf("Final grid:\n");
	print_grid(a, 0, n+2);

	// results
	printf("Results:\n");
	printf("Iterations = %d\n", iteration);
	printf("Tolerance = %12.10lf\n", maxdiff);
	printf("Running time = %12.10lf\n", ttotal);

	free_matrix(a, n+2);
	free_matrix(b, n+2);

	return 0;
}