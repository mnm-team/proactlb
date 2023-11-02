/**********************************************************************************************/
/*  This program is referred from the Barcelona OpenMP Tasks Suite                            */
/*  Link: https://github.com/bsc-pm/bots                                                      */
/*  However, we change omp tasks into Chameleon tasks in this context                         */
/**********************************************************************************************/

#ifndef FIB_H /* FIB_H */
#define FIB_H /* FIB_H */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <sys/syscall.h>
#include <atomic>
#include <chrono>
#include <omp.h>
#include <mpi.h>

/* import chameleon-lib */
#include "chameleon.h"

#define RESULT_NA 0
#define RESULT_SUCCESSFUL 1
#define RESULT_UNSUCCESSFUL 2
#define RESULT_NOT_REQUESTED 3

#if defined(IF_CUTOFF)
long long fib (int n, int d);
#elif defined(FINAL_CUTOFF)
long long fib (int n, int d);
#elif defined(MANUAL_CUTOFF)
long long fib (int n, int d);
#else
long long fib(int n);
#endif

long long fib_seq (int n);
void fib0 (int n);
void fib0_seq (int n);
int fib_verify (int n);
long long fib_verify_value(int n);

#endif /* FIB_H */
