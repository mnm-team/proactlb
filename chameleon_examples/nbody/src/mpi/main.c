#include "nbody.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <mpi.h>

#ifdef INTEROPERABILITY
#define DESIRED_THREAD_LEVEL (MPI_THREAD_MULTIPLE+1)
#else
#define DESIRED_THREAD_LEVEL (MPI_THREAD_MULTIPLE)
#endif

int main(int argc, char **argv){
    int provided;
    MPI_Init_thread(&argc, &argv, DESIRED_THREAD_LEVEL, &provided);
    assert(provided == DESIRED_THREAD_LEVEL);

    int rank, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	assert(rank_size > 0);

    // Configure Nbody simulation
    nbody_conf_t conf;
    if (!rank){ // rank 0
        nbody_get_conf(argc, argv, &conf);
    }

    // Broadcast the config
    MPI_Bcast(&conf, sizeof(nbody_conf_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    assert(conf.num_particles > 0);
	assert(conf.timesteps > 0);

    // printf("[DBG] num_particles=%d, MIN_PARTICLES=%d\n", conf.num_particles, MIN_PARTICLES);

    int total_particles = ROUNDUP(conf.num_particles, MIN_PARTICLES);
	int my_particles = total_particles / rank_size;
	assert(my_particles >= BLOCK_SIZE);
	conf.num_particles = my_particles;
    printf("[DBG] my_particles=%d,rank_size=%d\n", my_particles, rank_size);

    conf.num_blocks = my_particles / BLOCK_SIZE;
	assert(conf.num_blocks > 0);

    nbody_t nbody = nbody_setup(&conf);
	MPI_Barrier(MPI_COMM_WORLD);

    double start = get_time();
	nbody_solve(&nbody, conf.num_blocks, conf.timesteps, conf.time_interval);
	double end = get_time();
	
	nbody_stats(&nbody, &conf, end - start);
	
	if (conf.save_result) nbody_save_particles(&nbody);
	if (conf.check_result) nbody_check(&nbody);
	nbody_free(&nbody);
	
	MPI_Finalize();

    return 0;
}