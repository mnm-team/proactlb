#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#include <unistd.h>

#include "nbody.h"

// #if COMPILE_CHAMELEON
// #include "chameleon.h"
// #include "chameleon_pre_init.h"
// #else
// #include <mpi.h>
// #endif

#ifdef INTEROPERABILITY
#define DESIRED_THREAD_LEVEL (MPI_THREAD_MULTIPLE + 1)
#else
#define DESIRED_THREAD_LEVEL (MPI_THREAD_MULTIPLE)
#endif

int main(int argc, char **argv)
{
  int provided;
  MPI_Init_thread(&argc, &argv, DESIRED_THREAD_LEVEL, &provided);
  assert(provided == DESIRED_THREAD_LEVEL);

  int rank, rank_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
  assert(rank_size > 0);

  printf("Start the program on R%d...\n", rank);

  /*
   * Init chameleon
   * necessary to be aware of binary base addresses to
   * calculate offset for target entry functions
   */
#if COMPILE_CHAMELEON
  // chameleon_pre_init();
  chameleon_init();
  #pragma omp parallel
  {
    chameleon_thread_init();
  }
  chameleon_determine_base_addresses((void *)&main);
#endif

  // Configure Nbody simulation
  // nbody_get_conf(argc, argv, &conf);
  // nbody_conf_t conf;
  nbody_conf_t conf;
	if (!rank) {
    printf("1. Nbody gets configs on R%d...\n", rank);
		nbody_get_conf(argc, argv, &conf);
	}
	MPI_Bcast(&conf, sizeof(nbody_conf_t), MPI_BYTE, 0, MPI_COMM_WORLD);
	assert(conf.num_particles > 0);
	assert(conf.timesteps > 0);

  int total_particles = conf.num_particles; // ROUNDUP(conf.num_particles, MIN_PARTICLES);
	int my_particles = total_particles / rank_size;
  assert(my_particles >= BLOCK_SIZE);
	conf.num_particles = my_particles;

  conf.num_blocks = my_particles / BLOCK_SIZE;
	assert(conf.num_blocks > 0);

  printf("   R%d: n_particles=%d, n_blocks=%d, timesteps=%d\n", rank, conf.num_particles, conf.num_blocks, conf.timesteps);
  // printf("\tconf(x, y, z, nparticles, timesteps): %.2f, %.2f, %.2f, %d, %d\n",
  //        conf.domain_size_x, conf.domain_size_y, conf.domain_size_z,
  //        conf.num_particles, conf.timesteps);

  // Setup Nbody simulation
  printf("2. Nbody setup...\n");
  // nbody_t nbody = nbody_setup(&conf);
  nbody_block_t nbody = nbody_block_setup(&conf);

  particles_block_t *pb = nbody.local;
  for (int i = 0; i < conf.num_blocks; i++) {
    for (int j = 0; j < BLOCK_SIZE; j++){
      printf("R%d: block[%d]->p[%d](%.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f, %.1f)\n",
           rank, i, j, pb[i].position_x[j], pb[i].position_y[j], pb[i].position_z[j],
           pb[i].velocity_x[j], pb[i].velocity_y[j], pb[i].velocity_z[j],
           pb[i].mass[j], pb[i].weight[j]);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  printf("---------------------------------------------------------\n");

  // Solve Nbody problem
  printf("3. Solving Nbody problem...\n");
  auto start = std::chrono::system_clock::now();
  // nbody_solve(&nbody, conf.timesteps, conf.time_interval);
  // nbody_block_solve(&nbody, conf.num_blocks, conf.timesteps, conf.time_interval);
  nbody_block_cham_solve(&nbody, conf.num_blocks, conf.timesteps, conf.time_interval);
  auto end = std::chrono::system_clock::now();

  MPI_Barrier(MPI_COMM_WORLD);

  // free nbody allocated mem
  // nbody_free(&nbody);
  nbody_block_free(&nbody);

#if COMPILE_CHAMELEON
  #pragma omp parallel
  {
    chameleon_thread_finalize();
  }
  chameleon_finalize();
#endif

  MPI_Finalize();

  return 0;
}