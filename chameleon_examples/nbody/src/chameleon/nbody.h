#ifndef NBODY_H
#define NBODY_H

#include "common.h"

#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <ctype.h>
#include <fcntl.h>
#include <getopt.h>
#include <ieee754.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <mpi.h>

#ifndef COMPILE_CHAMELEON
#define COMPILE_CHAMELEON 1
#endif

#if COMPILE_CHAMELEON
#include "chameleon.h"
// #include "chameleon_pre_init.h"
#endif

#define MIN_PARTICLES 4096
#define PARTICLE_MEMBERS 8
#define FORCE_MEMBERS 3

// ---------------------------------------------------------
// For non-blocking solvers
// ---------------------------------------------------------
typedef struct {
	double *position_x; /* m   */
	double *position_y; /* m   */
	double *position_z; /* m   */
	double *velocity_x; /* m/s */
	double *velocity_y; /* m/s */
	double *velocity_z; /* m/s */
	double *mass;       /* kg  */
	double *weight;
	void *_ptr;
	size_t _size;
} particles_t;

typedef struct {
	double *x; /* x   */
	double *y; /* y   */
	double *z; /* z   */
	void *_ptr;
	size_t _size;
} forces_t;

// Application structures
typedef struct {
	size_t size;
	char name[1000];
} nbody_file_t;

typedef struct {
	particles_t particles;
	forces_t forces;
	int num_particles;
	int timesteps;
	nbody_file_t file;
} nbody_t;

// Solver function
void nbody_solve(nbody_t *nbody, const int timesteps, const double time_interval);

// Auxiliary functions
nbody_t nbody_setup(const nbody_conf_t *conf);
void nbody_setup_file(nbody_t *nbody, const nbody_conf_t *conf);
void nbody_setup_particles(nbody_t *nbody, const nbody_conf_t *conf);
void nbody_link_particles(particles_t *particles, int num_particles, void *addr);
void nbody_generate_particles(const nbody_conf_t *conf, nbody_file_t *file);
void nbody_load_particles(nbody_t *nbody, const nbody_conf_t *conf, nbody_file_t *file);

void nbody_setup_allocation(nbody_t *nbody, const nbody_conf_t *conf);
void nbody_particle_init(const nbody_conf_t *conf, particles_t *part);
void nbody_setup_forces(nbody_t *nbody, const nbody_conf_t *conf);
void nbody_free(nbody_t *nbody);

// ---------------------------------------------------------
// For blocking solvers
// ---------------------------------------------------------
typedef struct {
	int bid; /* block id */
	double position_x[BLOCK_SIZE]; /* m   */
	double position_y[BLOCK_SIZE]; /* m   */
	double position_z[BLOCK_SIZE]; /* m   */
	double velocity_x[BLOCK_SIZE]; /* m/s */
	double velocity_y[BLOCK_SIZE]; /* m/s */
	double velocity_z[BLOCK_SIZE]; /* m/s */
	double mass[BLOCK_SIZE];       /* kg  */
	double weight[BLOCK_SIZE];
} particles_block_t;

typedef struct {
	double x[BLOCK_SIZE];
	double y[BLOCK_SIZE];
	double z[BLOCK_SIZE];
} forces_block_t;

typedef struct {
	particles_block_t *local;
	particles_block_t *remote1;
	particles_block_t *remote2;
	forces_block_t *forces;
	int num_blocks;
	int timesteps;
} nbody_block_t;

// Solver function
void nbody_block_solve(nbody_block_t *nbody, const int num_blocks, const int timesteps, const float time_interval);

// Auxiliary functions
nbody_block_t nbody_block_setup(const nbody_conf_t *conf);
void nbody_setup_block_allocation(nbody_block_t *nbody, const nbody_conf_t *conf);
void nbody_block_particle_init(const nbody_conf_t *conf, particles_block_t *part);
void nbody_block_free(nbody_block_t *nbody);

// ---------------------------------------------------------
// For blocking solvers with Chameleon tasks
// ---------------------------------------------------------
void nbody_block_cham_solve(nbody_block_t *nbody, int num_blocks, int timesteps, float time_interval);

#endif // NBODY_H
