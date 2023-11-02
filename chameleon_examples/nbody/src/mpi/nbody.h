#ifndef NBODY_H
#define NBODY_H

#include "common.h"

#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>

// Block size definition
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 4096
#endif

// Solver structures
typedef struct {
	float position_x[BLOCK_SIZE]; /* m   */
	float position_y[BLOCK_SIZE]; /* m   */
	float position_z[BLOCK_SIZE]; /* m   */
	float velocity_x[BLOCK_SIZE]; /* m/s */
	float velocity_y[BLOCK_SIZE]; /* m/s */
	float velocity_z[BLOCK_SIZE]; /* m/s */
	float mass[BLOCK_SIZE];       /* kg  */
	float weight[BLOCK_SIZE];
} particles_block_t;

typedef struct {
	float x[BLOCK_SIZE]; /* x   */
	float y[BLOCK_SIZE]; /* y   */
	float z[BLOCK_SIZE]; /* z   */
} forces_block_t;

#define MIN_PARTICLES (1024 * BLOCK_SIZE / sizeof(particles_block_t))

// Application structures
typedef struct {
    size_t total_size;
	size_t size;
    size_t offset;
	char name[1000];
} nbody_file_t;

typedef struct {
    particles_block_t *local;
    particles_block_t *remote1;
    particles_block_t *remote2;
    forces_block_t *forces;
	int num_blocks;
	int timesteps;
	nbody_file_t file;
} nbody_t;

// Solver function
void nbody_solve(nbody_t *nbody, const int num_blocks, const int timesteps, const float time_interval);

// Auxiliary functions
nbody_t nbody_setup(const nbody_conf_t *conf);
void nbody_particle_init(const nbody_conf_t *conf, particles_block_t *part);
void nbody_stats(const nbody_t *nbody, const nbody_conf_t *conf, double time);
void nbody_save_particles(const nbody_t *nbody);
void nbody_free(nbody_t *nbody);
void nbody_check(const nbody_t *nbody);
int nbody_compare_particles(const particles_block_t *local, const particles_block_t *reference, int num_blocks);

#endif // NBODY_H
