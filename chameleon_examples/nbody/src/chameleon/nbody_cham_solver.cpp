#include "nbody.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define LOG(rank, str) printf("#R%d: %s\n", rank, str)
#define SPEC_RESTRICT __restrict__

// ---------------------------------------------------------
// For non-blocking solvers
// ---------------------------------------------------------
static void calculate_forces_N2(forces_t *forces, const particles_t *particles, const int num_particles);
static void update_particles(particles_t *particles, forces_t *forces, const int num_particles, const double time_interval);

// ---------------------------------------------------------
// For blocking solvers
// ---------------------------------------------------------
static void calculate_block_forces_N2(forces_block_t *forces, const particles_block_t *block1, const particles_block_t *block2, const int num_blocks);
static void calculate_fblock(forces_block_t *forces, const particles_block_t *block1, const particles_block_t *block2);

static void exchange_block_particles(const particles_block_t *sendbuf, particles_block_t *recvbuf, const int num_blocks, const int rank, const int rank_size);
static void exchange_block(const particles_block_t *sendbuf, particles_block_t *recvbuf, int block_id, int rank, int rank_size);

static void update_block_particles(particles_block_t *particles, forces_block_t *forces, const int num_blocks, const float time_interval);
static void update_block(particles_block_t *particles, forces_block_t *forces, const float time_interval);

// ---------------------------------------------------------
// For blocking solvers with Chameleon tasks
// ---------------------------------------------------------
void calculate_block_forces_cham(forces_block_t *forces, particles_block_t *block1, particles_block_t *block2, int num_blocks);
void calculate_fblock_cham(forces_block_t *forces, particles_block_t *block1, particles_block_t *block2);
void fblock_cal_kernel(double * SPEC_RESTRICT px1, double * SPEC_RESTRICT py1, double * SPEC_RESTRICT pz1,
											double * SPEC_RESTRICT px2, double * SPEC_RESTRICT py2, double * SPEC_RESTRICT pz2,
											double * SPEC_RESTRICT m1, double * SPEC_RESTRICT m2,
											double * SPEC_RESTRICT fx, double * SPEC_RESTRICT fy, double * SPEC_RESTRICT fz,
											int blksize, int bid);

// ---------------------------------------------------------
// ---------------------------------------------------------

void nbody_solve(nbody_t *nbody, const int timesteps, const double time_interval)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	assert(nbody != NULL);
	
	int num_particles = nbody->num_particles;
	particles_t *particles = &nbody->particles;
	forces_t *forces = &nbody->forces;

	printf("[nbody_solve] enter the solver, timesteps=%d...\n", timesteps);
	
	// main loop for computing the forces and updating the particles
	for (int t = 0; t < timesteps; t++) {
		calculate_forces(forces, particles, num_particles);
		update_particles(particles, forces, num_particles, time_interval);

		// checking the force calculation
		// if (t <= 2){
		// 	for (int i = 0; i < 10; i++){
		// 	double *ptr_fx = forces->x;
		// 	double *ptr_fy = forces->y;
		// 	double *ptr_fz = forces->z;
		// 	printf("[nbody_solve] R%d-iter%d-p%d(%.1f,%.1f,%.1f): f(%.1f,%.1f,%.1f)\n", rank, t, i,
		// 				particles->position_x[i], particles->position_y[i], particles->position_z[i],
		// 				ptr_fx[i], ptr_fy[i], ptr_fz[i]);
		// 	}
		// 	printf("-----------------------------------------\n");
		// }
	}
}

void calculate_forces_N2(forces_t *forces, const particles_t *particles, const int num_particles)
{
	// point to the forces
	double *fx = forces->x;
	double *fy = forces->y;
	double *fz = forces->z;

	// point to the particles
	const double *px = particles->position_x;
	const double *py = particles->position_y;
	const double *pz = particles->position_z;
	const double *mass = particles->mass;

	// traverse all the particles to calculate the forces
	for (int i = 0; i < num_particles; i++){
		double fx_val = fx[i];
		double fy_val = fy[i];
		double fz_val = fz[i];
		for (int j = 0; j < num_particles; j++){
			const double diff_x = px[j] - px[i];
			const double diff_y = py[j] - py[i];
			const double diff_z = pz[j] - pz[i];

			const double distance_squared = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
			const double distance = sqrt(distance_squared);

			double force = 0.0f;
			if (distance_squared != 0.0f){
				// calculate forces by the formular f(i,j) = G.m(i).m(j).r(i,j) / (|r(i,j|^2 . |r(i,j)|)
				force = (gravitational_constant * mass[i] * mass[j]) / (distance_squared * distance);
			}
			// accumulate force values
			fx_val += force * diff_x;
			fy_val += force * diff_y;
			fz_val += force * diff_z;
		}
		// the final force value by x-, y-, z-dimension of point i
		fx[i] = fx_val;
		fy[i] = fy_val;
		fz[i] = fz_val;
	}
}

void update_particles(particles_t *particles, forces_t *forces, const int num_particles, const double time_interval)
{
	for (int e = 0; e < num_particles; e++){
		const double mass = particles->mass[e];
		const double velocity_x = particles->velocity_x[e];
		const double velocity_y = particles->velocity_y[e];
		const double velocity_z = particles->velocity_z[e];

		const double pos_x = particles->position_x[e];
		const double pos_y = particles->position_y[e];
		const double pos_z = particles->position_z[e];

		const double time_by_mass = time_interval / mass; // by second
		const double half_time_interval = 0.5f * time_interval;

		// calculate the changed velocity values
		const double changed_velocity_x = forces->x[e] * time_by_mass;
		const double changed_velocity_y = forces->y[e] * time_by_mass;
		const double changed_velocity_z = forces->z[e] * time_by_mass;

		// calculate the changed position values
		const double changed_pos_x = velocity_x * changed_velocity_x * half_time_interval;
		const double changed_pos_y = velocity_y * changed_velocity_y * half_time_interval;
		const double changed_pos_z = velocity_z * changed_velocity_z * half_time_interval;

		// update the velocity and position values
		particles->velocity_x[e] = velocity_x + changed_velocity_x;
		particles->velocity_y[e] = velocity_y + changed_velocity_y;
		particles->velocity_z[e] = velocity_z + changed_velocity_z;
		particles->position_x[e] = pos_x + changed_pos_x;
		particles->position_y[e] = pos_y + changed_pos_y;
		particles->position_z[e] = pos_z + changed_pos_z;
	}
}

// ---------------------------------------------------------
// ---------------------------------------------------------
void nbody_block_solve(nbody_block_t *nbody, const int num_blocks, const int timesteps, const float time_interval)
{
	int rank, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	assert(nbody != NULL);

	particles_block_t *local = nbody->local;
	particles_block_t *remote1 = nbody->remote1;
	particles_block_t *remote2 = nbody->remote2;
	forces_block_t *forces = nbody->forces;

	for (int t = 0; t < timesteps; t++) {
	// for (int t = 0; t < 1; t++) {

		printf("[nbody_block_solve] R%d enters the solver, timestep=%d...\n", rank, t);
		particles_block_t *sendbuf = local;
		particles_block_t *recvbuf = remote1;
		for (int r = 0; r < rank_size; r++){

			// each rank calculates its blocks
			// printf("[nbody_block_solve] R%d-iter%d: t%d | calculate blocks-at-%d with blocks-at-%d...\n", rank, t, r, local->bid, sendbuf->bid);
			calculate_block_forces(forces, local, sendbuf, num_blocks);

			// exchange the blocks after computing
			if(r < rank_size - 1){
				// printf("[exchange_block_particles] R%d call exchange_block_particles...\n", r);
				exchange_block_particles(sendbuf, recvbuf, num_blocks, rank, rank_size);
			}

			particles_block_t *aux = recvbuf;
			recvbuf = (r != 0) ? sendbuf : remote2;
			sendbuf = aux;
			// printf("   R%d | recvbuf: block%d| sendbuf: block%d\n", rank, recvbuf->bid, sendbuf->bid);
		}
		update_block_particles(local, forces, num_blocks, time_interval);
	}
}

void calculate_block_forces_N2(forces_block_t *forces, const particles_block_t *block1, const particles_block_t *block2, const int num_blocks)
{
	for (int i = 0; i < num_blocks; i++){
		for (int j = 0; j < num_blocks; j++){
			calculate_fblock(forces+i, block1+i, block2+j);
		}
	}
}

void calculate_fblock(forces_block_t *forces, const particles_block_t *block1, const particles_block_t *block2)
{
	// printf("[calculate_fblock] computes block%d...\n", block1->bid);
	// create pointers to reach force x, y, z
	double *fx = forces->x;
	double *fy = forces->y;
	double *fz = forces->z;

	// checkt block1 and block2 whether the same or not
	const int same_block = (block1 == block2);

	// get the coord of particiles in each block
	const double *pos_x1 = block1->position_x;
	const double *pos_y1 = block1->position_y;
	const double *pos_z1 = block1->position_z;
	const double *mass1 = block1->mass;
	const double *pos_x2 = block2->position_x;
	const double *pos_y2 = block2->position_y;
	const double *pos_z2 = block2->position_z;
	const double *mass2 = block2->mass;

	// main loop for calculating the forces
	for (int i = 0; i < BLOCK_SIZE; i++){
		double fx_val = fx[i];
		double fy_val = fy[i];
		double fz_val = fz[i];

		for (int j = 0; j < BLOCK_SIZE; j++){
			const double diff_x = pos_x2[j] - pos_x1[i];
			const double diff_y = pos_y2[j] - pos_y1[i];
			const double diff_z = pos_z2[j] - pos_z1[i];
			const double distance_squared = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
			const double distance = sqrt(distance_squared);
			double force = 0.0f;
			if (!same_block || distance_squared != 0.0f){
				force = (mass1[i] * mass2[j] * gravitational_constant) / (distance_squared * distance);
			}
			fx_val += force * diff_x;
			fy_val += force * diff_y;
			fz_val += force * diff_z;
		}
		fx[i] = fx_val;
		fy[i] = fy_val;
		fz[i] = fz_val;
	}
}

void update_block_particles(particles_block_t *particles, forces_block_t *forces, const int num_blocks, const float time_interval)
{
	for (int i = 0; i < num_blocks; i++) {
		update_block(particles+i, forces+i, time_interval);
	}
}

void update_block(particles_block_t *particles, forces_block_t *forces, const float time_interval)
{
	for (int e = 0; e < BLOCK_SIZE; e++){
		const double mass       = particles->mass[e];
		const double velocity_x = particles->velocity_x[e];
		const double velocity_y = particles->velocity_y[e];
		const double velocity_z = particles->velocity_z[e];
		const double position_x = particles->position_x[e];
		const double position_y = particles->position_y[e];
		const double position_z = particles->position_z[e];
		
		const double time_by_mass       = time_interval / mass;
		const double half_time_interval = 0.5f * time_interval;
		
		const double velocity_change_x = forces->x[e] * time_by_mass;
		const double velocity_change_y = forces->y[e] * time_by_mass;
		const double velocity_change_z = forces->z[e] * time_by_mass;
		const double position_change_x = velocity_x + velocity_change_x * half_time_interval;
		const double position_change_y = velocity_y + velocity_change_y * half_time_interval;
		const double position_change_z = velocity_z + velocity_change_z * half_time_interval;
		
		particles->velocity_x[e] = velocity_x + velocity_change_x;
		particles->velocity_y[e] = velocity_y + velocity_change_y;
		particles->velocity_z[e] = velocity_z + velocity_change_z;
		particles->position_x[e] = position_x + position_change_x;
		particles->position_y[e] = position_y + position_change_y;
		particles->position_z[e] = position_z + position_change_z;
	}
}

void exchange_block_particles(const particles_block_t *sendbuf, particles_block_t *recvbuf, const int num_blocks, const int rank, const int rank_size)
{
	for (int i = 0; i < num_blocks; i++) {
		// exchange block by block
		exchange_block(sendbuf+i, recvbuf+i, i, rank, rank_size);
	}
}

void exchange_block(const particles_block_t *sendbuf, particles_block_t *recvbuf, int block_id, int rank, int rank_size)
{
	int src = MOD(rank-1, rank_size);
	int dst = MOD(rank+1, rank_size);
	int size = sizeof(particles_block_t);

	if (rank % 2){
		int bid = sendbuf->bid;
		// printf("   R%d(src=%d) send tag%d to dst=%d, block=%d\n", rank, src, block_id+10, dst, bid);
		MPI_Send(sendbuf, size, MPI_BYTE, dst, block_id+10, MPI_COMM_WORLD);
		MPI_Recv(recvbuf, size, MPI_BYTE, src, block_id+10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	} else {
		int bid = sendbuf->bid;
		MPI_Recv(recvbuf, size, MPI_BYTE, src, block_id+10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// printf("   R%d(src=%d) send tag%d to dst=%d, block=%d\n", rank, src, block_id+10, dst, bid);
		MPI_Send(sendbuf, size, MPI_BYTE, src, block_id+10, MPI_COMM_WORLD);
	}
}

// ---------------------------------------------------------
// ---------------------------------------------------------
void nbody_block_cham_solve(nbody_block_t *nbody, int num_blocks, int timesteps, float time_interval)
{
	int rank, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);
	assert(nbody != NULL);

	particles_block_t *local = nbody->local;
	particles_block_t *remote1 = nbody->remote1;
	particles_block_t *remote2 = nbody->remote2;
	forces_block_t *forces = nbody->forces;

	for (int t = 0; t < timesteps; t++) {
	// for (int t = 0; t < 1; t++) {

		printf("[nbody_block_solve] R%d enters the solver, timestep=%d...\n", rank, t);
		particles_block_t *sendbuf = local;
		particles_block_t *recvbuf = remote1;
		for (int r = 0; r < rank_size; r++){

			// each rank calculates its blocks
			// printf("[nbody_block_solve] R%d-iter%d: t%d | calculate blocks-at-%d with blocks-at-%d...\n", rank, t, r, local->bid, sendbuf->bid);
			calculate_block_forces_cham(forces, local, sendbuf, num_blocks);

			// call chameleon distributed taskwait to execute all the tasks
			// printf("[nbody_block_solve] R%d passed calculate_block_forces_cham...\n", rank);
    	int res = chameleon_distributed_taskwait(0);

			// synchronize the calculation before exchanging blocks
			#pragma omp single
			MPI_Barrier(MPI_COMM_WORLD);
			
			// exchange the blocks after computing
			if(r < rank_size - 1){
				printf("[exchange_block_particles] R%d call exchange_block_particles...\n", r);
				exchange_block_particles(sendbuf, recvbuf, num_blocks, rank, rank_size);
			}

			particles_block_t *aux = recvbuf;
			recvbuf = (r != 0) ? sendbuf : remote2;
			sendbuf = aux;
			// printf("   R%d | recvbuf: block%d| sendbuf: block%d\n", rank, recvbuf->bid, sendbuf->bid);
		}
		// TODO: think about porting updating blocks as tasks
		update_block_particles(local, forces, num_blocks, time_interval);
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

void calculate_block_forces_cham(forces_block_t *forces, particles_block_t *block1, particles_block_t *block2, int num_blocks)
{
	#pragma omp parallel for
	for (int i = 0; i < num_blocks; i++){
		for (int j = 0; j < num_blocks; j++){
			calculate_fblock_cham(forces+i, block1+i, block2+j);
		}
	}
}

void calculate_fblock_cham(forces_block_t *forces, particles_block_t *block1, particles_block_t *block2)
{
	// printf("[calculate_fblock] computes block%d...\n", block1->bid);
	// prepare the arguments of position/coord, px, py, pz
	int blk_size = BLOCK_SIZE;
	int blk_id = block1->bid;
	double * SPEC_RESTRICT px1 = block1->position_x;
	double * SPEC_RESTRICT py1 = block1->position_y;
	double * SPEC_RESTRICT pz1 = block1->position_z;
	double * SPEC_RESTRICT px2 = block2->position_x;
	double * SPEC_RESTRICT py2 = block2->position_y;
	double * SPEC_RESTRICT pz2 = block2->position_z;

	// prepare the arguments of mass, m1, m2
	double * SPEC_RESTRICT m1 = block1->mass;
	double * SPEC_RESTRICT m2 = block2->mass;

	// prepare the arguments of forces, fx, fy, fz
	double * SPEC_RESTRICT fx = forces->x;
	double * SPEC_RESTRICT fy = forces->y;
	double * SPEC_RESTRICT fz = forces->z;

	void* literal_size = *(void**)(&blk_size);
	void* literal_i = *(void**)(&blk_id);

	chameleon_map_data_entry_t* args = new chameleon_map_data_entry_t[13];
	args[0] = chameleon_map_data_entry_create(px1, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
	args[1] = chameleon_map_data_entry_create(py1, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
	args[2] = chameleon_map_data_entry_create(pz1, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
	args[3] = chameleon_map_data_entry_create(px2, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
	args[4] = chameleon_map_data_entry_create(py2, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
	args[5] = chameleon_map_data_entry_create(pz2, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);

	args[6] = chameleon_map_data_entry_create(m1, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);
	args[7] = chameleon_map_data_entry_create(m2, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_TO);

	args[8] = chameleon_map_data_entry_create(fx, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
	args[9] = chameleon_map_data_entry_create(fy, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);
	args[10] = chameleon_map_data_entry_create(fz, blk_size*sizeof(double), CHAM_OMP_TGT_MAPTYPE_FROM);

	args[11] = chameleon_map_data_entry_create(literal_size, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);
	args[12] = chameleon_map_data_entry_create(literal_i, sizeof(void*), CHAM_OMP_TGT_MAPTYPE_TO | CHAM_OMP_TGT_MAPTYPE_LITERAL);

	// create opaque task here
  cham_migratable_task_t *cur_task = chameleon_create_task((void *)&fblock_cal_kernel, 13, args);

	// get the id of the last task added
	// TYPE_TASK_ID last_t_id = chameleon_get_task_id(cur_task);

	// add tasks to the queue
	int32_t res = chameleon_add_task(cur_task);

	// clean up again
	delete[] args;
}

void fblock_cal_kernel(double * SPEC_RESTRICT px1, double * SPEC_RESTRICT py1, double * SPEC_RESTRICT pz1,
											double * SPEC_RESTRICT px2, double * SPEC_RESTRICT py2, double * SPEC_RESTRICT pz2,
											double * SPEC_RESTRICT m1, double * SPEC_RESTRICT m2,
											double * SPEC_RESTRICT fx, double * SPEC_RESTRICT fy, double * SPEC_RESTRICT fz,
											int blksize, int bid)
{
	// printf("[fblock_cal_kernel] computes block%d...\n", bid);
	// check the same blocks or not
	const int same_block = (px1 == px2);
	// main loop for calculating a block of forces
	for (int i = 0; i < blksize; i++){
		double fx_val = fx[i];
		double fy_val = fy[i];
		double fz_val = fz[i];
		for (int j = 0; j < blksize; j++){
			const double diff_x = px2[j] - px1[i];
			const double diff_y = py2[j] - py1[i];
			const double diff_z = pz2[j] - pz1[i];
			const double distance_squared = diff_x*diff_x + diff_y*diff_y + diff_z*diff_z;
			const double distance = sqrt(distance_squared);
			double force = 0.0f;
			if (!same_block || distance_squared != 0.0f){
				force = (m1[i] * m2[j] * gravitational_constant) / (distance_squared * distance);
			}
			fx_val += force * diff_x;
			fy_val += force * diff_y;
			fz_val += force * diff_z;
		}
		fx[i] = fx_val;
		fy[i] = fy_val;
		fz[i] = fz_val;
	}
}