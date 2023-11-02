#include "nbody.h"

// ---------------------------------------------------------
// For non-blocking solvers
// ---------------------------------------------------------
void nbody_setup_file(nbody_t *nbody, const nbody_conf_t *conf)
{
	int rank, rank_size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &rank_size);

	nbody_file_t file;
	file.size = nbody->particles._size;
	sprintf(file.name, "./data/%s-%s-%d-%d-%d-r%d", conf->name, TOSTRING(BIGO), conf->num_particles, conf->num_particles, conf->timesteps, rank);
	nbody->file = file;
}

void nbody_link_particles(particles_t *particles, int num_particles, void *addr)
{
	const size_t size = num_particles * sizeof(double);
	particles->position_x = (double *)addr + (0 * size);
	particles->position_y = (double *)addr + (1 * size);
	particles->position_z = (double *)addr + (2 * size);
	particles->velocity_x = (double *)addr + (3 * size);
	particles->velocity_y = (double *)addr + (4 * size);
	particles->velocity_z = (double *)addr + (5 * size);
	particles->mass = (double *)addr + (6 * size);
	particles->weight = (double *)addr + (7 * size);
	particles->_ptr = addr;
	particles->_size = size * PARTICLE_MEMBERS;
}

void nbody_generate_particles(const nbody_conf_t *conf, nbody_file_t *file)
{
	printf("[generate_particles] enter...\n");

	char fname[1024];
	struct stat st = {0};
	if (stat("data", &st) == -1)
	{
		mkdir("data", 0755);
	}
	sprintf(fname, "%s.in", file->name);
	if (!access(fname, F_OK))
	{
		return;
	}

	// open file for mapping
	const int fd = open(fname, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IRGRP | S_IROTH);
	assert(fd >= 0);
	// printf("[generate_particles] passed open file...\n");

	// check file size
	const int size = file->size;
	assert(size % PAGE_SIZE == 0);
	printf("[generate_particles] file_size = %d\n", size);

	// cause the regular file named by fd to be truncated to a size of precisely length bytes.
	int err = ftruncate(fd, size);
	assert(!err);

	void *addr = mmap(NULL, size, PROT_WRITE | PROT_READ, MAP_SHARED, fd, 0);
	assert(addr != MAP_FAILED);

    // if the mapped files exist
	particles_t particles;
	nbody_link_particles(&particles, conf->num_particles, addr);

	// initialize the particles
	nbody_particle_init(conf, &particles);
	
	// then, unmapping?
	err = munmap(addr, size);
	assert(!err);

	// and, close the file?
	err = close(fd);
	assert(!err);
}

void nbody_load_particles(nbody_t *nbody, const nbody_conf_t *conf, nbody_file_t *file)
{
	char fname[1024];
	sprintf(fname, "%s.in", file->name);
	// printf("[load_particles] file name = %s\n", file->name);

	// open the given input file
	const int fd = open(fname, O_RDONLY, 0);
	assert(fd >= 0);

	void *addr = mmap(NULL, file->size, PROT_READ|PROT_WRITE, MAP_PRIVATE, fd, 0);
	assert(addr != MAP_FAILED);

	// copy data from the mapped file to the nbody object
	// just copy at once, the data is copied to mem
	memcpy(nbody->particles._ptr, addr, nbody->particles._size);

	int err = munmap(addr, file->size);
	assert(!err);

	err = close(fd);
	assert(!err);
}

void nbody_setup_particles(nbody_t *nbody, const nbody_conf_t *conf)
{
  int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	// allocate memory using mmap for nbody
  // besides, to create a mapped file for initializing particle data
	void *addr = nbody_alloc(nbody->particles._size);
  printf("[nbody_setup_particles] addr = %p\n", addr);

	// link particles to the mapped file
  printf("[nbody_setup_particles] call nbody_link_particles...\n");
	nbody_link_particles(&nbody->particles, nbody->num_particles, addr);

	// generate particles
	// first, check the mapped file which is created or not
	// if yes, just need to load it
	// if not, we create, link particles, and initialize 
	printf("[nbody_setup_particles] call nbody_generate_particles...\n");
	nbody_generate_particles(conf, &nbody->file);

	// load particles in the given mapped file
  nbody_load_particles(nbody, conf, &nbody->file);
}

void nbody_setup_allocation(nbody_t *nbody, const nbody_conf_t *conf)
{
	const int num_particles = nbody->num_particles;
	const size_t size = num_particles * sizeof(double);
	
	particles_t *particles = &nbody->particles;
	particles->position_x = (double *)calloc(num_particles, size);
	particles->position_y = (double *)calloc(num_particles, size);
	particles->position_z = (double *)calloc(num_particles, size);
	particles->velocity_x = (double *)calloc(num_particles, size);
	particles->velocity_y = (double *)calloc(num_particles, size);
	particles->velocity_z = (double *)calloc(num_particles, size);
	particles->mass = (double *)calloc(num_particles, size);
	particles->weight = (double *)calloc(num_particles, size);
	particles->_ptr = nullptr;
	particles->_size = size * PARTICLE_MEMBERS;
}

void nbody_particle_init(const nbody_conf_t *conf, particles_t *part)
{
	srand(getpid());
	const int num_particles = conf->num_particles;
	// printf("[nbody_particle_init] R%d, seed=%d\n", rank, getpid());

	for (int i = 0; i < num_particles; i++){
		part->position_x[i] = conf->domain_size_x * ((double)rand() / ((double)RAND_MAX + 1.0));
		part->position_y[i] = conf->domain_size_y * ((double)rand() / ((double)RAND_MAX + 1.0));
		part->position_z[i] = conf->domain_size_z * ((double)rand() / ((double)RAND_MAX + 1.0));
		part->mass[i] = conf->mass_maximum * ((double)rand() / ((double)RAND_MAX + 1.0));
		part->weight[i] = gravitational_constant * part->mass[i];
	}
}

void nbody_setup_forces(nbody_t *nbody, const nbody_conf_t *conf)
{
	// void *base_addr = nbody_alloc(nbody->forces._size);
	// printf("[nbody_setup_forces] base_addr = %p\n", base_addr);

	const int num_particles = nbody->num_particles;
	const size_t size = num_particles * sizeof(double);

	// size_t array_size = conf->num_particles * sizeof(double);
	// nbody->forces._ptr = base_addr;
	// nbody->forces.x = (double *)base_addr + (0 * array_size);
	// nbody->forces.y = (double *)base_addr + (1 * array_size);
	// nbody->forces.z = (double *)base_addr + (2 * array_size);

	forces_t *f = &nbody->forces;
	f->x = (double *)calloc(num_particles, size);
	f->y = (double *)calloc(num_particles, size);
	f->z = (double *)calloc(num_particles, size);
	f->_ptr = nullptr;
	f->_size = size * 3; // 3d-force (x, y, z)
}

nbody_t nbody_setup(const nbody_conf_t *conf)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	nbody_t nbody;
	nbody.num_particles = conf->num_particles;
	nbody.timesteps = conf->timesteps;

	// Compute the size of structures
	nbody.particles._size = nbody.num_particles * PARTICLE_MEMBERS * sizeof(double);
	nbody.forces._size = nbody.num_particles * FORCE_MEMBERS * sizeof(double);

	// printf("\t[DEBUG] nbody_setup: nbody.particles._size=%d\n", nbody.particles._size);
	// printf("\t[DEBUG] nbody_setup: nbody.forces._size=%d\n", nbody.forces._size);

	// Set up mapped files for allocating memory
	// nbody_setup_file(&nbody, conf);
	
	// Set up Nbody by calloc
	nbody_setup_allocation(&nbody, conf);

	// Initialize the particles using mmap
	// nbody_setup_particles(&nbody, conf);
	nbody_particle_init(conf, &nbody.particles);

	// Initialize the forces
  nbody_setup_forces(&nbody, conf);

	return nbody;
}

void nbody_free(nbody_t *nbody)
{
	// int err = munmap(nbody->particles._ptr, nbody->particles._size);
	// err |= munmap(nbody->forces._ptr, nbody->forces._size);
	// assert(!err);

	const int num_particles = nbody->num_particles;

	// free all particles
	particles_t *p = &nbody->particles;
	free(p->position_x);
	free(p->position_y);
	free(p->position_z);
	free(p->velocity_x);
	free(p->velocity_y);
	free(p->velocity_z);
	free(p->mass);
	free(p->weight);

	// free all forces
	forces_t *f = &nbody->forces;
	free(f->x);
	free(f->y);
	free(f->z);
}

// ---------------------------------------------------------
// For blocking solvers
// ---------------------------------------------------------
void nbody_setup_block_allocation(nbody_block_t *nbody, const nbody_conf_t *conf)
{
	const int num_blocks = nbody->num_blocks;
	const size_t size_particle_block = num_blocks * sizeof(particles_block_t);
	const size_t size_force_block = num_blocks * sizeof(forces_block_t);
	nbody->local = (particles_block_t *)calloc(num_blocks, size_particle_block);
	nbody->remote1 = (particles_block_t *)calloc(num_blocks, size_particle_block);
	nbody->remote2 = (particles_block_t *)calloc(num_blocks, size_particle_block);
	nbody->forces = (forces_block_t *)calloc(num_blocks, size_force_block);
}

void nbody_block_particle_init(const nbody_conf_t *conf, particles_block_t *part)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	srand(getpid());
	const int num_blocks = conf->num_blocks;
	for (int b = 0; b < num_blocks; b++){
		part[b].bid = rank*num_blocks + b;
		for (int i = 0; i < BLOCK_SIZE; i++){
			part[b].position_x[i] = conf->domain_size_x * ((double)rand() / ((double)RAND_MAX + 1.0));
			part[b].position_y[i] = conf->domain_size_y * ((double)rand() / ((double)RAND_MAX + 1.0));
			part[b].position_z[i] = conf->domain_size_z * ((double)rand() / ((double)RAND_MAX + 1.0));
			part[b].mass[i] = conf->mass_maximum * ((double)rand() / ((double)RAND_MAX + 1.0));
			part[b].weight[i] = gravitational_constant * part[b].mass[i];
		}
	}
}

nbody_block_t nbody_block_setup(const nbody_conf_t *conf)
{
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	nbody_block_t nbody;
	nbody.num_blocks = conf->num_blocks;
	nbody.timesteps = conf->timesteps;

	// Set up mapped files for allocating memory
	nbody_setup_block_allocation(&nbody, conf);

	// Initialize the particles using mmap
	// nbody_particle_init(conf, &nbody.particles);
	nbody_block_particle_init(conf, nbody.local);

	return nbody;
}

void nbody_block_free(nbody_block_t *nbody)
{
	free(nbody->local);
	free(nbody->remote1);
	free(nbody->remote2);
	free(nbody->forces);
}
