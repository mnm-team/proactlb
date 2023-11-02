#include "common.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cstring>
#include <chrono>
#include <ctime>

#include <sys/mman.h>
#include <getopt.h>
#include <ieee754.h>

/* static if we do not want to export this function outside
 * just can be used in this file, e.g.,
 * static void nbody_print_usage(int argc, char **argv)
 */
void nbody_print_usage(int argc, char **argv)
{
	fprintf(stderr, "Usage: %s <-p particles> <-t timesteps> [OPTION]...\n", argv[0]);
	fprintf(stderr, "Parameters:\n");
	fprintf(stderr, "  -p, --particles=PARTICLES\t\tuse PARTICLES as the total number of particles (default: 16384)\n");
	fprintf(stderr, "  -t, --timesteps=TIMESTEPS\t\tuse TIMESTEPS as the number of timesteps (default: 10)\n\n");
	fprintf(stderr, "Optional parameters:\n");
	fprintf(stderr, "  -c, --check\t\t\t\tcheck the correctness of the result (disabled by default)\n");
	fprintf(stderr, "  -C, --no-check\t\t\tdo not check the correctness of the result\n");
	fprintf(stderr, "  -o, --output\t\t\t\tsave the computed particles to the default output file (disabled by default)\n");
	fprintf(stderr, "  -O, --no-output\t\t\tdo not save the computed particles to the default output file\n");
	fprintf(stderr, "  -h, --help\t\t\t\tdisplay this help and exit\n\n");
}

void nbody_get_conf(int argc, char **argv, nbody_conf_t *conf)
{
	assert(conf != NULL);
	conf->domain_size_x = default_domain_size_x;
	conf->domain_size_y = default_domain_size_y;
	conf->domain_size_z = default_domain_size_z;
	conf->mass_maximum  = default_mass_maximum;
	conf->time_interval = default_time_interval;
	conf->seed          = default_seed;
	conf->num_particles = default_num_particles;
	conf->num_blocks    = conf->num_particles / BLOCK_SIZE;
	conf->timesteps     = default_timesteps;
	conf->save_result   = default_save_result;
	conf->check_result  = default_check_result;
	strcpy(conf->name, default_name);
	
	static struct option long_options[] = {
		{"particles",	required_argument,	0, 'p'},
		{"timesteps",	required_argument,	0, 't'},
		{"check",		no_argument,		0, 'c'},
		{"no-check",	no_argument,		0, 'C'},
		{"output",		no_argument,		0, 'o'},
		{"no-output",	no_argument,		0, 'O'},
		{"help",		no_argument,		0, 'h'},
		{0, 0, 0, 0}
	};
	
	int c;
	int index;
	while ((c = getopt_long(argc, argv, "hoOcCp:t:", long_options, &index)) != -1) {
		switch (c) {
			case 'h':
				nbody_print_usage(argc, argv);
				exit(0);
			case 'o':
				conf->save_result = 1;
				break;
			case 'O':
				conf->save_result = 0;
				break;
			case 'c':
				conf->check_result = 1;
				break;
			case 'C':
				conf->check_result = 0;
				break;
			case 'p':
				conf->num_particles = atoi(optarg);
				break;
			case 't':
				conf->timesteps = atoi(optarg);
				break;
			case '?':
				exit(1);
			default:
				abort();
		}
	}
	
	if (!conf->num_particles || !conf->timesteps) {
		nbody_print_usage(argc, argv);
		exit(1);
	}
}

void *nbody_alloc(size_t size){

	void *addr = mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
	assert(addr != MAP_FAILED);
	return addr;

}