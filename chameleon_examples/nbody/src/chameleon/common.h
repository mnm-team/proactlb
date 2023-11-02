#ifndef COMMON_H
#define COMMON_H

#include <cstddef>

// BIGO definition
#ifndef BIGO
#define BIGO N2
#define _BIGO_N2
#endif

#define PART 1024
#define PAGE_SIZE 4096

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

#define TOLERATED_ERROR 0.0008

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define LOG2(a) (31-__builtin_clz((a)))
#define MOD(a, b)  ((a) < 0 ? ((((a) % (b)) + (b)) % (b)) : ((a) % (b)))
#define ROUNDUP(x, y) ({             \
    const typeof(y) __y = y;         \
    (((x) + (__y - 1)) / __y) * __y; \
})

#define STRINGIFY(s) #s
#define TOSTRING(s) STRINGIFY(s)
#define CALCULATE_FORCES(s) calculate_forces_##s
#define CALCULATE_BLOCK_FORCES(s) calculate_block_forces_##s
#define XCALCULATE_FORCES(s) CALCULATE_FORCES(s)
#define XCALCULATE_BLOCK_FORCES(s) CALCULATE_BLOCK_FORCES(s)
#define calculate_forces XCALCULATE_FORCES(BIGO)
#define calculate_block_forces XCALCULATE_BLOCK_FORCES(BIGO)

static const double gravitational_constant = 6.6726e-11f; /* N(m/kg)2 */
static const double default_domain_size_x  = 1.0e+10; /* m, 10 bilion  */
static const double default_domain_size_y  = 1.0e+10; /* m, 10 bilion  */
static const double default_domain_size_z  = 1.0e+10; /* m, 10 bilion */
static const double default_mass_maximum   = 1.0e+28; /* kg */
static const double default_time_interval  = 1.0e+0;  /* s  */
static const int   default_seed           = 12345;
static const char* default_name           = "./data";
static const int   default_num_particles  = 0;
static const int   default_timesteps      = 0;
static const int   default_save_result    = 0;
static const int   default_check_result   = 0;

typedef struct {
	double domain_size_x;
	double domain_size_y;
	double domain_size_z;
	double mass_maximum;
	double time_interval;
	int seed;
	char name[1024];
	int num_particles;
	int num_blocks;
	int timesteps;
	int save_result;
	int check_result;
} nbody_conf_t;

void nbody_get_conf(int argc, char **argv, nbody_conf_t *conf);
void nbody_print_usage(int argc, char **argv);
void *nbody_alloc(size_t size);

#endif // COMMON_H
