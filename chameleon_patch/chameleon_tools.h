#ifndef _CHAMELEON_TOOLS_H_
#define _CHAMELEON_TOOLS_H_

#include <stddef.h>
#include <stdint.h>
#include <dlfcn.h>
#include <vector>

//#include <list>
//#include <mutex>
//#include <atomic>

#pragma region Enums and Definitions
/*****************************************************************************
 * Enums
 ****************************************************************************/
typedef enum cham_t_callback_types_t {
    cham_t_callback_thread_init                 = 1,
    cham_t_callback_thread_finalize             = 2,
    cham_t_callback_task_create                 = 3,
    cham_t_callback_encode_task_tool_data       = 4,
    cham_t_callback_decode_task_tool_data       = 5,
    cham_t_callback_task_schedule               = 6,
    cham_t_callback_sync_region                 = 7,
    cham_t_callback_determine_local_load        = 8,
    cham_t_callback_select_num_tasks_to_offload = 9,
    cham_t_callback_select_tasks_for_migration  = 10,
    cham_t_callback_select_num_tasks_to_replicate = 11,
    cham_t_callback_change_freq_for_execution   = 12,
    cham_t_callback_get_load_stats_per_taskwait = 13,
    cham_t_callback_get_task_wallclock_time     = 14,
    cham_t_callback_train_prediction_model      = 15,
    cham_t_callback_load_prediction_model       = 16,
    cham_t_callback_get_numtasks_per_rank       = 17
} cham_t_callback_types_t;

typedef enum cham_t_set_result_t {
    cham_t_set_error            = 0,
    cham_t_set_never            = 1,
    cham_t_set_impossible       = 2,
    cham_t_set_sometimes        = 3,
    cham_t_set_sometimes_paired = 4,
    cham_t_set_always           = 5
} cham_t_set_result_t;

typedef enum cham_t_task_schedule_type_t {
    cham_t_task_start           = 1,
    cham_t_task_yield           = 2,
    cham_t_task_end             = 3,
    cham_t_task_cancel          = 4
} cham_t_task_schedule_type_t;

static const char* cham_t_task_schedule_type_t_values[] = {
    NULL,
    "cham_t_task_start",        // 1
    "cham_t_task_yield",        // 2
    "cham_t_task_end",          // 3
    "cham_t_task_cancel"        // 4
};

typedef enum cham_t_task_flag_t {
    cham_t_task_local           = 0x00000001,
    cham_t_task_remote          = 0x00000002,
    cham_t_task_replicated      = 0x00000004
} cham_t_task_flag_t;

static void cham_t_task_flag_t_value(int type, char *buffer) {
  char *progress = buffer;
  if (type & cham_t_task_local)
    progress += sprintf(progress, "cham_t_task_local");
  if (type & cham_t_task_remote)
    progress += sprintf(progress, "cham_t_task_remote");
  if (type & cham_t_task_replicated)
    progress += sprintf(progress, "cham_t_task_replicated");
}

typedef enum cham_t_sync_region_type_t {
    cham_t_sync_region_taskwait  = 1
} cham_t_sync_region_type_t;

static const char* cham_t_sync_region_type_t_values[] = {
    NULL,
    "cham_t_sync_region_taskwait"       // 1
};

typedef enum cham_t_sync_region_status_t {
    cham_t_sync_region_start    = 1,
    cham_t_sync_region_end      = 2
} cham_t_sync_region_status_t;

static const char* cham_t_sync_region_status_t_values[] = {
    NULL,
    "cham_t_sync_region_start",         // 1
    "cham_t_sync_region_end"            // 2
};

/*****************************************************************************
 * General definitions
 ****************************************************************************/
typedef struct cham_t_rank_info_t {
    int32_t comm_rank;
    int32_t comm_size;
} cham_t_rank_info_t;

typedef struct cham_t_task_param_info_t {
    int32_t num_args;
    int64_t *arg_sizes;
    int64_t *arg_types;
    void **arg_pointers;
} cham_t_task_param_info_t;

typedef void (*cham_t_interface_fn_t) (void);

typedef cham_t_interface_fn_t (*cham_t_function_lookup_t) (
    const char *interface_function_name
);

typedef void (*cham_t_callback_t) (void);

// either interprete data type as value or pointer
typedef union cham_t_data_t {
    uint64_t value;
    void *ptr;
} cham_t_data_t;

typedef struct cham_t_migration_tupel_t {
    TYPE_TASK_ID task_id;
    int32_t rank_id;
} cham_t_migration_tupel_t;

static cham_t_migration_tupel_t cham_t_migration_tupel_create(TYPE_TASK_ID task_id, int32_t rank_id) {
    cham_t_migration_tupel_t val;
    val.task_id = task_id;
    val.rank_id = rank_id;
    return val;
}

typedef struct cham_t_replication_info_t {
	int num_tasks, num_replication_ranks;
	int *replication_ranks;
} cham_t_replication_info_t;

static cham_t_replication_info_t cham_t_replication_info_create(int num_tasks, int num_replication_ranks, int *replication_ranks) {
	cham_t_replication_info_t info;
	info.num_tasks = num_tasks;
	info.num_replication_ranks = num_replication_ranks;
	info.replication_ranks = replication_ranks;
	return info;
}

static void free_replication_info(cham_t_replication_info_t *info) {
	free(info->replication_ranks);
	info = NULL;
}

/*****************************************************************************
 * Init / Finalize / Start Tool
 ****************************************************************************/
typedef void (*cham_t_finalize_t) (
    cham_t_data_t *tool_data
);

typedef int (*cham_t_initialize_t) (
    cham_t_function_lookup_t lookup,
    cham_t_data_t *tool_data
);

typedef struct cham_t_start_tool_result_t {
    cham_t_initialize_t initialize;
    cham_t_finalize_t finalize;
    cham_t_data_t tool_data;
} cham_t_start_tool_result_t;


/*****************************************************************************
 * Getter / Setter
 ****************************************************************************/
typedef cham_t_set_result_t (*cham_t_set_callback_t) (
    cham_t_callback_types_t event,
    cham_t_callback_t callback
);

typedef int (*cham_t_get_callback_t) (
    cham_t_callback_types_t event,
    cham_t_callback_t *callback
);

typedef cham_t_data_t *(*cham_t_get_thread_data_t) (void);
typedef cham_t_data_t *(*cham_t_get_rank_data_t) (void);
typedef cham_t_data_t *(*cham_t_get_task_data_t) (TYPE_TASK_ID);
typedef cham_t_task_param_info_t (*cham_t_get_task_param_info_by_id_t) (TYPE_TASK_ID);
typedef cham_t_task_param_info_t (*cham_t_get_task_param_info_t) (cham_migratable_task_t*);
typedef cham_t_rank_info_t *(*cham_t_get_rank_info_t) (void);

/*****************************************************************************
 * List of callbacks
 ****************************************************************************/
// 1. callback get status of init_thread
typedef void (*cham_t_callback_thread_init_t) (
    cham_t_data_t *thread_data
);

// 2. callback get status of finalize_thread
typedef void (*cham_t_callback_thread_finalize_t) (
    cham_t_data_t *thread_data
);

// 3. callback queuing tasks
typedef void (*cham_t_callback_task_create_t) (
    cham_migratable_task_t * task,                   // opaque data type for internal task
    std::vector<int64_t> arg_sizes,
    double queued_time,
    intptr_t codeptr_ra,
    int taskwait_counter
);

typedef void (*cham_t_callback_task_schedule_t) (
    cham_migratable_task_t * task,                   // opaque data type for internal task
    cham_t_task_flag_t task_flag,
    cham_t_data_t *task_data,
    cham_t_task_schedule_type_t schedule_type,
    cham_migratable_task_t * prior_task,             // opaque data type for internal task
    cham_t_task_flag_t prior_task_flag,
    cham_t_data_t *prior_task_data
);

// Encode custom task tool data (if any has been set) for a task that will be migrated to remote rank.
// Ensures that this data is also send to remote rank and available in tool calls.
// Note: Only necessary when task specific data is required
// Note: Only works in combination with cham_t_callback_decode_task_tool_data_t
typedef void *(*cham_t_callback_encode_task_tool_data_t) (
    cham_migratable_task_t * task,                   // opaque data type for internal task
    cham_t_data_t *task_data,
    int32_t *size
);

// Decode custom task tool data (if any has been set) for a task that has been migrated to remote rank.
// Restore data in corresponding task_data struct
// Note: Only necessary when task specific data is required
// Note: Only works in combination with cham_t_callback_encode_task_tool_data_t
typedef void (*cham_t_callback_decode_task_tool_data_t) (
    cham_migratable_task_t * task,                   // opaque data type for internal task
    cham_t_data_t *task_data,
    void *buffer,
    int32_t size
);

typedef void (*cham_t_callback_sync_region_t) (
    cham_t_sync_region_type_t sync_region_type,
    cham_t_sync_region_status_t sync_region_status,
    cham_t_data_t *thread_data,
    const void *codeptr_ra
);

typedef int32_t (*cham_t_callback_determine_local_load_t) (
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    TYPE_TASK_ID* task_ids_local_rep,
    int32_t num_tasks_local_rep,
    TYPE_TASK_ID* task_ids_stolen,
    int32_t num_tasks_stolen,
    TYPE_TASK_ID* task_ids_stolen_rep,
    int32_t num_tasks_stolen_rep
);

// information about current rank and number of ranks can be achived with cham_t_get_rank_info_t
typedef void (*cham_t_callback_select_num_tasks_to_offload_t) (
    int32_t* num_tasks_to_offload_per_rank,
    const int32_t* load_info_per_rank,
    int32_t num_tasks_local,
    int32_t num_tasks_stolen
);

typedef cham_t_replication_info_t* (*cham_t_callback_select_num_tasks_to_replicate_t) (
    const int32_t* load_info_per_rank,
    int32_t num_tasks_local,
    int32_t *num_replication_infos
);

// information about current rank and number of ranks can be achived with cham_t_get_rank_info_t
// task annotations can be queried with TODO
typedef cham_t_migration_tupel_t* (*cham_t_callback_select_tasks_for_migration_t) (
    const int32_t* load_info_per_rank,
    TYPE_TASK_ID* task_ids_local,
    int32_t num_tasks_local,
    int32_t num_tasks_stolen,
    int32_t* num_tuples
);

typedef int32_t (*cham_t_callback_change_freq_for_execution_t)(
    cham_migratable_task_t * task,
    int32_t load_info_per_rank,
    int32_t total_created_tasks_per_rank
);

typedef void (*cham_t_callback_get_load_stats_per_taskwait_t)(
    int32_t taskwait_counter,
    int32_t thread_id,
    double taskwait_load
);

typedef void (*cham_t_callback_get_task_wallclock_time_t)(
    int32_t taskwait_counter,
    int32_t thread_id,
    int task_id,
    double wallclock_time
);

typedef bool (*cham_t_callback_train_prediction_model_t)(
    int32_t taskwait_counter,
    int prediction_mode
);

typedef std::vector<double> (*cham_t_callback_load_prediction_model_t)(
    int32_t taskwait_counter,
    int prediction_mode
);

typedef int (*cham_t_callback_get_numtasks_per_rank_t)(
    int32_t taskwait_counter
);

#pragma endregion

#endif
