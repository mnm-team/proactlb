// Communication thread header
#ifndef _COMMTHREAD_H_
#define _COMMTHREAD_H_

#include <numeric>
#include <unordered_set>
#include <mpi.h>

#include "chameleon.h"
#include "chameleon_common.h"
#include "chameleon_tools.h"
#include "request_manager.h"

// communicator for remote task requests
extern MPI_Comm chameleon_comm;
// communicator for sending back mapped values
extern MPI_Comm chameleon_comm_mapped;
// communicator for load information
extern MPI_Comm chameleon_comm_load;
// communicator for task cancellation
extern MPI_Comm chameleon_comm_cancel;
// communicator for activating replicated tasks
extern MPI_Comm chameleon_comm_activate;

extern RequestManager request_manager_send;
extern RequestManager request_manager_receive;
extern RequestManager request_manager_cancel;

extern int chameleon_comm_rank;
extern int chameleon_comm_size;

// Communication tracking
extern std::atomic<int32_t> _num_active_communications_overall;
extern std::vector<intptr_t> _image_base_addresses;

// list with local task entries
// these can either be executed locally or offloaded/replicated to a different rank
extern thread_safe_task_list_t _local_tasks;
extern std::atomic<int32_t> _num_local_tasks_outstanding;

// list with stolen task entries that should be executed
extern thread_safe_task_list_t _stolen_remote_tasks;
extern std::atomic<int32_t> _num_remote_tasks_outstanding;

// list of replicated (i.e. offloaded) tasks
// they can be executed either on the remote rank or on the local rank
extern thread_safe_task_list_t _replicated_local_tasks;
extern std::atomic<int32_t> _num_replicated_local_tasks_outstanding_send;
extern std::atomic<int32_t> _num_replicated_local_tasks_outstanding_compute;
extern std::vector<int> _num_replicated_local_tasks_per_victim;

extern thread_safe_task_list_t _replicated_remote_tasks;
extern thread_safe_task_list_t _replicated_migrated_tasks;
extern std::atomic<int32_t> _num_replicated_remote_tasks_outstanding;
extern std::atomic<int32_t> _num_replicated_and_migrated_remote_tasks_outstanding;

// list with stolen task entries that need output data transfer
extern thread_safe_task_list_t _remote_tasks_send_back;
// list with replicated task entries that need initial transfer
extern thread_safe_list_t<cham_t_replication_info_t*> _replication_infos_list;

// map that maps tag ids back to local tasks that have been offloaded and expect result data
extern thread_safe_task_map_t _map_offloaded_tasks_with_outputs;
// map that maps tag ids back to stolen tasks
extern thread_safe_task_map_t _map_tag_to_remote_task;
// mapping of all active task ids and task
extern thread_safe_task_map_t _map_overall_tasks;

extern std::unordered_set<TYPE_TASK_ID> _cancelled_task_ids;

// ====== Info about outstanding jobs (local & stolen & offloaded (communication)) ======
extern std::vector<int32_t> _outstanding_jobs_ranks;
extern std::atomic<int32_t> _outstanding_jobs_local;
extern std::atomic<int32_t> _outstanding_jobs_sum;

// ====== Info about real load that is open or is being processed ======
extern std::vector<int32_t> _load_info_ranks;
extern std::vector<double>  _list_real_load;
extern std::vector<double> _total_load_info_ranks; // load after an iter is finished

// ====== Info about the number of taskwait count being processed ======
extern std::atomic<int> _commthread_time_taskwait_count;
extern int _global_flag_prev_taskwait_idx;
extern bool _flag_is_new_iteration;

// list that holds task ids (created at the current rank) that are not finsihed yet
extern thread_safe_list_t<TYPE_TASK_ID> _unfinished_locally_created_tasks;

// Threading section
extern std::atomic<int> _comm_thread_load_exchange_happend;

#if CHAMELEON_TOOL_SUPPORT && CHAM_PREDICTION_MODE > 0
// ====== Info about predicted load by the tool that is being processed ======
extern std::vector<double> _list_predicted_load;    // array of all pred-loads per iteration
extern std::vector<double> _predicted_load_info_ranks; // just keep the pred-load of the current iteration

// to check the number of created tasks at the setup-time
extern int _total_created_tasks_per_rank;

// a defined flag for checking the cham-tool prediction model is ready or not
extern std::atomic<bool> _flag_model_is_trained;

// setting for chameleon tool, default could be follows, but they are updated in
// the chameleon_init function by getting the environment vars
extern int MAX_EST_NUM_ITERS;
extern int TIME_TO_TRAIN_MODEL;

#if CHAM_PROACT_MIGRATION > 0
// ====== Flag to allow the predict-load exchange that could happen ======
extern std::atomic<int> _comm_thread_predload_exchange_happend;

// ====== Flags to mark the actions, i.e., create_gather pred_load, handle_gather pred_load
//        should do 1 once time when a new iter/cycle starts,
extern int _flag_create_gather_predload_happened;
extern int _flag_handle_gather_predload_happened;

#if CHAM_PROACT_MIGRATION==1 
extern int _flag_predict_iter_by_iter;
#elif CHAM_PROACT_MIGRATION==2
extern int _flag_enable_proactmig_by_the_predicted_future;
#endif

#endif /* CHAM_PROACT_MIGRATION > 0 */

#if CHAM_PREDICTION_MODE==2
extern int _flag_predict_for_the_whole_future;
#endif

#endif

// variables to indicate when it is save to break out of taskwait
extern std::mutex _mtx_taskwait;
extern std::atomic<int> _num_threads_involved_in_taskwait;
extern std::atomic<int> _num_threads_active_in_taskwait;
extern std::atomic<int32_t> _num_threads_idle;
extern std::atomic<int> _num_ranks_not_completely_idle;

// number of active migrations per target rank
// desired: should block new migration to target as long as there are still active migrations ongoing
extern std::atomic<int> _active_migrations_per_target_rank[];

extern std::atomic<bool> _trace_events_initialized;
extern int event_receive_tasks;
extern int event_recv_back;
extern int event_create_gather_request;
extern int event_exchange_outstanding;
extern int event_react_mig;
extern int event_workstealing_decision;
extern int event_send_back;
extern int event_progress_send;
extern int event_progress_recv;
extern int event_offload_action;
extern int event_encode_buffer;
extern int event_decode_buffer;
extern int event_prepare_taskarray_off1;
extern int event_proact_mig_off1;
extern int event_prepare_taskarray_off2;
extern int event_proact_mig_off2;

// lock used to ensure that currently only a single thread is doing communication progression
extern std::mutex _mtx_comm_progression;


//================================================================
// Class for commthread session data type
//================================================================
class chameleon_comm_thread_session_data_t {
    public:
    // =============== General Vars
    std::atomic<int> flag_thread_sleeping_set;
    std::atomic<int> num_threads_in_tw;

    // std::atomic<bool> has_not_replicated(true);
    std::atomic<double> time_last_load_exchange;
    std::atomic<double> time_gather_posted;
    std::atomic<bool> has_replicated;
    std::atomic<bool> is_migration_victim;

#if CHAM_STATS_RECORD
    std::atomic<double> time_start_comm;
#endif

    // =============== Send Thread Vars
    //int request_gather_created      = 0;
    std::atomic<int> request_gather_created;
    MPI_Request request_gather_out;
    MPI_Status  status_gather_out;
    std::atomic<int> offload_triggered;
    std::atomic<int> last_known_sum_outstanding;
    
    // =============== Monitoring load info
    std::vector<int32_t> tasks_to_offload;
    int transported_load_values[3];
    int *buffer_load_values;

#if CHAMELEON_TOOL_SUPPORT && CHAM_PREDICTION_MODE > 0 && CHAM_PROACT_MIGRATION > 0
    // for storing predicted values over iters
    double *buffer_predicted_load_values;

    // for proactive task migration algorithm
    std::vector<int32_t> proact_offload_tasks_table;

    // =============== Monitoring predicted load info
    // could not need a flag for prediction, temporarily
    // get it in the loop with request_gather_created in general
    MPI_Request request_gather_prediction_out;
    MPI_Status  status_gather_prediction_out;

    // for loop-to-loop migrating tasks at the communication thread side
    bool flag_allows_l2l_task_migration;

    // for separately migrating tasks by round-robin strategy
    std::vector<int32_t> victim_idx_order_to_proactmig_arr;
    // for migrating a encoded package of tasks
    std::vector<int32_t> victims_to_proactmig_arr;

#endif

    // =============== Num migrated tasks at once
    int n_task_send_at_once = 1;

    // =============== Recv Thread Vars
    std::vector<int> tracked_last_req_recv;

    // =============== Constructor
    chameleon_comm_thread_session_data_t() {
        // do nothing
    }
};

extern chameleon_comm_thread_session_data_t _session_data;


//================================================================
// Util Functions for CommThread
//================================================================

#ifdef __cplusplus
extern "C" {
#endif

void chameleon_comm_thread_session_data_t_init();

void print_arg_info(std::string prefix, cham_migratable_task_t *task, int idx);

void print_arg_info_w_tgt(std::string prefix, cham_migratable_task_t *task, int idx);

void cancel_offloaded_task(cham_migratable_task_t *task);

void cancel_offloaded_task_on_rank(cham_migratable_task_t *task, int rank);

void* comm_thread_action(void *arg);

int32_t start_communication_threads();

int32_t stop_communication_threads();

int32_t put_comm_threads_to_sleep();

void action_communication_progression(int comm_thread);

void cleanup_work_phase();

int exit_condition_met(int from_taskwait, int print);

#ifdef __cplusplus
}
#endif
#endif
