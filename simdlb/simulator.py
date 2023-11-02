"""An engine of the simulator for dynamic load balancing in task-parallel applications.

The program will try to simulate the execution of task parallel application.
In case of imbalance at runtime with the constrain of communication overhead, this 
helps to estimate the bounds of performane efficiency.

  Typical usage example (temporaily):
  $ python simulator.py <config_input.txt>
"""

from queue import Queue
from time import sleep, perf_counter
from threading import Thread

from task import *
from balancer import *
from migrator import *
from profiler import *

import re
import os
import sys
import copy
import numpy as np
import pandas as pd

# -----------------------------------------------------
# Constant Definition
# -----------------------------------------------------
CLOCK_RATE = 1E3 # count as miliseconds

# -----------------------------------------------------
# Init simulation
# -----------------------------------------------------
def init_simulation(cf):

    # extract setup information
    clock_rate = cf.iloc[1,0]
    num_processes = int(cf.iloc[6,0])
    num_slowdown_ranks = int(cf.iloc[7,0])
    num_tasks_per_rank = int(cf.iloc[8,0])
    slowdown = cf.iloc[9,0]
    
    # local task queues on each rank
    local_queues = []
    slowdown_ranks = []
    slowdown_scales = []

    # select the slowdown processes
    for p in range(num_slowdown_ranks):
        slowdown_ranks.append(p)
        slowdown_scales.append(slowdown)

    # init the local task queues
    for r in range(num_processes):
        arr_local_tasks = []
        for t in range(num_tasks_per_rank):
            tid = (r << 16) + t
            dur = 1.0
            if r in slowdown_ranks:
                dur = dur/slowdown_scales[r]
            dur = dur * clock_rate
            data = 1.0 # MB
            node = r
            task = Task(tid, dur, data, node)
            arr_local_tasks.append(task)

        # add the local queue of tasks per rank
        local_queues.append(arr_local_tasks)

    return [local_queues, slowdown_ranks, slowdown_scales, num_processes]

# -----------------------------------------------------
# Update load array
# -----------------------------------------------------
def update_total_load(rank, local_load_arr, remot_load_arr, cur_task):
    orig_node = cur_task.local_node
    if orig_node == rank:
        local_load_arr[rank] += 1
    else:
        remot_load_arr[rank] += 1

# -----------------------------------------------------
# Simulation engine
# -----------------------------------------------------
def simulate(local_queues, slowdown_procs, slowdown_scales, iter, clock_rate, arr_feedback,
                cost_balancing, cost_migration_delay, noise, arr_feedback_priorities):

    # make a copy of local_queues to keep it for next iterations
    copy_local_queues = copy.deepcopy(local_queues)
    
    # for clocking and tracking task execution
    clock = 0
    remote_queues = []
    arr_local_load = []
    arr_remot_load = []

    # arrays for work stealing
    arr_steal_request_msg = []
    arr_steal_accept_msg  = []

    # arrays for react task offloading
    arr_being_exe_tasks = []
    arr_victim_offloader = []
    arr_num_tasks_before_execution = []
    arr_tmp_buffer_migrated_tasks  = []

    # arrays for profiling tasks executed
    arr_profiled_tasks = []
    arr_queue_status = []

    # check and init the queues for the first time
    num_procs = len(copy_local_queues)
    num_tasks_being_executed = 0
    sum_queues_length = 0
    for i in range(num_procs):
        arr_num_tasks_before_execution.append(len(copy_local_queues[i]))
        sum_queues_length += len(copy_local_queues[i]) # get num total tasks
        remote_queues.append([])
        arr_local_load.append(0.0)
        arr_remot_load.append(0.0)
        
        # for work stealing
        arr_steal_request_msg.append([])
        arr_steal_accept_msg.append([])

        # for react task offloading
        arr_being_exe_tasks.append(None)
        arr_victim_offloader.append([])
        arr_tmp_buffer_migrated_tasks.append([])
        
        # profiled tasks arr for each process
        arr_profiled_tasks.append([])
        arr_queue_status.append([len(copy_local_queues[i])])

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    while(sum_queues_length != 0 or num_tasks_being_executed != 0):
        # --------------------------------------------------------
        # increase clock in milisecond
        # --------------------------------------------------------
        clock += 1

        # --------------------------------------------------------
        # balancing strategies
        # --------------------------------------------------------
        # 1. work-stealing
        # queue_status_res = check_queue_status(clock, num_procs, copy_local_queues)
        
        # exchange_steal_request(clock, queue_status_res, arr_steal_request_msg, arr_steal_accept_msg, cost_balancing)

        # arr_tasks2steal = recv_steal_accept(clock, queue_status_res, arr_steal_request_msg, arr_steal_accept_msg, cost_balancing)

        # steal_tasks(clock, copy_local_queues, arr_tasks2steal, arr_tmp_buffer_migrated_tasks, arr_steal_request_msg, arr_steal_accept_msg,
        #             iter, cost_migration_delay, slowdown_procs, slowdown_scales, clock_rate, noise)

        # 2. reactive task offloading
        Rimb = check_imbalance_status(clock, copy_local_queues, arr_being_exe_tasks)
        
        react_task_offloading(clock, num_procs, Rimb, copy_local_queues,
                                  arr_num_tasks_before_execution, arr_victim_offloader)

        select_tasks_to_offload(clock, copy_local_queues, arr_victim_offloader,
                                iter, cost_balancing, cost_migration_delay, noise)
        
        offload_tasks(clock, copy_local_queues, remote_queues, arr_tmp_buffer_migrated_tasks,
                        slowdown_procs, slowdown_scales, iter, clock_rate, noise)

        # 3. fb task offloading
        # Rimb = check_imbalance_status(clock, copy_local_queues, arr_being_exe_tasks)

        # if iter == 0:
        #     react_task_offloading(clock, num_procs, Rimb, copy_local_queues,
        #                         arr_num_tasks_before_execution, arr_victim_offloader)
            
        #     select_tasks_to_offload(clock, copy_local_queues, arr_victim_offloader,
        #                         iter, cost_balancing, cost_migration_delay, noise)
            
        #     offload_tasks(clock, copy_local_queues, remote_queues, arr_tmp_buffer_migrated_tasks,
        #                         slowdown_procs, slowdown_scales, iter, clock_rate, noise)

        # feedback_task_offloading(clock, num_procs, Rimb, copy_local_queues, arr_feedback_priorities, arr_feedback,
        #                             iter, arr_num_tasks_before_execution, arr_victim_offloader)

        # select_tasks_for_fb_offload(clock, copy_local_queues, arr_victim_offloader,
        #                             iter, cost_balancing, cost_migration_delay, noise)
        
        # fb_offload_tasks(clock, copy_local_queues, remote_queues, arr_tmp_buffer_migrated_tasks,
        #                     slowdown_procs, slowdown_scales, iter, clock_rate, noise)

        # --------------------------------------------------------
        # executing tasks and updating load
        # --------------------------------------------------------
        sum_remain_tasks = 0
        sum_being_exe_task = 0.0

        # pop task from the queues and execute them
        for i in range(num_procs):
            # --------------------------------------------------------
            # check the being-executed tasks
            # --------------------------------------------------------
            if arr_being_exe_tasks[i] != None:
                cur_task = arr_being_exe_tasks[i]
                end_time = cur_task.get_end_time()
                if clock == end_time:
                    update_total_load(i, arr_local_load, arr_remot_load, cur_task)
                    arr_being_exe_tasks[i] = None
                    # print('   P[{}]: end_task={:d}, end_time={:f}'.format(i, cur_task.tid, end_time))
                else:
                    update_total_load(i, arr_local_load, arr_remot_load, cur_task)
                    # print('   P[{}]: cur_task={:d}, end_time={:f}'.format(i, cur_task.tid, end_time))

            # --------------------------------------------------------
            # if no tasks being executed, then pop a new one
            # --------------------------------------------------------
            else:
                #--------------------------------
                # Prior 1: check remote queue
                #--------------------------------
                if len(remote_queues[i]) != 0:
                    rtask = remote_queues[i].pop(0)
                    stime = clock
                    etime = rtask.get_dur() + stime
                    rtask.set_time(stime, etime)
                    # denote remote-task being executed for Process i
                    arr_being_exe_tasks[i] = rtask
                    # profile tasks
                    arr_profiled_tasks[i].append(rtask)
                    
                #--------------------------------
                # Prior 2: check local queue
                #--------------------------------
                elif len(copy_local_queues[i]) != 0:
                    task = copy_local_queues[i].pop(0) # pop tasks from the front
                    stime = clock-1
                    etime = task.get_dur() + stime
                    task.set_time(stime, etime)
                    # denote task being executed for Process i
                    arr_being_exe_tasks[i] = task
                    # update the load value for Process i
                    update_total_load(i, arr_local_load, arr_remot_load, task)
                    # print('   P[{}]: new_task={:d}, end_time={:f}'.format(i, task.tid, etime))

                    # profile tasks
                    arr_profiled_tasks[i].append(task)

            # count remaining tasks
            sum_remain_tasks += len(copy_local_queues[i]) + len(remote_queues[i])
            if arr_being_exe_tasks[i] != None:
                sum_being_exe_task += 1
                
            # record the queue status at each clock
            arr_queue_status[i].append(len(copy_local_queues[i]) + len(remote_queues[i]))

        # --------------------------------------------------------
        # updating the queue status for main loop
        # --------------------------------------------------------
        sum_queues_length = sum_remain_tasks
        num_tasks_being_executed = sum_being_exe_task
        # print("[Tcomm] clock[{}]: (main_loop) sum_queues_length={}, num_tasks_being_executed={}".format(clock, queues_status, num_tasks_being_executed))

    # visualize task execution
    visualize_task_execution(arr_profiled_tasks, cost_balancing, cost_migration_delay)

    # feedback statistics
    # arr_task_execution_info = visualize_task_execution(arr_profiled_tasks, cost_balancing, cost_migration_delay)
    # get_feedback_statistics(arr_task_execution_info, arr_local_load, arr_remot_load, 
    #                             arr_num_tasks_before_execution, arr_feedback_priorities)

    # profile queue status
    # profile_queue_status(arr_queue_status, cost_balancing, cost_migration_delay)

    # return simulated results
    return [arr_local_load, arr_remot_load]

# -----------------------------------------------------
# Main function
# -----------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python estimate_upp... <context_input>')
        exit(1)
    filename = sys.argv[1]
    ctx_configs = pd.read_json(filename)

    print('-------------------------------------------')
    print('Configuration: ')
    print('-------------------------------------------')
    print(ctx_configs)
    print('-------------------------------------------\n')

    print('-------------------------------------------')
    print('Init the simulation: ')
    print('-------------------------------------------\n')
    init_res = init_simulation(ctx_configs)
    local_task_queues = init_res[0]
    slowdown_processes = init_res[1]
    slowdown_scales = init_res[2]
    num_procs = init_res[3]

    num_iterations = int(ctx_configs.iloc[5,0])
    clock_rate = ctx_configs.iloc[1,0]
    balancing_cost = ctx_configs.iloc[0,0]
    migration_cost = ctx_configs.iloc[3,0]
    fluctuation_noise = ctx_configs.iloc[4,0]

    # for feedback recording
    arr_feedback_values = []
    for i in range(num_iterations):
        arr_feedback_values.append(0)
    arr_feedback_based_priorities = []
    for i in range(num_procs):
        arr_feedback_based_priorities.append(0)

    for i in range(num_iterations):
        print('-------------------------------------------')
        print('ITERATION: {}'.format(i))
        print('-------------------------------------------\n')
        simu_res = simulate(local_task_queues, slowdown_processes, slowdown_scales, i, clock_rate, arr_feedback_values,
                            balancing_cost, migration_cost, fluctuation_noise, arr_feedback_based_priorities)

        statistic_info(simu_res[0], simu_res[1], clock_rate)
    
