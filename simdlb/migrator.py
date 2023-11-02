import numpy as np
import random

from task import *

"""
Offloading interface for migrating tasks from slow process to faster ones
    - run on a separate thread
    - bases on the result of balancers
    - take migrating tasks plus with Delay + Overhead(balancingcompute)
"""

# -----------------------------------------------------
# Util Functions
# -----------------------------------------------------
def randomize_cost(iter, clock, cost_val, noise):
    random.seed(iter + clock)
    # check noise value
    if noise > cost_val:
        noise = cost_val
    min = cost_val - noise
    max = cost_val + noise
    res = random.randint(min, max)
    return res

def estimate_new_task_runtime(old_dur, sld_processes, sld_scales, origin_node, new_node, iter, clock_rate, clock):
    
    # assume the task runtime is the same if remote node is not the slowdown one
    new_dur = old_dur

    # if the remote node is the slowdown one
    for i in range(len(sld_processes)):
        tmp_proc = sld_processes[i]
        tmp_scal = sld_scales[i]
        if origin_node == tmp_proc and new_node != tmp_proc:
            new_dur = randomize_cost(iter, clock, old_dur/2, old_dur/4)
        elif origin_node != tmp_proc and new_node == tmp_proc:
            dur = 1.0 / tmp_scal
            dur = dur * clock_rate
            new_dur = dur

    return new_dur

# -----------------------------------------------------
# Offload Functions
# -----------------------------------------------------
def select_tasks_to_offload(clock, local_queues, arr_victim_offloader, 
                            iter, cost_balancing, cost_migration_delay, noise):
    
    num_procs = len(arr_victim_offloader)
    for i in range(num_procs):
        num_tasks_for_migrating = len(arr_victim_offloader[i])
        if num_tasks_for_migrating > 0:
            # proceed the first pair
            off_rank = i
            vic_rank = arr_victim_offloader[off_rank].pop(0)
            queue_length_offloader = len(local_queues[off_rank])
            queue_length_victim = len(local_queues[vic_rank])
            diff_load = abs(queue_length_offloader - queue_length_victim)
            # print("Iter {} | clock {}: {}, diff_load({},{})={}".format(iter, clock, arr_victim_offloader, off_rank, vic_rank, diff_load))
        
            # choose the last task which is going to be offloaded
            if queue_length_offloader > 2 and diff_load > 2:
                for j in range(queue_length_offloader-1, 2, -1):
                    task2offload = local_queues[off_rank][j]
                    remote_node  = task2offload.remot_node
                    if remote_node == -1:
                        # set remote node
                        task2offload.set_remote_node(vic_rank)
                        # set migrate time
                        migrate_time = clock + randomize_cost(iter, clock, cost_balancing, noise)
                        task2offload.set_mig_time(migrate_time)
                        # set arrive time
                        arrive_time = migrate_time + randomize_cost(iter, clock, cost_migration_delay, noise)
                        task2offload.set_arr_time(arrive_time)
                        # print("[Tcomm] clock[{}]: (select_tasks) R{}-victim R{}, diff={}".format(clock, off_rank, vic_rank, diff_load))
                        break


def select_tasks_for_fb_offload(clock, local_queues, arr_victim_offloader, 
                            iter, cost_balancing, cost_migration_delay, noise):
    
    num_procs = len(arr_victim_offloader)
    for i in range(num_procs):
        num_tasks_for_migrating = len(arr_victim_offloader[i])
        if num_tasks_for_migrating > 0:
            # proceed the first pair
            off_rank = i
            vic_rank = arr_victim_offloader[off_rank].pop(0)
            queue_length_offloader = len(local_queues[off_rank])
            queue_length_victim = len(local_queues[vic_rank])
            
            # print("Iter {} | clock {}: {})".format(iter, clock, arr_victim_offloader))
        
            # choose the last task which is going to be offloaded
            if queue_length_offloader > 2:
                for j in range(queue_length_offloader-1, 2, -1):
                    task2offload = local_queues[off_rank][j]
                    remote_node  = task2offload.remot_node
                    if remote_node == -1:
                        # set remote node
                        task2offload.set_remote_node(vic_rank)
                        # set migrate time
                        migrate_time = clock + randomize_cost(iter, clock, cost_balancing, noise)
                        task2offload.set_mig_time(migrate_time)
                        # set arrive time
                        arrive_time = migrate_time + randomize_cost(iter, clock, cost_migration_delay, noise)
                        task2offload.set_arr_time(arrive_time)
                        print("[Tcomm] clock[{}]: (select_tasks) R{}-victim R{}".format(clock, off_rank, vic_rank))
                        break

def steal_tasks(clock, local_queues, arr_tasks_to_steal, arr_tmp_migrated_tasks, arr_request_msg, arr_accept_msg,
               iter, cost_migration_delay, sld_processes, sld_scales, clock_rate, noise):

    num_procs = len(arr_tmp_migrated_tasks)
    num_tmp_migrated_tasks = len(arr_tasks_to_steal)

    # ------------------------------------------------------
    # make notes for which tasks will be stolen
    # ------------------------------------------------------
    for i in range(num_tmp_migrated_tasks):
        tmp = arr_tasks_to_steal[i]
        idle_rank = tmp[0]
        busy_rank = tmp[1]
        time2migrate = tmp[2]
        queue_length_busy_rank = len(local_queues[busy_rank])

        if queue_length_busy_rank > 2:
            for j in range(queue_length_busy_rank-1, 2, -1):
                task2steal  = local_queues[busy_rank][j]
                time2arrive = time2migrate + randomize_cost(iter, clock, cost_migration_delay, noise)
                remote_proc = task2steal.remot_node
                if remote_proc == -1:
                    task2steal.set_remote_node(idle_rank)
                    task2steal.set_mig_time(time2migrate)
                    task2steal.set_arr_time(time2arrive)
                    print('[DEBUG] Clock {:5d}: R{} selected task to steal from R{}| len(arr_tasks_to_steal)={}\n'.format(clock, idle_rank, busy_rank, len(arr_tasks_to_steal)))
                    break

                    # remove the corresponding items in arr_tasks_to_steal
                    # arr_tasks_to_steal.pop(0)
                    
    # ------------------------------------------------------
    # proceed for sending tasks over the network
    # ------------------------------------------------------
    for i in range(num_procs):
        num_remain_tasks = len(local_queues[i])
        if num_remain_tasks > 2:
            # checkout the last task
            tmp_task = local_queues[i][-1]
            remote_proc = tmp_task.remot_node

            if remote_proc != -1 and tmp_task.mig_time == clock:
                print('[DEBUG] Clock {:5d}: R{} STEALING from R{}, migrate_time={}, arrive_time={}'.format(clock, remote_proc, i, tmp_task.mig_time, tmp_task.arr_time))
                tmp_task = None
                steal_candidate = local_queues[i].pop(-1)
                arr_tmp_migrated_tasks[i].append(steal_candidate)
                
                # remove all request/accept messages
                # print('[DEBUG] Clock {:5d}: arr_request_msg={}, arr_accept_msg={}\n'.format(clock, arr_request_msg, arr_accept_msg))
                
    # ------------------------------------------------------
    # proceed for receiving tasks over the network
    # ------------------------------------------------------
    for i in range(num_procs):
        num_tasks_being_migrated = len(arr_tmp_migrated_tasks[i])
        if num_tasks_being_migrated > 0:
            tmp_task = arr_tmp_migrated_tasks[i][0]
            arrival_time = tmp_task.arr_time
            if arrival_time == clock:
                # pop the task out
                tmp_task = None
                task2steal = arr_tmp_migrated_tasks[i].pop(0)
                # update the runtime
                old_proc = task2steal.local_node
                old_dur  = task2steal.get_dur()
                new_dur  = estimate_new_task_runtime(old_dur, sld_processes, sld_scales, old_proc, i, iter, clock_rate, clock)
                task2steal.dur = new_dur
                # add tasks to the local queue of idle processes
                new_proc = task2steal.remot_node
                local_queues[new_proc].append(task2steal)
                # remove 
                arr_accept_msg[old_proc].pop(0)
                arr_request_msg[new_proc].pop(0)
                print('[DEBUG] Clock {:5d}: R{} RECEIVED tasks from R{}'.format(clock, new_proc, old_proc))
                # print('[DEBUG] Clock {:5d}: popped items from arr_request_msg={}, arr_accept_msg={}\n'.format(clock, arr_request_msg, arr_accept_msg))        
                                   

def offload_tasks(clock, local_queues, remote_queues, arr_tmp_buffer_migrated_tasks, 
                    sld_processes, sld_scales, iter, clock_rate, noise):
    
    num_procs = len(arr_tmp_buffer_migrated_tasks)
    # ------------------------------------------------------
    # proceed for sending tasks over the network
    # ------------------------------------------------------
    for i in range(num_procs):
        num_remain_tasks = len(local_queues[i])
        if num_remain_tasks > 2:
            # check the tasks at rear
            tmp_offload = local_queues[i][-1] # choose the task at rear in the queue
            victim = tmp_offload.remot_node
            if victim != -1 and tmp_offload.mig_time == clock:
                # print("[Tcomm] clock[{}]: (offload_tasks) R{} to victim R{}".format(clock, i, victim))
                # pop the task
                tmp_offload = None
                task2migrate = local_queues[i].pop()
                # add task to the migrate buffer over network
                arr_tmp_buffer_migrated_tasks[victim].append(task2migrate)

    # ------------------------------------------------------
    # proceed for receiving tasks over the network
    # ------------------------------------------------------
    for i in range(num_procs):
        num_tasks_being_migrated = len(arr_tmp_buffer_migrated_tasks[i])
        if num_tasks_being_migrated > 0:
            tmp_receive = arr_tmp_buffer_migrated_tasks[i][0] # choose the task at front in the buffer
            arrive_time = tmp_receive.arr_time
            if arrive_time <= clock:
                # pop the task
                tmp_receive = None
                task2receive = arr_tmp_buffer_migrated_tasks[i].pop(0)
                # varied the task runtime
                old_node = task2receive.local_node
                old_dur = task2receive.get_dur()
                new_dur = estimate_new_task_runtime(old_dur, sld_processes, sld_scales, old_node, i, iter, clock_rate, clock)
                task2receive.dur = new_dur
                # add task to the remote queue at the victim side
                remote_queues[i].append(task2receive)
                # check migration
                # print("[Tcomm] clock[{}]: (receive_tasks) victim R{} from R{}, old_dur={}, new_dur={}".format(clock, i, old_node, old_dur, new_dur))
    return 0


def fb_offload_tasks(clock, local_queues, remote_queues, arr_tmp_buffer_migrated_tasks, 
                    sld_processes, sld_scales, iter, clock_rate, noise):
    
    num_procs = len(arr_tmp_buffer_migrated_tasks)
    # ------------------------------------------------------
    # proceed for sending tasks over the network
    # ------------------------------------------------------
    for i in range(num_procs):
        num_remain_tasks = len(local_queues[i])
        if num_remain_tasks > 2:
            # check the tasks at rear
            tmp_offload = local_queues[i][-1] # choose the task at rear in the queue
            victim = tmp_offload.remot_node
            if victim != -1 and tmp_offload.mig_time <= clock:
                print("[Tcomm] clock[{}]: (offload_tasks) R{} to victim R{}".format(clock, i, victim))
                # pop the task
                tmp_offload = None
                task2migrate = local_queues[i].pop()
                # add task to the migrate buffer over network
                arr_tmp_buffer_migrated_tasks[victim].append(task2migrate)

    # ------------------------------------------------------
    # proceed for receiving tasks over the network
    # ------------------------------------------------------
    for i in range(num_procs):
        num_tasks_being_migrated = len(arr_tmp_buffer_migrated_tasks[i])
        if num_tasks_being_migrated > 0:
            tmp_receive = arr_tmp_buffer_migrated_tasks[i][0] # choose the task at front in the buffer
            arrive_time = tmp_receive.arr_time
            if arrive_time <= clock:
                # pop the task
                tmp_receive = None
                task2receive = arr_tmp_buffer_migrated_tasks[i].pop(0)
                # varied the task runtime
                old_node = task2receive.local_node
                old_dur = task2receive.get_dur()
                new_dur = estimate_new_task_runtime(old_dur, sld_processes, sld_scales, old_node, i, iter, clock_rate, clock)
                task2receive.dur = new_dur
                # add task to the remote queue at the victim side
                remote_queues[i].append(task2receive)
                # check migration
                print("[Tcomm] clock[{}]: (receive_tasks) victim R{} from R{}, old_dur={}, new_dur={}".format(clock, i, old_node, old_dur, new_dur))
    return 0