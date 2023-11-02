import numpy as np
import random


"""Balancing interface for taking decision when we migrate tasks
    - run on a separate thread
    - check on-the-fly the status of queues
"""

"""Class Message for exchanging steal request and accept
    - sender: the rank who sends message
    - receiver: the rank who receives message. If -1, it means to all
    - info: e.g., 'steal_request', 'steal_accept'
    - send_time: time when the message is sent
    - recv_time: time when the message is arrived/received
"""
class Message:
    def __init__(self, sender, receiver, info):
        self.sender = sender
        self.receiver = receiver
        self.info = info
        # for time to send/recv messages
        self.send_time = 0.0
        self.recv_time = 0.0
        # for marking nodes accept a request
        self.accept_rank = -1
            
    def set_time(self, send_time, recv_time):
        self.send_time = send_time
        self.recv_time = recv_time

    def set_accept_proc(self, rank):
        self.accept_rank = rank
    
    def print_msg_info(self):
        print('Message: sender({}), receiver({}), info({}), time(send-{} - recv-{})'.format(self.sender,
                self.receiver, self.info, self.send_time, self.recv_time))

# -----------------------------------------------------
# Constant Definition
# -----------------------------------------------------
MIN_REL_LOAD_IMBALANCE = 0.05
MIN_TASKS_IN_QUEUE_FOR_OFFLOAD = 2
MIN_TASKS_IN_QUEUE_FOR_STEAL = 2
MIN_ABS_LOAD_DIFFERENCE = 2

# -----------------------------------------------------
# Util Functions
# -----------------------------------------------------
def check_imbalance_status(clock, local_queues, arr_being_exe_tasks):
    num_procs = len(local_queues)
    num_tasks_being_executed = 0
    queue_sizes = []
    for i in range(num_procs):
        queue_sizes.append(len(local_queues[i]))
        if arr_being_exe_tasks[i] != None:
            num_tasks_being_executed += 1

    # calculate Rimb
    lmin = np.min(queue_sizes)
    lmax = np.max(queue_sizes)
    lavg = np.average(queue_sizes)
    Rimb_avg = 0.0
    Rimb_minmax = 0.0
    if lavg != 0 and lmax != 0:
        Rimb_avg = (lmax - lavg) / lavg
        Rimb_minmax = (lmax - lmin) / lmax

    # show queue status
    # print("[Tcomm] Clock[{}], Rimb_minmax={:5.2f}, Q_size=[{},{},{},{},{},{},{},{}]".format(clock, Rimb_minmax, 
    #     queue_sizes[0], queue_sizes[1], queue_sizes[2], queue_sizes[3],
    #     queue_sizes[4], queue_sizes[5], queue_sizes[6], queue_sizes[7]))

    return Rimb_minmax

def check_queue_status(clock, num_procs, local_queues):
    arr_empty_queues = []
    arr_busy_queues  = []
    for r in range(num_procs):
        current_rank = r
        queue_size   = len(local_queues[r])
        if queue_size == 0:
            arr_empty_queues.append([current_rank, queue_size])
        elif queue_size > MIN_TASKS_IN_QUEUE_FOR_STEAL:
            arr_busy_queues.append([current_rank, queue_size])
    return [arr_empty_queues, arr_busy_queues]


def exchange_steal_request(clock, arr_queue_status, arr_request_msg, arr_accept_msg, cost_balancing):
    # check empty and busy queues
    empty_queues = arr_queue_status[0]
    busy_queues  = arr_queue_status[1]
    LATENCY = int(cost_balancing)

    # -------------------------------------
    # IDLE processes
    # -------------------------------------
    if len(empty_queues) != 0:
        for i in range(len(empty_queues)):
            idle_rank = empty_queues[i][0]
            # check if this rank has not sent any requests
            if len(arr_request_msg[idle_rank]) == 0:
                request_msg = Message(idle_rank, -1, 'steal_request')
                request_msg.set_time(clock, clock+LATENCY)
                arr_request_msg[idle_rank].append(request_msg)
                print('[DEBUG] Clock {:5d}: R{} SENDS steal_request (sent at {:5d}, recv at {:5d})\n'.format(
                        clock, idle_rank, clock, clock+LATENCY))

            # check if this rank has already sent a request but not selected, then we re-send it
            else:
                checkout_req_msg = arr_request_msg[idle_rank][0]
                if checkout_req_msg.info == 'steal_request' and checkout_req_msg.recv_time < clock:
                    checkout_req_msg.set_time(clock, clock+LATENCY) # renew time

        # -------------------------------------
        # BUSY processes
        # -------------------------------------
        # check ranks with the current steal-request message has been sent and arrived
        list_idle_ranks = []
        for i in range(len(arr_request_msg)):
            if len(arr_request_msg[i]) != 0:
                req_msg = arr_request_msg[i][0]
                if req_msg.info == 'steal_request' and req_msg.recv_time == clock:
                    list_idle_ranks.append(i)

        # check busy ranks available
        list_busy_ranks = []
        for i in range(len(busy_queues)):
            busy_rank = busy_queues[i][0]
            list_busy_ranks.append(busy_rank)

        # check the list of idle ranks and busy ranks
        # if clock > 19005 and clock < 19030:
        #     print('[DEBUG] Clock {:5d}: idle_ranks={}, busy_ranks={}\n'.format(clock, list_idle_ranks, list_busy_ranks))

        # check all pairs of idle and busy ranks
        num_busy_candidates = len(list_busy_ranks)
        num_idle_candidates = len(list_idle_ranks)
        min_loops = min(num_idle_candidates, num_busy_candidates)
        if num_busy_candidates != 0 and num_idle_candidates != 0:
            for i in range(min_loops):
                # random choose a busy candidate
                random.seed(i)
                busy_candidate = random.choice(list_busy_ranks)
                # random choose an idle candidate
                idle_candidate = random.choice(list_idle_ranks)
                # checkout the accept message of busy_candidate
                if len(arr_accept_msg[busy_candidate]) == 0:
                    # create a new accept message
                    accept_msg = Message(busy_candidate, idle_candidate, 'steal_accept')
                    accept_msg.set_time(clock, clock+LATENCY)
                    arr_accept_msg[busy_candidate].append(accept_msg)
                    print('[DEBUG] Clock {:5d}: R{} CONFIRMS accept for R{} (sent at {:5d}, will recv at {:5d})\n'.format(
                                clock, busy_candidate, idle_candidate, clock, clock+LATENCY))
                    
                    # remove the candidate
                    list_busy_ranks.remove(busy_candidate)

                    # if we let that choosing randomly
                    # do nothing with the selected idle candidate

                    # if we get away from the selected idle candidate
                    # list_idle_ranks.remove(idle_candidate)
                # -------------------------------------
                # else: means the accept message is not yet arrived,
                # because if it arrived, then it will be removed
                # -------------------------------------

def recv_steal_accept(clock, arr_queue_status, arr_request_msg, arr_accept_msg, cost_balancing):

    # array for storing which tasks will be stolen
    arr_tasks_to_steal = []
    LATENCY = int(cost_balancing)

    # -------------------------------------
    # IDLE processes
    # -------------------------------------
    empty_queues = arr_queue_status[0]
    if len(empty_queues) != 0:

        # array for marking busy processes which accepted the request
        list_accepted_busy_ranks = []
        for i in range(len(arr_request_msg)):
            list_accepted_busy_ranks.append([])

        # array of idle candidates at the current clock
        list_idle_ranks = []
        for i in range(len(empty_queues)):
            idle_rank = empty_queues[i][0]
            list_idle_ranks.append(idle_rank)

        # filtering the busy candidates which accepted the requests corresponding to each idle candidate
        list_busy_ranks = []
        for i in range(len(arr_accept_msg)):
            if len(arr_accept_msg[i]) != 0:
                checkout_accept_msg = arr_accept_msg[i][0]
                arrived_time = checkout_accept_msg.recv_time
                selected_idle_rank = checkout_accept_msg.receiver
                if arrived_time == clock and selected_idle_rank in list_idle_ranks:
                    list_busy_ranks.append(i)
                    list_accepted_busy_ranks[selected_idle_rank].append(i)
        
        # check the idle ranks one-by-one
        for i in list_idle_ranks:
            if len(list_accepted_busy_ranks[i]) != 0:
                # randomly choose a busy candidate
                busy_candidate = random.choice(list_accepted_busy_ranks[i])
                # add tasks to the array of marking which tasks can be stolen
                idle_candidate = i
                time2steal = clock + LATENCY
                task2steal = [idle_candidate, busy_candidate, time2steal]
                arr_tasks_to_steal.append(task2steal)
                # check and remove the request/accept messages
                checkout_acc_msg = arr_accept_msg[busy_candidate][0]
                checkout_acc_msg.info = 'sending_task'
                checkout_req_msg = arr_request_msg[i][0]
                checkout_req_msg.info = 'stealing_task'
                checkout_req_msg.set_accept_proc(busy_candidate)
                # print('[DEBUG] Clock {:5d}: R{} will STEAL task from R{} at {}\n'.format(clock, i, busy_candidate, time2steal))
    
    return arr_tasks_to_steal

# -----------------------------------------------------
# Debug Functions
# -----------------------------------------------------
def show_sorted_load(clock, sortLCQ, local_queues):
    str_R = ""
    str_L = ""
    for i in sortLCQ:
        str_R += str(i) + ","
        str_L += str(len(local_queues[i])) + ","
    print("[Tcomm] clock[{}]: R{} | L={}".format(clock, str_R, str_L))

# -----------------------------------------------------
# Balancing Functions
# -----------------------------------------------------

def react_task_offloading(clock, num_procs, Rimb, local_queues, arr_num_tasks_before_execution, arr_victim_offloader):
    
    # check the imb condition
    if Rimb >= MIN_REL_LOAD_IMBALANCE:

        # check queue sizes
        queue_sizes = []
        for i in range(num_procs):
            queue_sizes.append(len(local_queues[i]))

        # sort the queue sizes
        sortLCQ = np.argsort(queue_sizes)
        # show_sorted_load(clock, sortLCQ, local_queues)
        
        # select offloader-victim
        for i in range(int(num_procs/2), num_procs):
            offloader_rank = sortLCQ[i]
            victim_rank = sortLCQ[num_procs-i-1]

            offloader_qsize = len(local_queues[offloader_rank])
            victim_qsize = len(local_queues[victim_rank])

            diff_load = abs(offloader_qsize - victim_qsize)

            # check the number of selected tasks for migration at the moment
            num_offload_candidates = len(arr_victim_offloader[offloader_rank])

            # if ok, select more tasks for offloading
            if diff_load > MIN_ABS_LOAD_DIFFERENCE and \
                offloader_qsize > MIN_TASKS_IN_QUEUE_FOR_OFFLOAD and \
                num_offload_candidates < arr_num_tasks_before_execution[offloader_rank]:

                # a pair of offloader-victim
                arr_victim_offloader[offloader_rank].append(victim_rank)
                # print("[Tcomm] clock[{}]: offloader={}, victim={}...".format(clock, offloader_rank, victim_rank))
    return 0

def feedback_task_offloading(clock, num_procs, Rimb, local_queues, arr_feedback_priorities, arr_feedback,
                                   iter, arr_num_tasks_before_execution, arr_victim_offloader):
    # check the imb condition
    if Rimb >= MIN_REL_LOAD_IMBALANCE and iter > 0 and arr_feedback[iter] == 0:

        # check queue sizes
        queue_sizes = []
        for i in range(num_procs):
            queue_sizes.append(len(local_queues[i]))

        # sort the queue sizes
        sortLCQ = np.argsort(queue_sizes)

        # check the current number of victims
        is_having_victims = 0
        for i in range(num_procs):
            num_vics = len(arr_victim_offloader[i])
            if num_vics > 0:
                is_having_victims = 1
            
        is_having_feedback = np.sum(arr_feedback_priorities)
        arr_overloaded_ranks = []
        arr_underloaded_ranks = []
        if is_having_feedback != 0:
            for i in range(num_procs):
                fb_value = arr_feedback_priorities[i]
                if fb_value > 0:
                    arr_overloaded_ranks.append(i)
                else:
                    arr_underloaded_ranks.append(i)

        # in the case, there is no victims yet
        if is_having_victims == 0 and is_having_feedback != 0:
            for o in arr_overloaded_ranks:
                num_offloaded_tasks = int(arr_feedback_priorities[o])
                num_underloaded_ranks = len(arr_underloaded_ranks)
                for t in range(num_offloaded_tasks):
                    rid = t % num_underloaded_ranks
                    arr_victim_offloader[o].append(arr_underloaded_ranks[rid])
            # enable feedback flag to not enter this again
            arr_feedback[iter] = 1
                    