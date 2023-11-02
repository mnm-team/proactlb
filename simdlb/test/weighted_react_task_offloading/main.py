from queue import Queue
import numpy as np
import sys


# -----------------------------------------------------
# Load information
# -----------------------------------------------------

LOCAL_TASKS  = [6, 8, 8, 10, 10, 10, 18, 18]
REMOTE_TASKS = [94, 81, 84, 81, 86, 92, 96, 98]
TOTAL_TASKS  = []

N = 100 # given distribution of tasks
P = len(LOCAL_TASKS)

LOCAL_LOAD  = [30.0, 40.0, 40.0, 10.0, 10.0, 10.0, 18.0, 18.0]
REMOTE_LOAD = [454.789, 391.962, 420.000, 197.928, 148.118, 148.637, 201.966, 205.793]
TOTAL_LOAD  = []

# Check the total number of tasks and load values
for i in range(P):
    total_tasks = LOCAL_TASKS[i] + REMOTE_TASKS[i]
    total_load = LOCAL_LOAD[i] + REMOTE_LOAD[i]
    TOTAL_TASKS.append(total_tasks)
    TOTAL_LOAD.append(total_load)

# Check the average load value and calculate the distance
Lavg = np.average(TOTAL_LOAD)
DISTANCE_ARR = []
for i in range(P):
    diff_load = TOTAL_LOAD[i] - Lavg
    DISTANCE_ARR.append(int(diff_load))

# Interpolate the local load values
INTP_LOCAL_LOAD = []
for i in range(P):
    local_load_per_task = LOCAL_LOAD[i] / LOCAL_TASKS[i]
    interpolated_local_load = local_load_per_task * N
    INTP_LOCAL_LOAD.append(interpolated_local_load)




  