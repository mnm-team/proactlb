import numpy as np
import pandas as pd
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import random
import imageio
import sys
import os

# Some constant definition
P  = 8   # total num. of ranks/processes
T  = 40  # total num. of tasks
Ti = 5   # num. tasks per rank
w  = 5   # wallclock time per task

# ----------------------------------------
# Total load per rank
# ----------------------------------------
def estimate_total_load():
    load_arr = []
    for i in range(P):
        Li = Ti * w
        load_arr.append(Li)
    return load_arr

# ----------------------------------------
# Performance Slowdown
# ----------------------------------------
def slowdown_performance(slow, n_impact_processes):
    speed_arr = []
    for i in range(P):
        if i < n_impact_processes:
            SPi = 1.0 + slow
            speed_arr.append(SPi)
        else:
            SPi = 1.0 + 0.0
            speed_arr.append(SPi)
    return speed_arr

def random_slowdown_performance(niter, nprocesses):
    speed_arr = []
    for iter in range(niter):
        tmp_speed = []
        if iter == 0: # keep the balanced perf
            for i in range(P):
                SPi = 1.0 + 0.0
                tmp_speed.append(SPi)
        else: # randomly slowdown the load
            rand_npro = random.randint(1, nprocesses/2)
            for i in range(P):
                if i <= rand_npro:
                    rand_slow = random.uniform(1.0, 20.0)
                    SPi = 1.0 + rand_slow
                    tmp_speed.append(SPi)
                else:
                    SPi = 1.0 + 0.0
                    tmp_speed.append(SPi)
        speed_arr.append(tmp_speed)
    
    return speed_arr


# ----------------------------------------
# Main
# ----------------------------------------
if __name__ == '__main__':

    # Set slowdown coefficient
    slow_set = np.arange(1.0, 9.0, 1.0, dtype=float)
    
    # Init a given load
    LOAD = estimate_total_load()
    
    # Slowdown load array
    SLOW_LOAD_ARR = []
    RIMB_MAT = []
    STDV_MAT = []
    
    # Generate random slowdown and Rimb over 100 iters of execution
    print('------------------------------------')
    S_arr = random_slowdown_performance(100, P)
    L_arr = []
    for i in range(len(S_arr)):
        L_tmp = np.array(LOAD) * np.array(S_arr[i])
        L_arr.append(L_tmp)
    
    # Calculate the Rimb ratios
    print('------------------------------------')
    Rimb_arr = []
    Std_arr = []
    for i in range(len(L_arr)):
        Lmax = np.max(L_arr[i])
        Lmin = np.min(L_arr[i])
        Lavg = np.average(L_arr[i])
        Rimb_arr.append(Lmax/Lavg - 1)
        Std_arr.append(np.std(L_arr[i]))
    print('------------------------------------')
    print(Rimb_arr)
    print(Std_arr)
    
    # ---------------------------------------------------------
    # Line styles
    # ---------------------------------------------------------
    lstyles = ['solid', 'solid']
    ldashstyles = ['dashed', 'dashed']
    lcolors = ['black', 'blue', 'darkgreen', 'red', 'darkorange', 'purple', 'teal']
    lmarkers = ['*', '^', 'o', 's', 'P', 'p', 'd']

    fig, ax1 = plt.subplots()
    ax1_indices = np.arange(1, len(Rimb_arr)+1)
    ax1.plot(ax1_indices, Rimb_arr, label='$R_{imb}$', linestyle='solid', marker='*', color='darkorange')
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Imbalance Ratio ($R_{imb}$)")
    ax1.set_ylim(0, 10)
    ax1.legend(loc="upper left", fontsize=7)

    ax2 = ax1.twinx()
    ax2_indices = np.arange(1, len(Rimb_arr)+1)
    ax2.plot(ax1_indices, Std_arr, label='$std$', linestyle='solid')
    ax2.set_ylabel("Standard Deviation ($std$)")
    ax2.legend(loc="upper right", fontsize=7)

    plt.grid()
    # plt.show()

    # save the figure
    fig_filename = "./double_line_randomized_slowdown.pdf"
    plt.savefig(fig_filename, bbox_inches='tight')
    