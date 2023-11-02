from numpy.lib.npyio import load
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import statistics
import re
import sys
import os
import csv



"""The main function
"""
if __name__ == "__main__":

    # -------------------------------------------
    # Input about matrix sizes and their amount
    # -------------------------------------------
    LIST_SIZES = [128, 256, 512, 768, 1024, 1280]
    LIST_AMOUNT = [200, 200, 100, 100, 50, 50]
    matrix_sizes = []
    load_count = 0.0
    for i in range(len(LIST_AMOUNT)):
        ntasks = LIST_AMOUNT[i]
        for j in range(ntasks):
            matrix_sizes.append(LIST_SIZES[i])
        load_count += ntasks * LIST_SIZES[i]

    std_dev_matrix_sizes = statistics.stdev(matrix_sizes)
    pstd_dev_matrix_sizes = statistics.pstdev(matrix_sizes)
    # print("matrix_sizes: len={}, vals={}".format(len(matrix_sizes), matrix_sizes))

    # -------------------------------------------
    # Generate list of matrix sizes for each rank
    # -------------------------------------------
    NUM_RANKS = 8
    TASK_INPUT = []
    TASK_AMOUNT = []
    for i in range(NUM_RANKS):
        if i == 0:
            TASK_INPUT.append(matrix_sizes)
            TASK_AMOUNT.append(LIST_AMOUNT)
        else:
            TASK_INPUT.append([])
            TASK_AMOUNT.append([])

    # -------------------------------------------
    # Key factors for generating the input
    # -------------------------------------------
    stdev_taskweight = 0.0
    stdev_rankimb = 2.0
    I = 4.0
    N = load_count
    # Assume, max num tasks at R0 = sum(amount_list)
    #   + total_load at R0: a0*s0 + a1*s1 + ...
    #   + the remaining load of other ranks
    #       R = (num_ranks - 1 - I)*N / (I + 1)
    # where, I is the imbalance ratio
    R = (NUM_RANKS - 1 - I) * N / (I + 1)
    if R <= 0:
        print("Error: num_ranks, I, and N cannot match to generate the input")
        exit(1)

    avg_load_per_rranks = (int)(R / (NUM_RANKS - 1))
    ratio_vs_N = avg_load_per_rranks / N

    for r in range(1, NUM_RANKS):
        # estimate new amount of tasks per each size
        count = 0
        if r == NUM_RANKS - 1:
            ratio_vs_N = (R - avg_load_per_rranks*(NUM_RANKS-2)) / N

        for i in range(len(LIST_AMOUNT)):
            if i == len(LIST_AMOUNT) - 1:
                new_amount = (int)(ratio_vs_N * sum(LIST_AMOUNT)) - count
                TASK_AMOUNT[r].append(new_amount)
            else:
                new_amount = (int)(ratio_vs_N * LIST_AMOUNT[i])
                TASK_AMOUNT[r].append(new_amount)
                count += new_amount

        # generate the input
        for i in range(len(TASK_AMOUNT[r])):
            ntasks = TASK_AMOUNT[r][i]
            for j in range(ntasks):
                TASK_INPUT[r].append(LIST_SIZES[i])

    max_load = sum(TASK_INPUT[0])
    avg_load = 0.0
    tot_load = 0.0
    tot_load_per_rank_arr = []
    for i in range(len(TASK_INPUT)):
        tot_load += sum(TASK_INPUT[i])
        tot_load_per_rank_arr.append(sum(TASK_INPUT[i]))
    avg_load = tot_load / len(TASK_INPUT)

    print("-----------------------------------------------")
    print("Imbalance Ratio: {}".format((max_load - avg_load) / avg_load))
    print("stdev(RANK_IMB): {}, arr={}".format(statistics.stdev(tot_load_per_rank_arr), tot_load_per_rank_arr))
    print("stdev(LIST_SIZES): {}".format(std_dev_matrix_sizes))
    print("pstdev(LIST_SIZES): {}".format(pstd_dev_matrix_sizes))
    print("-----------------------------------------------")
    for r in range(NUM_RANKS):
        print("Rank{}: num_tasks={}, total_load={}, size_arr={}".format(r, sum(TASK_AMOUNT[r]), sum(TASK_INPUT[r]), TASK_INPUT[r]))
        print("-----------------------------------------------")
