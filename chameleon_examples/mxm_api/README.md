## MxM exaple for testing Chameleon with Load Balancing problems
Each task in Chameleon is defined as a function `mxm` with the inputs are Matrix A, Matrix B, the ouput would be Matrix C.

### Experiment 1: Data and task migration throughput
The test is performed on 2 ranks, R0 has 500 tasks, while R1 has 10 tasks to create the imbalance ratio about ~0.99 between 2 ranks (uniform-load). This means that R0 has to migrate tasks to R1. There are 3 testcases with the varied matrix sizes (task-sizes): 256, 512, 1024; running on 2 nodes, 1 rank/node, 1 omp-thread for executing tasks and 1 pthread for communication.

* On CoolMUC2:

|Task-sizes | R(Imb) | Task-mig-throughput | Data-trans-throughput |
|-----------|--------|---------------------|-----------------------|
| 256       |  0.99  | 792.144 (tasks/s)   |  1252.491 (MB/s)      |
| 512       |  0.99  | 205.427 (tasks/s)   |  1295.773 (MB/s)      |
| 1024      |  0.99  |  75.766 (tasks/s)   |  1906.738 (MB/s)      |

* On SuperMUC-NG:

|Task-sizes | R(Imb) | Task-mig-throughput | Data-trans-throughput |
|-----------|--------|---------------------|-----------------------|
| 256       |  0.99  | 1083.292 (tasks/s)  |  1704.062 (MB/s)      |
| 512       |  0.99  |  407.508 (tasks/s)  |  2565.322 (MB/s)      |
| 1024      |  0.99  |  112.749 (tasks/s)  |  2863.799 (MB/s)      |

* On BEAST-system:

|Task-sizes | R(Imb) | Task-mig-throughput | Data-trans-throughput |
|-----------|--------|---------------------|-----------------------|
| 256       |  0.99  | 1984.238 (tasks/s)  |  3165.823 (MB/s)      |
| 512       |  0.99  |  494.987 (tasks/s)  |  3159.353 (MB/s)      |
| 1024      |  0.99  |  162.855 (tasks/s)  |  3858.058 (MB/s)      |

### Experiment 2: Speedup and Load Balancing Ratio
The test is devided into uniform-load and nonuniform-load with MxM example. The task is still defined by the mxm kernel which the inputs are Matrix A, B. Uniform load means each task has the same input sizes, while nonuniform load has different sizes. The imbalance ratio (R(Imb)) is calculated by max(load) and avg(load), R(Imb) = (max - avg)/avg.
* On CoolMUC2:

(1) Imbalance Ratio Table:

|Case 		      | Imb.0  | Imb.1 	| Imb.2  | Imb.3  | Imb.4  |
|-----------------|--------|--------|--------|--------|--------|
|Baseline 	      | 0.018  | 0.990 	| 0.994  | 1.765  | 3.927  |
|Baseline-nonuni  | 0.029  | 0.527 	| 1.003  | 1.368  | 1.743  |
|Migration 	      | 0.009  | 0.091 	| 0.078  | 0.634  | 1.890  |
|Migration-nonuni | 0.025  | 0.025 	| 0.025  | 0.025  | 0.025  |

(2) Taskwait-Sum Time:

|Case 		      | Imb.0    | Imb.1 	| Imb.2    | Imb.3    | Imb.4    |
|-----------------|----------|----------|----------|----------|----------|
|Baseline 	      |   10.385 |  110.816 |  111.659 |   97.461 |   97.677 |
|Baseline-nonuni  | 1778.401 | 1864.739 | 1848.493 | 1961.131 | 1857.556 |
|Migration 	      |   10.954 |   68.449 |   69.584 |   60.047 |   59.588 |
|Migration-nonuni | 1891.465 | 1352.962 | 1144.062 | 1100.013 |  930.931 |

Notes:
* Uniform-load: matrix size = 512, number of tasks per rank = `400 400 400 400 400 400 400 400` (baseline or Imb.0), 4 nodes, 2 ranks/node, 13 omp-threads/rank for executing tasks and 1 pthread/rank for communication.
* Nonuniform-load: matrix sizes = `128,256,512,768,1024,1280`, varied number of tasks per rank (based on Imb.). Also, 4 nodes, 2 ranks/node, 13 omp-threads/rank for executing tasks and 1 pthread/rank for communication.

### Investigation: Case Imb.4, uniform-load
```
--------------------------------------------------------------------
Reading file: err_81457_uniform_cham_commthread_case4.txt...
--------------------------------------------------------------------
Rank | sum(runtime|loc_time|rem_time) | num(loc_tasks|rem_tasks) | avg_taskwait | data-trans | migr-rate
5 	   6.34     6.34     0.00 		        250	        0	            1.41 	   0.00 	   0.00
2 	   6.35     6.35     0.00 		        250	        0	            1.40 	   0.00 	   0.00
3 	   6.41     6.41     0.00 		        250	        0	            1.40 	   0.00 	   0.00
4 	   6.44     6.44     0.00 		        250	        0	            1.40 	   0.00 	   0.00
6 	  11.19    11.19     0.00 		        450	        0	            1.33 	   0.00 	   0.00
7 	  11.22    11.22     0.00 		        450	        0	            1.33 	   0.00 	   0.00
1 	  12.46    12.46     0.00 		        500	        0	            1.31 	   0.00 	   0.00
0 	  96.82    96.82     0.00 		        4000	    0	            0.01 	   0.00 	   0.00
--------------------------------------------------------------------
Imb.ratio by max&mean: 4.926537
Imb.ratio by max&min:  0.934548
Imb.ratio by max&avg:  3.926537
--------------------------------------------------------------------
```
```
--------------------------------------------------------------------
Reading file: err_81458_uniform_cham_mig_case4.txt...
--------------------------------------------------------------------
Rank | sum(runtime|loc_time|rem_time) | num(loc_tasks|rem_tasks) | avg_taskwait | data-trans | migr-rate
4 	   8.86     6.03     2.83 		        242           119           0.78 	1185.81 	     17.39
5 	   8.91     5.97     2.94 		        231	          103	        0.78 	 952.77 	     36.05
3 	   9.08     5.94     3.14 		        230	          117	        0.78 	 982.87 	     40.01
7 	   9.78     9.14     0.63 		        361	          25	        0.77 	1178.48 	    254.15
6 	   9.99     9.40     0.59 		        363	          25	        0.76 	1072.75 	    236.05
2 	  16.99     6.10    10.89 		        243	         458	        0.66 	1192.65 	      4.68
1 	  39.67     9.74    29.93 		        376	        1271            0.31 	1939.94 	     50.12
0 	  58.43    58.43     0.00 		        2236	       0	        0.02 	2090.89 	    590.79
--------------------------------------------------------------------
Imb.ratio by max&mean: 2.890366
Imb.ratio by max&min:  0.848285
Imb.ratio by max&avg:  1.890366
--------------------------------------------------------------------
```

### Investigation: Case Imb.4, nonuniform-load
```
--------------------------------------------------------------------
Reading file: err_81448_nonuniform_cham_commthread_case4.txt...
--------------------------------------------------------------------
Rank | sum(runtime|loc_time|rem_time) | num(loc_tasks|rem_tasks) | avg_taskwait | data-trans | migr-rate
7 	    115.19      115.19     0.00 		 295	        0	    26.81 	        0.00 	   0.00
3 	    506.98      506.98     0.00 		 270	        0	    20.78 	        0.00 	   0.00
2 	    509.54      509.54     0.00 		 270	        0	    20.74 	        0.00 	   0.00
6 	    531.42      531.42     0.00 		 270	        0	    20.40 	        0.00 	   0.00
4 	    582.33      582.33     0.00 		 270	        0	    19.62 	        0.00 	   0.00
5 	    592.81      592.81     0.00 		 270	        0	    19.46 	        0.00 	   0.00
1 	    609.55      609.55     0.00 		 270	        0	    19.20 	        0.00 	   0.00
0 	   1799.34     1799.34     0.00 		3500	        0	     0.89 	        0.00 	   0.00
--------------------------------------------------------------------
Imb.ratio by max&mean: 2.743340
Imb.ratio by max&min:  0.935981
Imb.ratio by max&avg:  1.743340
--------------------------------------------------------------------
```

```
--------------------------------------------------------------------
Reading file: err_81449_nonuniform_cham_mig_case4.txt...
--------------------------------------------------------------------
Rank | sum(runtime|loc_time|rem_time) | num(loc_tasks|rem_tasks) | avg_taskwait | data-trans | migr-rate
0 	    681.87      534.89   146.97 		803	         85	        3.83 	        1142.88 	 131.56
1 	    696.56      320.13   376.43 		150	        447	        3.61 	        1112.86 	  19.40
7 	    718.68      138.33   580.34 		207	        626	        3.27 	         924.14 	  10.31
3 	    719.74      238.81   480.93 		138	        498	        3.25 	         998.87 	  16.19
6 	    723.18      277.30   445.88 		135	        411	        3.20 	        1082.36 	  18.21
5 	    743.36      342.06   401.30 		107	        444	        2.89 	        1073.36 	  21.08
4 	    752.11      251.30   500.81 		140	        514	        2.75 	         955.68 	  15.46
2 	    770.61      196.63   573.98 		134	        576	        2.47 	        1024.43 	  15.74
--------------------------------------------------------------------
Imb.ratio by max&mean: 1.061794
Imb.ratio by max&min:  0.115162
Imb.ratio by max&avg:  0.061794
--------------------------------------------------------------------
```