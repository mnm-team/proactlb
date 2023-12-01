## Rebalancing Problem

* Formulation by Aggarwal et al. [SPAA-06]
    + Given $n$ jobs with the following sizes $\{s_{1}, s_{2}, ..., s_{n}\}$
    + Assigned on $m$ processors $\{P_{1}, P_{2}, ..., P_{m}\}$
    + Problem: relocating jobs among the processors to minimize the makespan, where
        - $makespan$: the completion time of all processors
        - relocating cost: $c_{ij}$ when moving a job from processor $i$ to processor $j$

* Re-formulating and mapping the problem to load rebalancing for task-based parallel applications in HPC
    + Given $n$ similar tasks in total, with the following execution time (load) $\{t_{0}, t_{1}, ..., t_{n-1}\}$
    + Assigned on $m$ processes (processors) $\{P_{0}, P_{1}, ..., P_{m-1}\}$
    + Problem: load imbalance among processes due to performance slowdown, need to relocate tasks or migrate tasks, where we also aim to minimize makespan,
        - $makespan$: the completion time of all processes
        - migration cost: $c_{ij}$ when moving a task from process $i$ to process $j$

* Example: $20$ tasks in total, assigned to $4$ processes, the load values are illustrated as follows.

![Example 1](./figures/rebalancing_formulation.png)

## A try for QUBO formulation

* Given $n$ tasks with execution time/load: $\{t_{0}, t_{1}, ..., t_{n-1}\}$
* Given a distribution on $m$ processes: $\{P_{0}, P_{1}, ..., P_{m-1}\}$.
* Binary variables following the given tasks: $\{x_{0}, x_{1}, ..., x_{n-1}\}$
* According the given information we know the load imbalance, e.g.,
    + In the above example, $n = 20$ tasks, $m = 4$ processes, tasks are binarized $\{x_{0}, x_{1}, ..., x_{19}\}$.
    + We know that: $P_{0}, P_{2}$ are underloaded processes, $P_{1}, P_{3}$ are overloaded.
    + Assume task migration happens, we have the objective function:

        `minimize` $y = \sum_{i \in n_{0}} t_{i} x_{i} + \sum_{i \in n_{2}} t_{i} x_{i} - (\sum_{i \in n_{1}} t_{i} x_{i} + \sum_{i \in n_{3}} t_{i} x_{i})$

        where, $\{n_{0}, n_{1}, ..., n_{m-1}\}$ is a new subset of tasks on each process.
    
    + The constraints include:

        $n_{0} + n_{1} + n_{2} + n_{3} = n$

        $n_{0} \leq k_{0}$, $n_{1} \leq k_{1}$, $n_{2} \leq k_{2}$, $n_{3} \leq k_{3}$, with $k_{i}$ is the maximun number of tasks that a process $i$ can hold.

## Another way to formulate the problem

* Transform to multi-partition problem
