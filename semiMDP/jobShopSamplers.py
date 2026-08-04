"""Random instance generation for the Job-Shop Scheduling Problem (JSSP)."""

import numpy as np


def jssp_sampling(m, n, low=5, high=100):
    """Sample a random JSSP instance.

    Args:
        m: number of machines.
        n: number of jobs (each job performs ``m`` operations, one per machine).
        low: inclusive lower bound for processing times.
        high: exclusive upper bound for processing times.

    Returns:
        machine_mat: ``(n, m)`` int array; row ``i`` is a random permutation of
            ``range(m)`` giving the order in which job ``i`` visits the machines.
        process_time_mat: ``(n, m)`` int array of processing times drawn
            independently and uniformly from ``[low, high)``.
    """
    machine_mat = np.ndarray(shape=(n, m))
    process_time_mat = np.random.randint(low, high, size=(n, m))
    for i in range(n):
        machine_mat[i] = np.random.permutation(m)
    return machine_mat, process_time_mat
