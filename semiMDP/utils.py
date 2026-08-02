"""Small assorted helpers: graph pretty-printing, equality checks, random
problem-size generation, and a simple timing context manager.
"""

import time
from pprint import pprint
import numpy as np


def pprint_graph(graph):
    """Pretty-print every node and edge attribute of a networkx graph."""
    print("Node information")
    for n in graph.nodes():
        print('{}:'.format(n))
        pprint(graph.nodes[n])

    print("\n Edge information")
    for e in graph.edges():
        print('{}:'.format(e))
        pprint(graph.edges[e])


def check_equal(l):
    """Return True if every element of array ``l`` is identical."""
    cond = (len(set(l.ravel().tolist())) == 1)
    return cond


def instance_generate():
    """Pick a random small problem size ``(num_machine, num_job)``.

    The number of jobs is drawn no smaller than the number of machines, which
    is a constraint enforced by :meth:`Simulator._sample_jssp_graph`.
    """
    num_machine = np.random.randint(5, 10)
    num_job = np.random.randint(num_machine, 10)

    return num_machine, num_job


def doable_ops_dr(machine):
    """Return the operations runnable on ``machine`` with no look-ahead.

    Standalone duplicate of the no-delay branch of :meth:`Machine.doable_ops`:
    an operation is doable when it has no predecessor (first op of a job) or
    its predecessor is already done.
    """
    doable_ops = []
    for op in machine.remain_ops:
        prev_start = op.prev_op is None
        if prev_start:
            doable_ops.append(op)
        else:
            prev_done = op.prev_op.node_status == 1  # DONE_NODE_SIG
            if prev_done:
                doable_ops.append(op)
    return doable_ops


class Timer:
    """Context manager that measures and prints the elapsed wall-clock time."""

    def __init__(self, name=None):
        if name is None:
            self.name = 'Operation'
        else:
            self.name = name

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print('{} : {} sec'.format(self.name, self.interval))
