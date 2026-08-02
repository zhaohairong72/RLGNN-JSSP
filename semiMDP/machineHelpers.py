"""Machine-side data structures for the JSSP environment.

:mod:`operationHelpers` models the operations; this module models the
machines that execute them. A :class:`MachineManager` owns every machine and
forwards dispatch / time-stepping calls; each :class:`Machine` tracks the
operations assigned to it and its own processing state, including the
"delayed op" look-ahead reservation mechanic.
"""

import random
from collections import OrderedDict
import numpy as np
#from semiMDP.operationHelpers import Operation, NodeProcessingTimeOperation
from semiMDP.configs import (PROCESSING_NODE_SIG,
                             DONE_NODE_SIG,
                             DELAYED_NODE_SIG)


class MachineManager:
    """Owns every machine of an instance and routes dispatch / time-step calls.

    Args:
        machine_matrix: ``(num_jobs, num_machines)`` int array.
        job_manager: the :class:`JobManager` whose operations are scheduled here.
        delay: whether machines may use look-ahead reservations (see Machine).
        verbose: print machine load/unload events when True.
    """

    def __init__(self,
                 machine_matrix,
                 job_manager,
                 delay=True,
                 verbose=False):

        machine_matrix = machine_matrix.astype(int)

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(job_manager[job_id][step_id])
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose)

    def do_processing(self, t):
        """Advance processing of every machine by one time step at time ``t``."""
        for _, machine in self.machines.items():
            machine.do_processing(t)

    def load_op(self, machine_id, op, t):
        """Load operation ``op`` onto machine ``machine_id`` at time ``t``."""
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        """Return the machine with id ``index``."""
        return self.machines[index]

    def get_available_machines(self, shuffle_machine=True):
        """Return machines that currently have doable, unreserved work.

        Args:
            shuffle_machine: if True, return the machines in random order
                (used so a random policy does not always pick the same machine).

        Returns:
            List of available :class:`Machine` objects.
        """
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list

    # get idle machines' list
    def get_idle_machines(self):
        """Return machines that are idle (no current op) and still have work."""
        m_list = []
        for _, m in self.machines.items():
            if m.current_op is None and not m.work_done():
                m_list.append(m)
        return m_list

    # calculate the length of queues for all machines
    def cal_total_cost(self):
        """Total queued-work cost: sum of no-delay doable op counts per machine.

        Used as the ``utilization`` reward (its negation) in :meth:`Simulator.observe`.
        """
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)
        return c

    # update all cost functions of machines
    def update_cost_function(self, cost):
        """Add ``cost`` to every machine's accumulated cost counter."""
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        """Return all machines in random order."""
        m_list = [m for _, m in self.machines.items()]
        return random.sample(m_list, len(m_list))

    def all_delayed(self):
        """True if every machine is currently holding a delayed (reserved) op."""
        return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        """True if the fab is deadlocked: no machine available and all delayed.

        No op can make progress, so the schedule cannot continue.
        """
        # All machines are not available and all machines are delayed.
        all_machines_not_available_cond = not self.get_available_machines()
        all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond and all_machines_delayed_cond


class NodeProcessingTimeMachineManager(MachineManager):
    """:class:`MachineManager` variant paired with the node-processing-time graph.

    Rebuilds the machine dict over :class:`NodeProcessingTimeOperation` jobs;
    operation behaviour is inherited unchanged from :class:`MachineManager`.
    """

    def __init__(self, machine_matrix, job_manager, delay=True, verbose=False):

        super().__init__(machine_matrix, job_manager, delay, verbose)
        machine_matrix = machine_matrix.astype(int)

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(job_manager[job_id][step_id])
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose)


class Machine:
    """A single machine that processes a fixed set of possible operations.

    Attributes include the backlog (``possible_ops`` / ``remain_ops``), the
    currently loaded ``current_op``, an optional look-ahead ``delayed_op``
    reservation, and a running ``cost`` counter.

    The ``delay`` flag enables look-ahead: an op may be loaded once its
    predecessor is merely *processing* (not just done), except for the very
    first op placed on a machine, which must wait for the predecessor to be
    fully done.
    """

    def __init__(self, machine_id, possible_ops, delay, verbose):
        self.machine_id = machine_id
        self.possible_ops = possible_ops
        self.remain_ops = possible_ops
        self.current_op = None
        self.delayed_op = None
        self.prev_op = None
        self.remaining_time = 0
        self.done_ops = []
        self.num_done_ops = 0
        self.cost = 0
        self.delay = delay
        self.verbose = verbose

    def __str__(self):
        return "Machine {}".format(self.machine_id)

    def available(self):
        """True if the machine can accept an op right now.

        Requires: some op is doable, no op is currently being processed, and
        the machine is not blocked waiting on a delayed (reserved) op.
        """
        future_work_exist_cond = bool(self.doable_ops())
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and not_wait_for_delayed_cond
        return ret

    def wait_for_delayed(self):
        """True if the machine is waiting for a delayed op whose predecessor is not done yet."""
        wait_for_delayed_cond = self.delayed_op is not None
        ret = wait_for_delayed_cond
        if wait_for_delayed_cond:
            delayed_op_ready_cond = self.delayed_op.prev_op.node_status == DONE_NODE_SIG
            ret = ret and not delayed_op_ready_cond
        return ret

    def doable_ops(self):
        """Return the subset of remaining ops that can run right now.

        An op is doable if it has no predecessor (first op of a job) or its
        predecessor satisfies the ready condition. With ``delay`` enabled,
        non-first ops on a machine may start while the predecessor is merely
        *processing* (look-ahead); the very first op placed on a machine must
        still wait for the predecessor to be *done*. Without ``delay`` the
        predecessor must always be done.
        """
        # doable_ops are subset of remain_ops.
        # some ops are doable when the prev_op is 'done' or 'processing' or 'start'
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG
                prev_process = op.prev_op.node_status == PROCESSING_NODE_SIG
                first_op = not bool(self.done_ops)
                if self.delay:
                    # each machine's first processing operation should not be a reserved operation
                    if first_op:
                        cond = prev_done
                    else:
                        cond = (prev_done or prev_process)
                else:
                    cond = prev_done

                if cond:
                    doable_ops.append(op)
                else:
                    pass

        return doable_ops

    @property
    def doable_ops_id(self):
        """Return the ids of currently doable ops."""
        doable_ops_id = []
        doable_ops = self.doable_ops()
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        """Doable ops *without* look-ahead: predecessor must be fully done.

        Used to compute the no-delay queue length (utilization cost).
        """
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == DONE_NODE_SIG
                if prev_done:
                    doable_ops.append(op)
        return doable_ops

    def work_done(self):
        """True when there are no remaining ops left to process."""
        return not self.remain_ops

    def load_op(self, t, op):
        """Start processing ``op`` on this machine at time ``t``.

        Performs double-checks (machine available, op processible, op is one of
        this machine's possible ops) and clears any matching delayed reservation,
        then flips the op into PROCESSING and removes it from the backlog.
        """

        # Procedures for double-checkings
        # If machine waits for the delayed job is done:
        if self.wait_for_delayed():
            raise RuntimeError("Machine {} waits for the delayed job {} but load {}".format(self.machine_id,
                                                                                  print(self.delayed_op), print(op)))

        # ignore input when the machine is not available
        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        # ignore when input op's previous op is not done yet:
        if not op.processible():
            raise RuntimeError("Operation {} is not processible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform ops {}{}".format(self.machine_id,
                                                                          op.job_id,
                                                                          op.step_id))

        # Essential condition for checking whether input is delayed
        # if delayed then, flush delayed_op attr
        if op == self.delayed_op:
            if self.verbose:
                print("[DELAYED OP LOADED] / MACHINE {} / {} / at {}".format(self.machine_id, op, t))
            self.delayed_op = None

        else:
            if self.verbose:
                print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        # Update operation's attributes
        op.node_status = PROCESSING_NODE_SIG
        op.remaining_time = op.processing_time
        op.start_time = t

        # Update machine's attributes
        self.current_op = op
        self.remaining_time = op.processing_time
        self.remain_ops.remove(self.current_op)

    def unload(self, t):
        """Mark the current op as done at time ``t`` and free the machine."""
        if self.verbose:
            print("[UNLOAD] / Machine {} / Op {} / t = {}".format(self.machine_id, self.current_op, t))
        self.current_op.node_status = DONE_NODE_SIG
        self.current_op.end_time = t
        self.done_ops.append(self.current_op)
        self.num_done_ops += 1
        self.prev_op = self.current_op
        self.current_op = None
        self.remaining_time = 0

    def do_processing(self, t):
        """Advance this machine's processing by one time unit at time ``t``.

        * If an op is being processed, decrement its remaining time and unload
          it when it reaches zero.
        * If the machine is instead holding a delayed op, accrue its delay time.
        * Waiting doable ops accumulate ``waiting_time`` each tick they wait.
        """
        if self.remaining_time > 0:  # When machine do some operation
            if self.current_op is not None:
                self.current_op.remaining_time -= 1
                if self.current_op.remaining_time <= 0:
                    if self.current_op.remaining_time < 0:
                        raise RuntimeWarning("Negative remaining time observed")
                    if self.verbose:
                        print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                    self.unload(t)
            # to compute idle_time reward, we need to count delayed_time
            elif self.delayed_op is not None:
                self.delayed_op.delayed_time += 1
                self.delayed_op.remaining_time -= 1

            doable_ops = self.doable_ops()
            if doable_ops:
                for op in doable_ops:
                    op.waiting_time += 1
            else:
                pass

            self.remaining_time -= 1

    def transit(self, t, a):
        """Apply decision ``a`` (an operation) to this machine at time ``t``.

        If ``a`` is processible now, load it. Otherwise it is a look-ahead
        reservation: mark it DELAYED and have the machine wait for it.
        """
        if self.available():  # Machine is ready to process.
            if a.processible():  # selected action is ready to be loaded right now.
                self.load_op(t, a)
            else:  # When input operation turns out to be 'delayed'
                a.node_status = DELAYED_NODE_SIG
                self.delayed_op = a
                # Reserve for the predecessor's remaining processing time plus this op's own time.
                self.delayed_op.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.current_op = None  # MACHINE is now waiting for delayed ops
                if self.verbose:
                    print("[DELAYED OP CHOSEN] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.delayed_op, t))
        else:
            raise RuntimeError("Access to not available machine")
