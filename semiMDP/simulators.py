"""The JSSP semi-Markov decision process (semi-MDP) simulator.

:class:`Simulator` ties together a :class:`~semiMDP.operationHelpers.JobManager`
(the operations/jobs) and a :class:`~semiMDP.machineHelpers.MachineManager`
(the machines) and exposes the standard MDP surface used by the RL training
and rollout loops:

* :meth:`Simulator.reset`
* :meth:`Simulator.transit` (apply an action = assign an operation to a machine)
* :meth:`Simulator.observe` (build the disjunctive-graph state + reward)
* :meth:`Simulator.process_one_time` / :meth:`Simulator.flush_trivial_ops`
  (advance time and auto-play forced moves)

Because the schedule evolves in real time while actions only fire at certain
decision points, this is a *semi*-MDP: :meth:`flush_trivial_ops` collapses the
stretches where every available machine has exactly one doable operation
(no real decision to make) into automatic transitions.

Two concrete simulators are provided: :class:`Simulator` (processing-time
graph) and :class:`NodeProcessingTimeSimulator` (complete-ratio graph with an
explicit end node per job).
"""

import random
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from semiMDP.jobShopSamplers import jssp_sampling
from semiMDP.operationHelpers import (JobManager,
                                      NodeProcessingTimeJobManager,
                                      get_edge_color_map,
                                      get_node_color_map)
from semiMDP.machineHelpers import (MachineManager,
                                    NodeProcessingTimeMachineManager)
from semiMDP.configs import (N_SEP, SEP, NEW)


class Simulator:
    """Job-shop scheduling semi-MDP built around the processing-time graph.

    Args:
        num_machines: number of machines (used only when sampling an instance).
        num_jobs: number of jobs (used only when sampling an instance).
        detach_done: if True, finished ops are removed from the observed graph.
        name: optional instance name (used for plotting / saving).
        machine_matrix / processing_time_matrix: if provided, use these
            instance matrices instead of sampling a random one.
        embedding_dim: forwarded to the job manager.
        use_surrogate_index: forward each op a flat integer id (see JobManager).
        delay: enable machine look-ahead reservations (see Machine).
        verbose: print machine events when True.

    Simulation procedure per tick: ``global_time += 1`` -> do_processing -> transit.
    """

    def __init__(self,
                 num_machines,
                 num_jobs,
                 detach_done=False,
                 name=None,
                 machine_matrix=None,
                 processing_time_matrix=None,
                 embedding_dim=16,
                 use_surrogate_index=True,
                 delay=False,
                 verbose=False):

        if machine_matrix is None or processing_time_matrix is None:
            ms, prts = self._sample_jssp_graph(num_machines, num_jobs)
            self.machine_matrix = ms.astype(int)
            self.processing_time_matrix = prts.astype(float)
        else:
            self.machine_matrix = machine_matrix.astype(int)
            self.processing_time_matrix = processing_time_matrix.astype(float)

        if name is None:
            self.name = '{} machine {} job'.format(num_machines, num_jobs)
        else:
            self.name = name

        self._machine_set = list(set(self.machine_matrix.flatten().tolist()))
        self.num_machine = len(self._machine_set)
        self.detach_done = detach_done
        self.embedding_dim = embedding_dim
        self.num_jobs = self.processing_time_matrix.shape[0]
        self.num_steps = self.processing_time_matrix.shape[1]
        self.use_surrogate_index = use_surrogate_index
        self.delay = delay
        self.verbose = verbose
        self.reset()
        # simulation procedure : global_time +=1 -> do_processing -> transit

    def reset(self):
        """Re-initialise the job and machine managers and zero the clock."""
        self.job_manager = JobManager(self.machine_matrix,
                                      self.processing_time_matrix,
                                      embedding_dim=self.embedding_dim,
                                      use_surrogate_index=self.use_surrogate_index)
        self.machine_manager = MachineManager(self.machine_matrix,
                                              self.job_manager,
                                              self.delay,
                                              self.verbose)
        self.global_time = 0  # -1 matters a lot

    def process_one_time(self):
        """Advance the simulation clock by one tick and process all machines."""
        self.global_time += 1
        self.machine_manager.do_processing(self.global_time)

    def transit(self, action=None):
        """Assign ``action`` (an operation) to its machine at the current time.

        Args:
            action: surrogate op id (when ``use_surrogate_index``) or a
                ``(job_id, step_id)`` pair; if ``None``, a random available
                machine and a random doable op on it are chosen.

        Returns:
            The surrogate op id of the chosen action when ``action is None``;
            otherwise ``None``.
        """
        if action is None:
            # Perform random action
            machine = random.choice(self.machine_manager.get_available_machines())
            op_id = random.choice(machine.doable_ops_id)
            job_id, step_id = self.job_manager.sur_index_dict[op_id]
            operation = self.job_manager[job_id][step_id]
            action = operation
            # print(machine)
            machine.transit(self.global_time, action)
            return op_id
        else:
            if self.use_surrogate_index:
                if action in self.job_manager.sur_index_dict.keys():
                    job_id, step_id = self.job_manager.sur_index_dict[action]
                else:
                    raise RuntimeError("Input action is not valid")
            else:
                job_id, step_id = action

            operation = self.job_manager[job_id][step_id]
            machine_id = operation.machine_id
            machine = self.machine_manager[machine_id]
            action = operation
            machine.transit(self.global_time, action)

    def flush_trivial_ops(self, reward='utilization', gamma=1.0):
        """Auto-execute forced moves until a real decision point is reached.

        A move is "trivial" when, for some available machine, exactly one
        operation is doable — there is nothing to choose, so the simulator
        executes it itself. This collapses the stretches of the semi-MDP
        between real decisions.

        Args:
            reward: reward type forwarded to :meth:`observe`.
            gamma: discount applied to the per-step reward accumulation.

        Returns:
            m_list: machine ids that still have a non-trivial choice pending.
            cum_reward: discounted sum of rewards gathered during flushing.
            done: whether all jobs finished during flushing.
            sub_list: list of op ids that were auto-executed.
        """
        done = False
        cum_reward = 0
        t = 0
        # print(random.random())
        sub_list = []
        while True:
            # print('in the while')
            t += 1
            m_list = []
            do_op_dict = self.get_doable_ops_in_dict()
            # print(do_op_dict)
            all_machine_work = False if bool(do_op_dict) else True

            if all_machine_work:  # all machines are on processing. keep process!
                self.process_one_time()
            else:  # some machine has a possibly trivial action. the others not.
                # load trivial ops to the machines
                num_ops_counter = 1
                for m_id, op_ids in do_op_dict.items():
                    num_ops = len(op_ids)
                    if num_ops == 1:
                        # print(op_ids[0])
                        sub_list.append(op_ids[0])
                        self.transit(op_ids[0])  # load trivial action
                        g, r, _ = self.observe(reward)
                        cum_reward = r + gamma*cum_reward
                    else:
                        m_list.append(m_id)
                        num_ops_counter *= num_ops

                # not-all-trivial: at least one machine had >1 doable op, so a
                # real decision is needed -> stop flushing and let the agent act.
                if num_ops_counter != 1:
                    break

            # if simulation is done
            jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
            # print(jobs_done)
            done = True if np.prod(jobs_done) == 1 else False

            if done:
                # print('done')
                break
        # print(t)
        return m_list, cum_reward, done, sub_list

    def get_available_machines(self, shuffle_machine=True):
        """Return machines that currently have doable, unreserved work."""
        return self.machine_manager.get_available_machines(shuffle_machine)

    def get_doable_ops_in_dict(self, machine_id=None, shuffle_machine=True):
        """Return currently doable ops, per machine.

        Args:
            machine_id: if given, return only that machine's doable ops;
                otherwise return a ``{machine_id: [op_id, ...]}`` dict.
            shuffle_machine: whether to randomise machine iteration order.

        Returns:
            A dict (all machines) or list (single machine) of doable op ids.
        """
        if machine_id is None:
            doable_dict = {}
            if self.get_available_machines():
                for m in self.get_available_machines(shuffle_machine):
                    _id = m.machine_id
                    _ops = m.doable_ops_id
                    doable_dict[_id] = _ops
            ret = doable_dict
        else:
            available_machines = [m.machine_id for m in self.get_available_machines()]
            if machine_id in available_machines:
                ret = self.machine_manager[machine_id].doable_ops_id
            else:
                raise RuntimeWarning("Access to the not available machine {}. Return is None".format(machine_id))
        return ret

    def get_doable_ops_in_list(self, machine_id=None, shuffle_machine=True):
        """Return all currently doable ops (across machines) as a flat list."""
        doable_dict = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        do_ops = []
        for _, v in doable_dict.items():
            do_ops += v
        return do_ops

    def get_doable_ops(self, machine_id=None, return_list=False, shuffle_machine=True):
        """Convenience accessor: return doable ops as a dict (default) or list."""
        if return_list:
            ret = self.get_doable_ops_in_list(machine_id, shuffle_machine)
        else:
            ret = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        return ret

    def observe(self, reward='utilization', return_doable=True):
        """Observe the current state: returns the job-shop graph, reward, done flag.

        A simple wrapper for JobManager's observe function that additionally
        computes the per-step reward and the done flag, and optionally annotates
        each node with whether it is currently doable (and on which machine).

        Reward types:
            * ``'makespan'``: ``-global_time`` once all jobs are done, else 0.
            * ``'utilization'``: minus the total no-delay queue length across machines.
            * ``'idle_time'``: minus the idle-machine fraction.

        Returns:
            ``(g, r, done)``: networkx disjunctive graph, scalar reward, done flag.

        Args:
            reward: which reward shaping to use (see above).
            return_doable: if True, tag each node with ``doable`` (bool) and
                ``machine`` (machine id, 0 when not doable) attributes.
        """

        jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
        # check jobs_done contains only True or False
        if np.prod(jobs_done) == 1:
            done = True
        else:
            done = False
        if reward == 'makespan':
            if done:
                r = -self.global_time
            else:
                r = 0
        # return reward as total sum of queues for all machines
        elif reward == 'utilization':
            t_cost = self.machine_manager.cal_total_cost()
            r = -t_cost

        elif reward == 'idle_time':
            r = -float(len(self.machine_manager.get_idle_machines()))/float(self.num_machine)

        g = self.job_manager.observe(detach_done=self.detach_done)

        if return_doable:
            # Annotate each node with whether it is currently doable and,
            # if so, the machine id it would run on. Non-doable nodes get machine 0.
            if self.use_surrogate_index:
                do_ops_list = self.get_doable_ops(return_list=True)
                for n in g.nodes:
                    if n in do_ops_list:
                        job_id, op_id = self.job_manager.sur_index_dict[n]
                        m_id = self.job_manager[job_id][op_id].machine_id
                        g.nodes[n]['doable'] = True
                        g.nodes[n]['machine'] = m_id
                    else:
                        g.nodes[n]['doable'] = False
                        g.nodes[n]['machine'] = 0

        return g, r, done

    def plot_graph(self, draw=True,
                   node_type_color_dict=None,
                   edge_type_color_dict=None,
                   half_width=None,
                   half_height=None,
                   **kwargs):
        """Render the current job-shop graph with networkx (see JobManager.plot_graph)."""

        g = self.job_manager.observe(self.detach_done)
        node_colors = get_node_color_map(g, node_type_color_dict)
        edge_colors = get_edge_color_map(g, edge_type_color_dict)

        if half_width is None:
            half_width = 30
        if half_height is None:
            half_height = 10

        num_horizontals = self.num_steps + 1
        num_verticals = self.num_jobs + 1

        def xidx2coord(x):
            return np.linspace(-half_width, half_width, num_horizontals)[x]

        def yidx2coord(y):
            return np.linspace(half_height, -half_height, num_verticals)[y]

        # Place nodes on a grid by their (job, step) index (surrogate id or tuple id).
        pos_dict = OrderedDict()
        for n in g.nodes:
            if self.use_surrogate_index:
                y, x = self.job_manager.sur_index_dict[n]
                pos_dict[n] = np.array((xidx2coord(x), yidx2coord(y)))
            else:
                pos_dict[n] = np.array((xidx2coord(n[1]), yidx2coord(n[0])))

        if kwargs is None:
            kwargs['figsize'] = (10, 5)
            kwargs['dpi'] = 300

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1, 1, 1)

        nx.draw(g, pos_dict,
                node_color=node_colors,
                edge_color=edge_colors,
                with_labels=True,
                ax=ax)
        if draw:
            plt.show()
        else:
            return fig, ax

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        # Draw a gantt chart
        self.job_manager.draw_gantt_chart(path, benchmark_name, max_x)

    @staticmethod
    def _sample_jssp_graph(m, n):
        """Sample a random instance, after rounding ``m``/``n`` to ``N_SEP`` units.

        With ``N_SEP == 1`` the rounding is a no-op; the guards keep both
        dimensions at least ``N_SEP``. Requires ``m <= n``.
        """
        if not m % N_SEP == 0:
            m = int(N_SEP * (m // N_SEP))
            if m < N_SEP:
                m = N_SEP
        if not n % N_SEP == 0:
            n = int(N_SEP * (n // N_SEP))
            if n < N_SEP:
                n = N_SEP
        if m > n:
            raise RuntimeError(" m should be smaller or equal to n ")

        return jssp_sampling(m, n, 5, 100)
        # return jssp_sampling(m, n, 1, 5)

    @classmethod
    def from_path(cls, jssp_path, **kwargs):
        """Build a simulator from a standard JSSP instance file.

        Each line is a job; fields alternate ``machine_id processing_time``.
        """
        with open(jssp_path) as f:
            ms = []  # machines
            prts = []  # processing times
            for l in f:
                l_split = " ".join(l.split()).split(' ')
                m = l_split[0::2]
                prt = l_split[1::2]
                ms.append(np.array(m, dtype=int))
                prts.append(np.array(prt, dtype=float))

        ms = np.stack(ms)
        prts = np.stack(prts)
        num_job, num_machine = ms.shape
        name = jssp_path.split('/')[-1].replace('.txt', '')

        return cls(num_machines=num_machine,
                   num_jobs=num_job,
                   name=name,
                   machine_matrix=ms,
                   processing_time_matrix=prts,
                   **kwargs)

    @classmethod
    def from_TA_path(cls, pt_path, m_path, **kwargs):
        """Build a simulator from a TA-format instance (separate PT and machine files).

        Fields are separated by :data:`SEP` (whitespace). The trailing field
        of each line may carry a :data:`NEW` marker, which is stripped. Machine
        ids are 1-based in the file, so ``1`` is subtracted to make them 0-based.
        """
        with open(pt_path) as f1:
            prts = []
            for l in f1:
                l_split = l.split(SEP)
                prt = [e for e in l_split if e != '']
                if NEW in prt[-1]:
                    prt[-1] = prt[-1].split(NEW)[0]
                prts.append(np.array(prt, dtype=float))

        with open(m_path) as f2:
            ms = []
            for l in f2:
                l_split = l.split(SEP)
                m = [e for e in l_split if e != '']
                if NEW in m[-1]:
                    m[-1] = m[-1].split(NEW)[0]
                ms.append(np.array(m, dtype=int))

        ms = np.stack(ms)-1  # TA-format machine ids are 1-based -> make 0-based
        prts = np.stack(prts)
        num_job, num_machine = ms.shape
        name = pt_path.split('/')[-1].replace('_PT.txt', '')

        return cls(num_machines=num_machine,
                   num_jobs=num_job,
                   name=name,
                   machine_matrix=ms,
                   processing_time_matrix=prts,
                   **kwargs)


class NodeProcessingTimeSimulator(Simulator):
    """:class:`Simulator` variant using the node-processing-time graph.

    Swaps the job/machine managers for their NodeProcessingTime counterparts
    (complete-ratio-weighted conjunctive edges plus an explicit end node).
    Only :meth:`reset` is overridden to wire those managers on (re)initialisation.
    """

    def reset(self):
        self.job_manager = NodeProcessingTimeJobManager(self.machine_matrix,
                                                        self.processing_time_matrix,
                                                        embedding_dim=self.embedding_dim,
                                                        use_surrogate_index=self.use_surrogate_index)
        self.machine_manager = NodeProcessingTimeMachineManager(self.machine_matrix,
                                                                self.job_manager,
                                                                self.delay,
                                                                self.verbose)
        self.global_time = 0  # -1 matters a lot
