"""Operation- and job-level data structures for the JSSP environment.

This module models the *objects* that make up a schedule — jobs, their
operations, and the dummy start/end markers — together with the
:class:`JobManager` that owns them and builds the disjunctive-graph view the
GNN consumes.

Representation notes
--------------------
* A job is a chain of operations performed in a fixed order on different
  machines (conjunctive / precedence order). Operations of different jobs
  that share a machine are linked by disjunctive (conflict) edges.
* Each operation carries a node "signature" stored as ``node_status`` (see
  :mod:`semiMDP.configs`). The :prop:`Operation.x` property turns that status
  plus bookkeeping fields into the per-node feature dict attached to the
  networkx graph.
* Operations are keyed by their natural ``(job_id, step_id)`` id by default.
  When ``use_surrogate_index`` is set, a flat integer ``sur_id`` is also
  assigned per operation so the GNN can index nodes with a contiguous range.

Two flavours are provided:

* The base :class:`Job` / :class:`Operation` build a *processing-time* graph
  whose conjunctive edges are weighted by processing time.
* :class:`NodeProcessingTimeJob` / :class:`NodeProcessingTimeOperation`
  instead weight conjunctive edges by ``complete_ratio`` deltas and append an
  explicit :class:`EndOperation` terminator per job.
"""

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from collections import OrderedDict

from plotly.offline import plot

from semiMDP.configs import (NOT_START_NODE_SIG,
                             PROCESSING_NODE_SIG,
                             DONE_NODE_SIG,
                             DELAYED_NODE_SIG,
                             DUMMY_NODE_SIG,
                             CONJUNCTIVE_TYPE,
                             DISJUNCTIVE_TYPE,
                             FORWARD,
                             BACKWARD)


def get_edge_color_map(g, edge_type_color_dict=None):
    """Return a list of colors, one per edge of ``g``, keyed by edge type.

    Args:
        g: networkx graph whose edges carry a ``type`` attribute.
        edge_type_color_dict: optional mapping from edge type to a matplotlib
            color. Defaults to black for conjunctive edges and light-coral
            (#F08080) for disjunctive edges.

    Returns:
        List of edge colors in ``g.edges`` iteration order.
    """
    if edge_type_color_dict is None:
        edge_type_color_dict = OrderedDict()
        edge_type_color_dict[CONJUNCTIVE_TYPE] = 'k'
        edge_type_color_dict[DISJUNCTIVE_TYPE] = '#F08080'

    colors = []
    for e in g.edges:
        edge_type = g.edges[e]['type']
        colors.append(edge_type_color_dict[edge_type])
    return colors


def calc_positions(g, half_width=None, half_height=None):
    """Lay graph nodes out on a grid for plotting.

    The horizontal axis corresponds to the operation step index and the
    vertical axis to the job index, sampled across
    ``[-half_width, half_width]`` / ``[-half_height, half_height]``.

    Returns:
        ``OrderedDict`` mapping each node id ``(job_id, step_id)`` to a
        ``(x, y)`` numpy coordinate.
    """
    if half_width is None:
        half_width = 30
    if half_height is None:
        half_height = 10

    min_idx = min(g.nodes)
    max_idx = max(g.nodes)

    num_horizontals = max_idx[1] - min_idx[1] + 1
    # NB: keeps the original axis-offset behaviour (mixes min_idx[1]); preserved as-is.
    num_verticals = max_idx[0] - min_idx[1] + 1

    def xidx2coord(x):
        return np.linspace(-half_width, half_width, num_horizontals)[x]

    def yidx2coord(y):
        return np.linspace(-half_height, half_height, num_verticals)[y]

    pos_dict = OrderedDict()
    for n in g.nodes:
        pos_dict[n] = np.array((xidx2coord(n[1]), yidx2coord(n[0])))
    return pos_dict


def get_node_color_map(g, node_type_color_dict=None):
    """Return a list of colors, one per node of ``g``, keyed by node type.

    Defaults: khaki (not-start), green (processing), blue (delayed),
    light-grey (done), white (dummy).
    """
    if node_type_color_dict is None:
        node_type_color_dict = OrderedDict()
        node_type_color_dict[NOT_START_NODE_SIG] = '#F0E68C'
        node_type_color_dict[PROCESSING_NODE_SIG] = '#ADFF2F'
        node_type_color_dict[DELAYED_NODE_SIG] = '#829DC9'
        node_type_color_dict[DONE_NODE_SIG] = '#E9E9E9'
        node_type_color_dict[DUMMY_NODE_SIG] = '#FFFFFF'

    colors = []
    for n in g.nodes:
        node_type = g.nodes[n]['type']
        colors.append(node_type_color_dict[node_type])
    return colors


class JobManager:
    """Owns all jobs/operations of an instance and builds the disjunctive graph.

    Args:
        machine_matrix: ``(num_jobs, num_machines)`` int array; entry
            ``[j, s]`` is the (0-indexed) machine that job ``j`` uses at step ``s``.
        processing_time_matrix: matching array of processing times.
        embedding_dim: reserved embedding dimension forwarded to jobs/ops.
        use_surrogate_index: if True, assign each op a flat integer ``sur_id``
            and populate :attr:`sur_index_dict` for ``sur_id -> (job_id, step_id)``.
    """

    def __init__(self,
                 machine_matrix,
                 processing_time_matrix,
                 embedding_dim=16,
                 use_surrogate_index=True):

        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive (precedence) chains: one Job per row.
        # +1 so machine ids start from 1 (the simulator's convention).
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = Job(job_i, m, pr_t, embedding_dim)

        # Constructing disjunctive edges: link ops of different jobs that share a machine.
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops

        self.use_surrogate_index = use_surrogate_index

        if self.use_surrogate_index:
            # Assign a contiguous integer id to every op and record the
            # sur_id -> (job_id, step_id) lookup used elsewhere for indexing.
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def __call__(self, index):
        """Return the job with id ``index``."""
        return self.jobs[index]

    def __getitem__(self, index):
        """Return the job with id ``index``."""
        return self.jobs[index]

    def observe(self, detach_done=True):
        """Build and return the current disjunctive job-shop graph.

        Each operation becomes a node carrying its feature dict (``op.x``).
        Edges added per operation:

        * forward conjunctive edge ``op -> next_op`` (weighted by ``processing_time``)
        * backward conjunctive edge ``op -> prev_op`` (negatively weighted)
        * disjunctive edges ``op -> disj_op`` for every same-machine op

        Args:
            detach_done: if True, finished operations (and edges incident to
                them) are omitted from the graph so they no longer influence
                message passing.

        Returns:
            The current time-stamp job-shop graph (an :class:`nx.OrderedDiGraph`).
        """

        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not (op == job.ops[-1])

                done_cond = op.x['type'] == DONE_NODE_SIG

                if detach_done:
                    if not done_cond:
                        g.add_node(op.id, **op.x)
                        if not_end_cond:  # Construct forward-flow conjunctive edges only
                            g.add_edge(op.id, op.next_op.id,
                                       processing_time=op.processing_time,
                                       type=CONJUNCTIVE_TYPE,
                                       direction=FORWARD)

                        if not_start_cond:  # Construct backward-flow conjunctive edges only
                            # Skip the backward edge when the predecessor is already done
                            # (done nodes are detached), so the graph stays consistent.
                            if op.prev_op.x['type'] != DONE_NODE_SIG:
                                g.add_edge(op.id, op.prev_op.id,
                                           processing_time=-1 * op.prev_op.processing_time,
                                           type=CONJUNCTIVE_TYPE,
                                           direction=BACKWARD)

                        for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                            if disj_op.x['type'] != DONE_NODE_SIG:
                                g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

                else:
                    g.add_node(op.id, **op.x)
                    if not_end_cond:  # Construct forward-flow conjunctive edges only
                        g.add_edge(op.id, op.next_op.id,
                                   processing_time=op.processing_time,
                                   type=CONJUNCTIVE_TYPE,
                                   direction=FORWARD)

                    if not_start_cond:  # Construct backward-flow conjunctive edges only
                        g.add_edge(op.id, op.prev_op.id,
                                   processing_time=-1 * op.prev_op.processing_time,
                                   type=CONJUNCTIVE_TYPE,
                                   direction=BACKWARD)

                    for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                        g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

        return g

    def plot_graph(self, draw=True,
                   node_type_color_dict=None,
                   edge_type_color_dict=None,
                   half_width=None,
                   half_height=None,
                   **kwargs):
        """Render the current job-shop graph with networkx.

        Args:
            draw: if True, ``plt.show()`` the figure; otherwise return it.
            Returns:
                ``None`` if ``draw`` is True, else ``(fig, ax)``.
        """

        g = self.observe()
        node_colors = get_node_color_map(g, node_type_color_dict)
        edge_colors = get_edge_color_map(g, edge_type_color_dict)
        pos = calc_positions(g, half_width, half_height)

        if kwargs is None:
            kwargs = {'figsize': (10, 5), 'dpi': 300}

        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(1, 1, 1)

        nx.draw(g, pos,
                node_color=node_colors,
                edge_color=edge_colors,
                with_labels=True,
                ax=ax)
        if draw:
            plt.show()
        else:
            return fig, ax

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        """Render a Plotly Gantt chart of the completed schedule to ``path``.

        Args:
            path: output HTML file for the Plotly figure.
            benchmark_name: title prefix for the chart.
            max_x: fixed upper bound of the time axis.
        """
        gantt_info = []
        for _, job in self.jobs.items():
            for op in job.ops:
                if not isinstance(op, DummyOperation):
                    temp = OrderedDict()
                    temp['Task'] = "Machine" + str(op.machine_id)
                    temp['Start'] = op.start_time
                    temp['Finish'] = op.end_time
                    temp['Resource'] = "Job" + str(op.job_id)
                    gantt_info.append(temp)
        gantt_info = sorted(gantt_info, key=lambda k: k['Task'])
        # Assign a random color per job (Resource) for visual distinction.
        color = OrderedDict()
        for g in gantt_info:
            _r = random.randrange(0, 255, 1)
            _g = random.randrange(0, 255, 1)
            _b = random.randrange(0, 255, 1)
            rgb = 'rgb({}, {}, {})'.format(_r, _g, _b)
            color[g['Resource']] = rgb
        fig = ff.create_gantt(gantt_info, colors=color, show_colorbar=True, group_tasks=True, index_col='Resource',
                              title=benchmark_name + ' gantt chart', showgrid_x=True, showgrid_y=True)
        fig['layout']['xaxis'].update({'type': None})
        fig['layout']['xaxis'].update({'range': [0, max_x]})
        fig['layout']['xaxis'].update({'title': 'time'})

        plot(fig, filename=path)


class NodeProcessingTimeJobManager(JobManager):
    """:class:`JobManager` variant for the node-processing-time graph.

    Uses :class:`NodeProcessingTimeJob` / :class:`NodeProcessingTimeOperation`,
    whose conjunctive edges carry ``complete_ratio`` deltas (not raw
    processing-time deltas) and that append a dummy :class:`EndOperation`.
    """

    def __init__(self, machine_matrix, processing_time_matrix, embedding_dim=16, use_surrogate_index=True):
        super().__init__(machine_matrix, processing_time_matrix, embedding_dim, use_surrogate_index)
        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = NodeProcessingTimeJob(job_i, m, pr_t, embedding_dim)

        # Constructing disjunctive edges
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops

        self.use_surrogate_index = use_surrogate_index

        if self.use_surrogate_index:
            # Constructing surrogate indices
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def observe(self, detach_done=True):
        """Build the current job-shop graph (node-processing-time flavour).

        Like :meth:`JobManager.observe` but conjunctive edges are weighted by
        ``complete_ratio`` deltas and the job terminator is the explicit
        :class:`EndOperation` (detected via isinstance instead of ``ops[-1]``).
        """

        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not isinstance(op, EndOperation)

                done_cond = op.x['type'] == DONE_NODE_SIG

                if detach_done:
                    if not done_cond:
                        g.add_node(op.id, **op.x)
                        if not_end_cond:  # Construct forward-flow conjunctive edges only
                            g.add_edge(op.id, op.next_op.id,
                                       distance=(op.next_op.complete_ratio-op.complete_ratio),
                                       type=CONJUNCTIVE_TYPE,
                                       direction=FORWARD)
                        if not_start_cond:  # Construct backward-flow conjunctive edges only
                            g.add_edge(op.id, op.prev_op.id,
                                       distance=-(op.complete_ratio - op.prev_op.complete_ratio),
                                       type=CONJUNCTIVE_TYPE,
                                       direction=BACKWARD)

                        for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                            g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)

                else:
                    g.add_node(op.id, **op.x)
                    if not_end_cond:  # Construct forward-flow conjunctive edges only
                        g.add_edge(op.id, op.next_op.id,
                                   distance=(op.next_op.complete_ratio - op.complete_ratio),
                                   type=CONJUNCTIVE_TYPE,
                                   direction=FORWARD)

                    if not_start_cond:  # Construct backward-flow conjunctive edges only
                        g.add_edge(op.id, op.prev_op.id,
                                   distance=-(op.complete_ratio - op.prev_op.complete_ratio),
                                   type=CONJUNCTIVE_TYPE,
                                   direction=BACKWARD)

                    for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                        g.add_edge(op.id, disj_op.id, type=DISJUNCTIVE_TYPE)
        return g


class Job:
    """A single job: an ordered chain of :class:`Operation` objects.

    Builds the linked ``prev_op``/``next_op`` pointers between successive
    operations and records each operation's ``complete_ratio`` (cumulative
    processing time up to and including this op, divided by the job's total
    processing time).

    Note: this base flavour has no explicit end node; the last real operation
    doubles as the job terminator.
    """

    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim):
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        self.num_sequence = processing_time_order.size
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            cum_pr_t += pr_t
            op = Operation(job_id=job_id, step_id=step_id, machine_id=m_id,
                           prev_op=None,
                           processing_time=pr_t,
                           complete_ratio=cum_pr_t/self.processing_time,
                           job=self)
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]


        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i+1]

    def __getitem__(self, index):
        """Return the operation at position ``index`` in this job."""
        return self.ops[index]

    # To check job is done or not using last operation's node status
    @property
    def job_done(self):
        """True once the job's last operation is done."""
        if self.ops[-1].node_status == DONE_NODE_SIG:
            return True
        else:
            return False

    # To check the number of remaining operations
    @property
    def remaining_ops(self):
        """Count of operations not yet done."""
        c = 0
        for op in self.ops:
            if op.node_status != DONE_NODE_SIG:
                c += 1
        return c


class NodeProcessingTimeJob(Job):
    """Job variant that appends a dummy :class:`NodeProcessingTimeEndOperation`.

    Used together with the node-processing-time graph flavour, whose
    conjunctive edges carry ``complete_ratio`` deltas. The appended end op
    becomes step ``len(ops)`` and carries ``complete_ratio`` of 1.0.
    """

    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim):
        super().__init__(job_id, machine_order, processing_time_order, embedding_dim)
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            op = NodeProcessingTimeOperation(job_id=job_id,
                                             step_id=step_id,
                                             machine_id=m_id,
                                             prev_op=None,
                                             processing_time=pr_t,
                                             complete_ratio=cum_pr_t / self.processing_time,
                                             job=self)
            cum_pr_t += pr_t
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # instantiate DUMMY END node and link it as the successor of the last real op
        _prev_op = self.ops[-1]
        self.ops.append(NodeProcessingTimeEndOperation(job_id=job_id,
                                                       step_id=_prev_op.step_id + 1,
                                                       embedding_dim=embedding_dim))
        self.ops[-1].prev_op = _prev_op
        self.num_sequence = len(self.ops) - 1  # end node is not a real operation

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i + 1]


class DummyOperation:
    """Marker node carrying no real work (used as start/end sentinels)."""

    def __init__(self,
                 job_id,
                 step_id,
                 embedding_dim):
        self.job_id = job_id
        self.step_id = step_id
        self._id = (job_id, step_id)
        self.machine_id = 'NA'
        self.processing_time = 0
        self.embedding_dim = embedding_dim
        self.built = False
        self.type = DUMMY_NODE_SIG
        self._x = {'type': self.type}
        self.node_status = DUMMY_NODE_SIG
        self.remaining_time = 0

    @property
    def id(self):
        """Return ``sur_id`` when surrogate indexing is enabled, else ``_id``."""
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id


class StartOperation(DummyOperation):
    """Synthetic "source" node prepended to a job (no predecessor).

    ``complete_ratio`` is 0.0; setting its ``next_op`` marks the node as built.
    """

    def __init__(self, job_id, embedding_dim):
        super().__init__(job_id=job_id, step_id=-1, embedding_dim=embedding_dim)
        self.complete_ratio = 0.0
        self._next_op = None

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, op):
        self._next_op = op
        self.built = True

    @property
    def x(self):
        """Feature dict for the start node (type + zero complete ratio)."""
        ret = self._x
        ret['complete_ratio'] = self.complete_ratio
        return ret


class EndOperation(DummyOperation):
    """Synthetic "sink" node appended to a job (``complete_ratio`` = 1.0).

    Setting its ``prev_op`` marks the node as built.
    """

    def __init__(self, job_id, step_id, embedding_dim):
        super().__init__(job_id=job_id, step_id=step_id, embedding_dim=embedding_dim)
        self.remaining_time = -1.0
        self.complete_ratio = 1.0
        self._prev_op = None

    @property
    def prev_op(self):
        return self._prev_op

    @prev_op.setter
    def prev_op(self, op):
        self._prev_op = op
        self.built = True

    @property
    def x(self):
        """Feature dict for the end node (type + complete ratio + remain time)."""
        ret = self._x
        ret['complete_ratio'] = self.complete_ratio
        ret['remain_time'] = self.remaining_time
        return ret


class NodeProcessingTimeEndOperation(EndOperation):
    """End node flavour that also exposes processing_time in its features."""

    @property
    def x(self):
        """Feature dict (type + processing time + remain time)."""
        ret = self._x
        ret['processing_time'] = self.processing_time
        ret['remain_time'] = self.remaining_time
        return ret


class Operation:
    """A single schedulable operation of a job on a specific machine.

    Tracks its lifecycle via ``node_status`` and exposes the node-feature dict
    through :prop:`x`. Held by a :class:`Job` and linked to its predecessor
    (``prev_op``), successor (``next_op``) and same-machine neighbours
    (``disjunctive_ops``). The ``built`` flag latches True once both the
    successor and disjunctive neighbours have been wired up.
    """

    def __init__(self,
                 job_id,
                 step_id,
                 machine_id,
                 complete_ratio,
                 prev_op,
                 processing_time,
                 job,
                 next_op=None,
                 disjunctive_ops=None):

        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.delayed_time = 0
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        # Operations remaining in this job after the current step (step_id is 0-based).
        self.remaining_ops = self.job.num_sequence - (self.step_id + 1)
        self.waiting_time = 0
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    def __str__(self):
        return "job {} step {}".format(self.job_id, self.step_id)

    def processible(self):
        """True if the op has no predecessor or its predecessor is done.

        (I.e. the operation can be loaded onto its machine right now.)
        """
        prev_none = self.prev_op is None
        if self.prev_op is not None:
            prev_done = self.prev_op.node_status is DONE_NODE_SIG
        else:
            prev_done = False
        return prev_done or prev_none

    @property
    def id(self):
        """``sur_id`` when surrogate indexing is enabled, else ``(job_id, step_id)``."""
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id

    @property
    def disjunctive_ops(self):
        """Same-machine neighbour operations of this op."""
        return self._disjunctive_ops

    @disjunctive_ops.setter
    def disjunctive_ops(self, disj_ops):
        for ops in disj_ops:
            if not isinstance(ops, Operation):
                raise RuntimeError("Given {} is not Operation instance".format(ops))
        self._disjunctive_ops = disj_ops
        self.disjunctive_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def next_op(self):
        """The successor operation of this op along the job."""
        return self._next_op

    @next_op.setter
    def next_op(self, next_op):
        self._next_op = next_op
        self.next_op_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def x(self):  # return node attribute
        """Build the per-node feature dict depending on lifecycle state.

        All states expose: id, type, complete_ratio, processing_time,
        remaining_ops. ``waiting_time`` and ``remain_time`` vary by state:
        ``remain_time`` is -1 while waiting, the live remaining processing time
        while PROCESSING, and -1 again once DONE.
        """
        not_start_cond = (self.node_status == NOT_START_NODE_SIG)
        delayed_cond = (self.node_status == DELAYED_NODE_SIG)
        processing_cond = (self.node_status == PROCESSING_NODE_SIG)
        done_cond = (self.node_status == DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = self.waiting_time
            _x["remain_time"] = -1
        elif processing_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = 0
            _x["remain_time"] = self.remaining_time
        elif done_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = 0
            _x["remain_time"] = -1
        elif delayed_cond:
            raise NotImplementedError("delayed operation")
        else:
            raise RuntimeError("Not supporting node type")
        return _x


class NodeProcessingTimeOperation(Operation):
    """Operation variant for the node-processing-time graph.

    Tracks explicit ``start_time`` / ``end_time`` (for Gantt charts) and
    exposes a slimmer ``x`` feature dict carrying processing_time, type and
    remain_time.
    """

    def __init__(self, job_id, step_id, machine_id, complete_ratio, prev_op, processing_time, job, next_op=None,
                 disjunctive_ops=None):

        super().__init__(job_id, step_id, machine_id, complete_ratio, prev_op, processing_time, job, next_op,
                         disjunctive_ops)
        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        # Time bookkeeping filled in by the machine when this op is loaded/unloaded.
        self.start_time = None
        self.end_time = None

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    @property
    def x(self):  # return node attribute
        """Per-node feature dict for this flavour (processing_time/type/remain_time)."""
        not_start_cond = (self.node_status == NOT_START_NODE_SIG)
        delayed_cond = (self.node_status == DELAYED_NODE_SIG)
        processing_cond = (self.node_status == PROCESSING_NODE_SIG)
        done_cond = (self.node_status == DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x["processing_time"] = self.processing_time
            _x["type"] = self.node_status
            _x["remain_time"] = -1
        elif processing_cond or done_cond or delayed_cond:
            _x = OrderedDict()
            _x["processing_time"] = self.processing_time
            _x["type"] = self.node_status
            _x["remain_time"] = self.remaining_time
        else:
            raise RuntimeError("Not supporting node type")
        return _x
