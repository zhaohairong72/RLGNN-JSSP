"""Graph neural network for learning job-shop scheduling policies.

Implements (a re-implementation of) the RL-GNN model from Park et al., 2021:

* :func:`to_pyg` converts the networkx disjunctive graph produced by the
  simulator into three PyTorch-Geometric (PyG) graphs that split the edges by
  direction/type — predecessor (``pre``), successor (``suc``), and disjunctive
  (``dis``) — each sharing the same node features.
* :class:`RLGNN` is the GNN embedding backbone: a stack of
  :class:`RLGNNLayer`s that run message passing along each of the three edge
  sets and merge the results into updated node embeddings.
* :class:`PolicyNet` maps node embeddings to a distribution over feasible
  operations and samples an action (plus its log-probability).
* :class:`CriticNet` maps the pooled graph embedding to a scalar value.

All networks share the lightweight :class:`MLP` building block.
"""

from semiMDP.simulators import Simulator
import torch
import random
import numpy as np
import networkx as nx
from torch.nn.functional import relu
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch.distributions.categorical import Categorical


def count_parameters(model, verbose=False, print_model=False):
    """Print the total number of trainable parameters of ``model``.

    Args:
        model: a ``torch.nn.Module``.
        verbose: if True, also print every trainable parameter tensor.
        print_model: if True, print the model's structure first.
    """
    if print_model:
        print('Model:', model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    print('The model has {} parameters'.format(pytorch_total_params))


def to_pyg(g, dev):
    """Convert the simulator's networkx graph into three PyG ``Data`` graphs.

    The node feature matrix ``x`` concatenates, per node:
      1. a 3-dim one-hot of the node ``type``,
      2. ``processing_time`` (scalar),
      3. ``complete_ratio`` (scalar),
      4. ``remaining_ops`` (scalar),
      5. ``waiting_time`` (scalar),
      6. ``remain_time`` (scalar).

    Finished operations (``type == DONE_NODE_SIG``) are zeroed out so they no
    longer carry signal. The edges are then split into three disjoint graphs:
      * ``pre`` — forward conjunctive edges (intra-job, step s -> s+1),
      * ``suc`` — backward conjunctive edges (intra-job, step s -> s-1),
      * ``dis`` — disjunctive edges (cross-job, same machine),
    and each is returned as a ``Data(x, edge_index)`` object moved to ``dev``.
    """

    x = []
    one_hot = np.eye(3, dtype=np.float32)[np.fromiter(nx.get_node_attributes(g, 'type').values(), dtype=np.int32)]
    x.append(one_hot)
    x.append(np.fromiter(nx.get_node_attributes(g, 'processing_time').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'complete_ratio').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'remaining_ops').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'waiting_time').values(), dtype=np.float32).reshape(-1, 1))
    x.append(np.fromiter(nx.get_node_attributes(g, 'remain_time').values(), dtype=np.float32).reshape(-1, 1))
    x = np.concatenate(x, axis=1)
    x = torch.from_numpy(x)

    for n in g.nodes:
        if g.nodes[n]['type'] == 1:
            x[n] = 0  # finished op has feature 0

    # Build per-edge-type adjacency matrices from node ids ``id = (job, step)``.
    adj_pre = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_suc = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    adj_dis = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
    for e in g.edges:
        s, t = e
        if g.nodes[s]['id'][0] == g.nodes[t]['id'][0]:  # conjunctive edge (same job)
            if g.nodes[s]['id'][1] < g.nodes[t]['id'][1]:  # forward
                adj_pre[s, t] = 1
            else:  # backward
                adj_suc[s, t] = 1
        else:  # disjunctive edge
            adj_dis[s, t] = 1
    edge_index_pre = torch.nonzero(torch.from_numpy(adj_pre)).t().contiguous()
    edge_index_suc = torch.nonzero(torch.from_numpy(adj_suc)).t().contiguous()
    edge_index_dis = torch.nonzero(torch.from_numpy(adj_dis)).t().contiguous()

    g_pre = Data(x=x, edge_index=edge_index_pre).to(dev)
    g_suc = Data(x=x, edge_index=edge_index_suc).to(dev)
    g_dis = Data(x=x, edge_index=edge_index_dis).to(dev)

    return g_pre, g_suc, g_dis


class MLP(torch.nn.Module):
    """A simple feed-forward MLP with ReLU activations between linear layers.

    Args:
        num_layers: number of logical layers (controls depth).
        in_chnl: input dimension.
        hidden_chnl: hidden layer width.
        out_chnl: output dimension.
    """

    def __init__(self,
                 num_layers=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layers):
            if l == 0:  # first layer
                self.layers.append(torch.nn.Linear(in_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                if num_layers == 1:
                    self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))
            elif l <= num_layers - 2:  # hidden layers
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
            else:  # last layer
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))

    def forward(self, h):
        """Sequentially apply all (linear + ReLU) layers."""
        for lyr in self.layers:
            h = lyr(h)
        return h


class RLGNNLayer(MessagePassing):
    """One message-passing layer of the RL-GNN.

    Runs aggregation along each of the three edge graphs (``pre`` / ``suc`` /
    ``dis``), projects each aggregated message with its own :class:`MLP`, then
    merges them together with the node's own embedding and the raw input
    feature. The merge concatenates **six** vectors of width ``out_chnl``
    (hence :attr:`module_merge` takes ``6*out_chnl`` inputs):

        [ReLU(out_pre), ReLU(out_suc), ReLU(out_dis),
         ReLU(global-pooled h), h, raw_feature]

    where ``global-pooled h`` is the graph-level sum of node embeddings tiled
    back to every node, giving each node access to global context.

    Args:
        num_mlp_layer: depth of each internal :class:`MLP`.
        in_chnl: input node-feature width.
        hidden_chnl: internal MLP width.
        out_chnl: output node-embedding width.
    """

    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(RLGNNLayer, self).__init__()

        self.module_pre = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_suc = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_dis = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_merge = MLP(num_layers=num_mlp_layer, in_chnl=6*out_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.reset_parameters()

    def reset_parameters(self):
        """Re-initialise the predecessor, successor and disjunctive MLPs."""
        reset(self.module_pre)
        reset(self.module_suc)
        reset(self.module_dis)

    def message(self, x_j: Tensor) -> Tensor:
        """Message function: just forward the source node feature ``x_j``."""
        return x_j

    def forward(self, raw_feature, **graphs):
        """Run one aggregation+merge step over the ``pre``/``suc``/``dis`` graphs.

        Args:
            raw_feature: the original node features (kept as a residual input).
            graphs: keyword graphs ``pre``, ``suc`` and ``dis`` (PyG ``Data``).

        Returns:
            A dict with updated ``pre``/``suc``/``dis`` graphs whose node
            features are the merged embeddings of this layer.
        """

        graph_pre = graphs['pre']
        graph_suc = graphs['suc']
        graph_dis = graphs['dis']

        num_nodes = graph_pre.num_nodes  # either pre, suc, or dis will work
        h_before_process = graph_pre.x  # either pre, suc, or dis will work

        # message passing: aggregate source features along each edge set.
        out_pre = self.propagate(graph_pre.edge_index, x=graph_pre.x, size=None)
        out_suc = self.propagate(graph_suc.edge_index, x=graph_suc.x, size=None)
        out_dis = self.propagate(graph_dis.edge_index, x=graph_dis.x, size=None)

        # process aggregated messages (one MLP per edge type).
        out_pre = self.module_pre(out_pre)
        out_suc = self.module_suc(out_suc)
        out_dis = self.module_dis(out_dis)

        # merge different h: 6 terms of width out_chnl are concatenated.
        h = torch.cat([relu(out_pre),
                       relu(out_suc),
                       relu(out_dis),
                       relu(h_before_process.sum(dim=0).tile(num_nodes, 1)),
                       h_before_process,
                       raw_feature], dim=1)
        h = self.module_merge(h)

        # new graphs after processed by this layer
        graph_pre = Data(x=h, edge_index=graph_pre.edge_index)
        graph_suc = Data(x=h, edge_index=graph_suc.edge_index)
        graph_dis = Data(x=h, edge_index=graph_dis.edge_index)

        return {'pre': graph_pre, 'suc': graph_suc, 'dis': graph_dis}


class RLGNN(torch.nn.Module):
    """The RL-GNN embedding backbone: a stack of :class:`RLGNNLayer`s.

    The first layer takes ``in_chnl`` per node; subsequent layers take
    ``out_chnl`` (the previous layer's output width).
    """

    def __init__(self,
                 num_mlp_layer=2,
                 num_layer=3,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(RLGNN, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layer):
            if l == 0:  # initial layer
                self.layers.append(RLGNNLayer(num_mlp_layer=num_mlp_layer,
                                              in_chnl=in_chnl,
                                              hidden_chnl=hidden_chnl,
                                              out_chnl=out_chnl))
            else:  # the rest layers
                self.layers.append(RLGNNLayer(num_mlp_layer=num_mlp_layer,
                                              in_chnl=out_chnl,
                                              hidden_chnl=hidden_chnl,
                                              out_chnl=out_chnl))

    def forward(self, raw_feature, **graphs):
        """Pass the feature+graphs through every layer in sequence."""
        for layer in self.layers:
            graphs = layer(raw_feature, **graphs)
        return graphs


class PolicyNet(torch.nn.Module):
    """Stochastic policy over operations.

    Scores every node (operation) with an :class:`MLP`, restricts the logits to
    the currently feasible operations, softmaxes them into a categorical
    distribution, and samples an action. Supports :meth:`sample` for rollout
    and :meth:`evaluate` for PPO re-forward passes on stored actions.
    """

    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=1):
        super(PolicyNet, self).__init__()

        self.policy = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def _distribution(self, node_h, feasible_op_id):
        """Build a :class:`Categorical` over feasible operation indices."""
        logit = self.policy(node_h).view(-1)
        feasible_t = torch.as_tensor(feasible_op_id, device=node_h.device, dtype=torch.long)
        logits = logit[feasible_t]
        return Categorical(logits=logits)

    def sample(self, node_h, feasible_op_id):
        """Sample a feasible operation and return its log-probability and entropy.

        Returns:
            ``(action, log_prob, entropy)`` where ``action`` is the surrogate op id.
        """
        dist = self._distribution(node_h, feasible_op_id)
        sampled_idx = dist.sample()
        action = feasible_op_id[sampled_idx.item()]
        return action, dist.log_prob(sampled_idx), dist.entropy()

    def evaluate(self, node_h, feasible_op_id, action):
        """Evaluate log-probability and entropy for a stored action.

        Args:
            action: surrogate operation id that was taken at this decision step.
        """
        dist = self._distribution(node_h, feasible_op_id)
        try:
            action_idx = feasible_op_id.index(action)
        except ValueError:
            raise RuntimeError("Action {} not in feasible set {}".format(action, feasible_op_id))
        action_t = torch.tensor(action_idx, device=node_h.device, dtype=torch.long)
        return dist.log_prob(action_t), dist.entropy()

    def forward(self, node_h, feasible_op_id):
        """Sample an action (legacy API used by :func:`rollout`)."""
        action, log_prob, _ = self.sample(node_h, feasible_op_id)
        return action, log_prob


class CriticNet(torch.nn.Module):
    """Value (critic) network: pools node embeddings into a scalar state value.

    Pooling is a simple sum over the node dimension (``node_h.sum(dim=0)``)
    followed by an :class:`MLP`.
    """

    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=1):
        super(CriticNet, self).__init__()

        self.critic = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def forward(self, node_h):
        """Compute the scalar value ``v`` of the graph state from ``node_h``."""
        v = self.critic(node_h.sum(dim=0))
        return v.view(-1)


class RLGNNAgent(torch.nn.Module):
    """Joint GNN embedding backbone, policy, and critic for PPO training."""

    def __init__(self,
                 num_mlp_layer=2,
                 num_gnn_layer=3,
                 in_chnl=8,
                 hidden_chnl=256,
                 embed_chnl=8):
        super(RLGNNAgent, self).__init__()
        self.embed = RLGNN(num_mlp_layer=num_mlp_layer,
                           num_layer=num_gnn_layer,
                           in_chnl=in_chnl,
                           hidden_chnl=hidden_chnl,
                           out_chnl=embed_chnl)
        self.policy = PolicyNet(num_mlp_layer=num_mlp_layer,
                                in_chnl=embed_chnl,
                                hidden_chnl=hidden_chnl)
        self.critic = CriticNet(num_mlp_layer=num_mlp_layer,
                                in_chnl=embed_chnl,
                                hidden_chnl=hidden_chnl)

    def encode(self, g, dev):
        """Convert a networkx graph to node embeddings ``h``."""
        g_pre, g_suc, g_dis = to_pyg(g, dev)
        raw_feature = g_pre.x
        graphs = self.embed(raw_feature, pre=g_pre, suc=g_suc, dis=g_dis)
        return graphs['pre'].x

    def act(self, g, dev, feasible_op_id):
        """Sample an action and value estimate at a decision state."""
        h = self.encode(g, dev)
        action, log_prob, entropy = self.policy.sample(h, feasible_op_id)
        value = self.critic(h)
        return action, log_prob, entropy, value

    def evaluate(self, g, dev, feasible_op_id, action):
        """Re-forward for PPO: log-prob, entropy, and value at a stored state."""
        h = self.encode(g, dev)
        log_prob, entropy = self.policy.evaluate(h, feasible_op_id, action)
        value = self.critic(h)
        return log_prob, entropy, value


if __name__ == '__main__':
    # Smoke test: build a tiny instance, run every component, and check that
    # gradients flow through each network.
    random.seed(0)
    np.random.seed(1)
    torch.manual_seed(1)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    # dev = 'cpu'

    s = Simulator(3, 3, verbose=False)
    print(s.machine_matrix)
    print(s.processing_time_matrix)
    s.reset()

    g, r, done = s.observe()

    g_pre, g_suc, g_dis = to_pyg(g, dev)
    # raw feature
    raw_feature = g_pre.x

    # test mlp
    mlp = MLP().to(dev)
    # count_parameters(mlp)
    out = mlp(g_pre.x)
    mlp_grad = torch.autograd.grad(out.mean(), [param for param in mlp.parameters()])

    # test rlgnn_layer
    rlgnn_layer = RLGNNLayer().to(dev)
    # count_parameters(rlgnn_layer)
    new_graphs = rlgnn_layer(raw_feature, **{'pre': g_pre, 'suc': g_suc, 'dis': g_dis})
    loss = sum([pyg.x.mean() for pyg in new_graphs.values()])
    rlgnn_layer_grad = torch.autograd.grad(loss, [param for param in rlgnn_layer.parameters()])

    # test rlgnn net
    net = RLGNN().to(dev)
    # count_parameters(net)
    new_graphs = net(raw_feature, **{'pre': g_pre, 'suc': g_suc, 'dis': g_dis})
    loss = sum([pyg.x.mean() for pyg in new_graphs.values()])
    rlgnn_grad = torch.autograd.grad(loss, [param for param in net.parameters()])

    # test policy net
    policy = PolicyNet().to(dev)
    # count_parameters(policy)
    _, log_p = policy(new_graphs['pre'].x, s.get_doable_ops_in_list())
    policy_grad = torch.autograd.grad(log_p, [param for param in policy.parameters()])

    # test critic net
    critic = CriticNet().to(dev)
    # count_parameters(critic)
    v = critic(new_graphs['pre'].x)
    critic_grad = torch.autograd.grad(v, [param for param in critic.parameters()])
