"""Rollout driver for the JSSP simulator under the RL-GNN policy.

:meth:`rollout` runs one full episode of the semi-MDP: it alternates between
advancing the clock (when every machine is busy) and making decisions using
either the GNN policy (if networks are supplied) or a uniformly random policy
(when they are not). In between decisions it relies on the simulator to auto-
play the forced ("trivial") moves via :meth:`Simulator.flush_trivial_ops`.

The ``__main__`` block is the run-time complexity benchmark described in the
README: it warms up the networks on a small instance, then rolls out across an
increasing series of problem sizes and saves the measured per-instance wall
clock time to ``.npy`` for plotting.
"""

import numpy as np
import torch
from semiMDP.simulators import Simulator
import random
import numpy
import time
from model import to_pyg, RLGNN, PolicyNet, CriticNet, RLGNNAgent


def collect_episode(sim, agent, dev, reward='utilization', gamma=1.0):
    """Collect one on-policy trajectory for PPO (decision steps only).

    Rewards are aligned with the semi-MDP: each decision step receives the
    discounted sum of ``utilization`` rewards accumulated by
    :meth:`Simulator.flush_trivial_ops` since the previous decision.

    Args:
        sim: :class:`Simulator` instance (will be reset).
        agent: :class:`RLGNNAgent` used to sample actions.
        dev: torch device.
        reward: reward type forwarded to flush / observe (default ``utilization``).
        gamma: discount for reward segments between decisions.

    Returns:
        ``(transitions, makespan, p_list)`` where ``transitions`` is a list of
        dicts with keys ``graph``, ``feasible_op_id``, ``action``, ``log_prob``,
        ``entropy``, ``value``, and ``reward``.
    """
    agent.eval()
    sim.reset()
    transitions = []
    p_list = []

    with torch.no_grad():
        while True:
            _, cum_reward, done, sub_list = sim.flush_trivial_ops(reward=reward, gamma=gamma)
            p_list += sub_list

            if transitions and transitions[-1]['reward'] is None:
                transitions[-1]['reward'] = cum_reward

            if done:
                break

            g, _, _ = sim.observe(return_doable=True)
            feasible_op_id = sim.get_doable_ops_in_list()
            action, log_prob, entropy, value = agent.act(g, dev, feasible_op_id)

            transitions.append({
                'graph': g,
                'feasible_op_id': feasible_op_id,
                'action': action,
                'log_prob': log_prob.detach(),
                'entropy': entropy.detach(),
                'value': value.detach(),
                'reward': None,
            })

            sim.transit(action)
            p_list.append(action)

    for trans in transitions:
        if trans['reward'] is None:
            trans['reward'] = 0.0

    return transitions, sim.global_time, p_list


def rollout(s, dev, embedding_net=None, policy_net=None, critic_net=None, verbose=True):
    """Roll out one episode of the JSSP simulator and return the schedule.

    Args:
        s: a :class:`Simulator` instance.
        dev: torch device to run the networks on.
        embedding_net: :class:`RLGNN` backbone, or ``None`` for random actions.
        policy_net: :class:`PolicyNet`, or ``None`` for random actions.
        critic_net: :class:`CriticNet`, or ``None`` for random actions.
        verbose: print makespan and timing when True.

    Returns:
        ``(p_list, elapsed, makespan)`` where ``p_list`` is the ordered list of
        scheduled operation ids, ``elapsed`` is the episode wall-clock time, and
        ``makespan`` is the simulator's final ``global_time``.
    """

    # Move the networks to the target device only when all three are provided.
    if embedding_net is not None and \
            policy_net is not None and \
            critic_net is not None:
        embedding_net.to(dev)
        policy_net.to(dev)
        critic_net.to(dev)

    s.reset()
    done = False

    p_list = []
    t1 = time.time()
    while True:
        do_op_dict = s.get_doable_ops_in_dict()
        # No doable ops anywhere => every machine is busy processing. Advance time.
        all_machine_work = False if bool(do_op_dict) else True

        if all_machine_work:  # all machines are on processing. keep process!
            s.process_one_time()
        else:  # some of the machine has a possibly trivial action. the others not.
            _, _, done, sub_list = s.flush_trivial_ops(reward='makespan')  # flush the trivial actions
            p_list += sub_list
            if done:
                break  # env rollout finished
            g, r, done = s.observe(return_doable=True)
            if embedding_net is not None and \
                    policy_net is not None and \
                    critic_net is not None:  # network forward goes here
                # Convert the observed graph to the three PyG graphs and embed.
                g_pre, g_suc, g_dis = to_pyg(g, dev)
                raw_feature = g_pre.x  # either pre, suc, or dis will work
                pyg_graphs = {'pre': g_pre, 'suc': g_suc, 'dis': g_dis}
                pyg_graphs = embedding_net(raw_feature, **pyg_graphs)
                # Sample a feasible operation from the policy and apply it.
                feasible_op_id = s.get_doable_ops_in_list()
                sampled_action, _ = policy_net(pyg_graphs['pre'].x, feasible_op_id)  # either pre, suc, or dis will work
                s.transit(sampled_action)
                p_list.append(sampled_action)
                # Critic forward is exercised here (its value is unused by this
                # timing benchmark) so each step mirrors a full training pass.
                v = critic_net(pyg_graphs['pre'].x)  # either pre, suc, or dis will work
            else:
                op_id = s.transit()
                p_list.append(op_id)

        if done:
            break  # env rollout finish
    t2 = time.time()
    if verbose:
        print('All job finished, makespan={}. Rollout takes {} seconds'.format(s.global_time, t2 - t1))
    return p_list, t2 - t1, s.global_time


if __name__ == "__main__":
    random.seed(0)
    numpy.random.seed(1)
    torch.manual_seed(1)

    setting = 'free_for_all'  # 'm=10', 'j=40', 'free_for_all'

    # print(np.load('./plt/RL-GNN_complexity_free_for_all_reimplement_upper.npy'))

    # Pick the series of problem sizes to benchmark. ``m`` is machines, ``j`` jobs.
    if setting == 'm=10':
        j = [10, 15, 20, 25, 30, 35, 40]
        m = [10 for _ in range(len(j))]
    elif setting == 'j=40':
        m = [10, 15, 20, 25, 30, 35, 40]
        j = [40 for _ in range(len(m))]
    else:
        # j = [15, 20, 20, 30, 30, 50, 50, 100, 10, 20, 6, 10, 20]
        # m = [15, 15, 20, 15, 20, 15, 20, 20, 10, 15, 6, 10, 5]
        j = [10, 15, 20, 10, 15, 20, 30, 15, 20, 20, 50, 10, 20]
        m = [5, 5, 5, 10, 10, 10, 10, 15, 10, 15, 10, 10, 20]
    save_dir = 'plt/RL-GNN_complexity_{}_reimplement.npy'.format(setting)

    embed = RLGNN()
    policy = PolicyNet()
    critic = CriticNet()
    # Warm start: one small rollout so CUDA kernels / lazy allocations happen
    # before the timed measurements.
    print('Warm start...')
    for p_m, p_j in zip([5], [5]):  # select problem size
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dev = 'cpu'
        s = Simulator(p_m, p_j, verbose=False)
        _, t, _ = rollout(s, dev, embed, policy, critic, verbose=False)
    times = []
    for p_m, p_j in zip(m, j):  # select problem size
        print('Problem size = (m={}, j={})'.format(p_m, p_j))
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        # dev = 'cpu'
        s = Simulator(p_m, p_j, verbose=False)
        _, t, _ = rollout(s, dev, embed, policy, critic)
        times.append(t)

    # print(times)

    numpy.save(save_dir, np.array(times))
