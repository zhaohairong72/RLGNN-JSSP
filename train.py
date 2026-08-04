"""PPO training for the RL-GNN JSSP scheduler (single-episode updates).

Each iteration collects one episode with the current policy, then applies one
PPO gradient step on that trajectory. Hyperparameters follow Park et al. (2021)
where noted in :mod:`ppo`.
"""

import argparse
import random

import numpy as np
import torch

from model import RLGNNAgent
from ppo import ppo_update
from rollout import collect_episode
from semiMDP.jobShopSamplers import jssp_sampling
from semiMDP.simulators import Simulator


def sample_training_instance(num_machines_min=5,
                             num_machines_max=9,
                             pt_low=1,
                             pt_high=100):
    """Sample a random JSSP instance (paper training distribution)."""
    m = np.random.randint(num_machines_min, num_machines_max)
    n = np.random.randint(m, num_machines_max)
    machine_mat, process_time_mat = jssp_sampling(m, n, low=pt_low, high=pt_high)
    return m, n, machine_mat, process_time_mat


def build_simulator(machine_mat, process_time_mat, verbose=False):
    """Construct a :class:`Simulator` from sampled instance matrices."""
    return Simulator(num_machines=machine_mat.shape[1],
                     num_jobs=machine_mat.shape[0],
                     machine_matrix=machine_mat,
                     processing_time_matrix=process_time_mat,
                     verbose=verbose)


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dev = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print('Device:', dev)

    agent = RLGNNAgent().to(dev)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    if args.fixed_m is not None and args.fixed_n is not None:
        machine_mat, process_time_mat = jssp_sampling(args.fixed_m, args.fixed_n,
                                                      low=args.pt_low, high=args.pt_high)
        sim = build_simulator(machine_mat, process_time_mat, verbose=False)
        print('Fixed training instance: m={}, n={}'.format(args.fixed_m, args.fixed_n))
    else:
        sim = None

    for iteration in range(1, args.iterations + 1):
        if sim is None:
            m, n, machine_mat, process_time_mat = sample_training_instance(
                args.num_machines_min, args.num_machines_max, args.pt_low, args.pt_high)
            sim = build_simulator(machine_mat, process_time_mat, verbose=False)

        agent.train()
        transitions, makespan, _ = collect_episode(
            sim, agent, dev, reward=args.reward, gamma=args.gamma)

        if len(transitions) == 0:
            print('Iteration {}: no decision steps, skipping update'.format(iteration))
            continue

        stats = ppo_update(
            agent,
            optimizer,
            transitions,
            dev,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=args.normalize_advantages,
        )

        if iteration % args.log_every == 0 or iteration == 1:
            print(
                'Iter {:4d} | steps {:3d} | makespan {:4d} | return {:.1f} | '
                'loss {:.4f} | pi {:.4f} | vf {:.4f} | H {:.4f}'.format(
                    iteration,
                    stats['num_steps'],
                    makespan,
                    stats['return'],
                    stats['loss'],
                    stats['policy_loss'],
                    stats['value_loss'],
                    stats['entropy'],
                )
            )

        if args.resample_every > 0 and iteration % args.resample_every == 0:
            sim = None
        elif args.fixed_m is None:
            sim = None

        if args.checkpoint_every > 0 and iteration % args.checkpoint_every == 0:
            path = args.checkpoint_path.format(iteration=iteration)
            torch.save({
                'iteration': iteration,
                'agent': agent.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, path)
            print('Saved checkpoint to {}'.format(path))

    if args.save_final and not args.no_save_final:
        torch.save({
            'iteration': args.iterations,
            'agent': agent.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.save_final)
        print('Saved final weights to {}'.format(args.save_final))


def parse_args():
    parser = argparse.ArgumentParser(description='PPO training for RL-GNN JSSP')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA is available')
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--clip-eps', type=float, default=0.2)
    parser.add_argument('--value-coef', type=float, default=0.5)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--reward', type=str, default='utilization',
                        choices=['utilization', 'idle_time', 'makespan'])
    parser.add_argument('--normalize-advantages', action='store_true', default=True)
    parser.add_argument('--no-normalize-advantages', dest='normalize_advantages',
                        action='store_false')
    parser.add_argument('--num-machines-min', type=int, default=5)
    parser.add_argument('--num-machines-max', type=int, default=10,
                        help='Exclusive upper bound (U(5,9) when min=5, max=10)')
    parser.add_argument('--pt-low', type=int, default=1)
    parser.add_argument('--pt-high', type=int, default=100)
    parser.add_argument('--fixed-m', type=int, default=None,
                        help='If set with --fixed-n, train on one resampled instance size')
    parser.add_argument('--fixed-n', type=int, default=None)
    parser.add_argument('--resample-every', type=int, default=0,
                        help='Resample random instance every N iterations (0 = each iter)')
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--checkpoint-path', type=str,
                        default='checkpoints/rlgnn_iter_{iteration}.pt')
    parser.add_argument('--save-final', type=str, default='checkpoints/rlgnn_final.pt')
    parser.add_argument('--no-save-final', action='store_true',
                        help='Skip writing the final checkpoint')
    return parser.parse_args()


if __name__ == '__main__':
    train(parse_args())
