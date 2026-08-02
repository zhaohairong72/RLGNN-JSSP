"""Proximal Policy Optimization (PPO) helpers for RL-GNN training.

Implements generalized advantage estimation (GAE) and the clipped surrogate
objective from Park et al. (2021), adapted for single-trajectory updates.
"""

import torch


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """Generalized advantage estimation along one trajectory.

    Args:
        rewards: list of per-decision rewards (float).
        values: 1D tensor of critic values ``V(s_t)`` at each decision.
        gamma: discount factor.
        lam: GAE smoothing parameter.

    Returns:
        1D tensor of advantages with the same length as ``rewards``.
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=values.device, dtype=values.dtype)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
        next_value = values[t]
    return advantages


def compute_value_targets(rewards, device, dtype=torch.float32):
    """Undiscounted return-to-go targets ``sum_{i=t}^{T-1} r_i`` (paper, gamma=1)."""
    T = len(rewards)
    targets = torch.zeros(T, device=device, dtype=dtype)
    for t in range(T):
        targets[t] = sum(rewards[t:])
    return targets


def ppo_update(agent,
               optimizer,
               transitions,
               dev,
               clip_eps=0.2,
               value_coef=0.5,
               entropy_coef=0.01,
               gamma=1.0,
               gae_lambda=0.95,
               normalize_advantages=True):
    """Run one PPO gradient step on a single collected trajectory.

    Args:
        agent: :class:`model.RLGNNAgent`.
        optimizer: torch optimizer over ``agent.parameters()``.
        transitions: list of dicts from :func:`rollout.collect_episode`.
        dev: torch device.
        clip_eps: PPO clipping epsilon.
        value_coef: critic loss coefficient (paper alpha).
        entropy_coef: entropy bonus coefficient (paper beta).
        gamma: MDP discount factor.
        gae_lambda: GAE lambda.
        normalize_advantages: subtract mean / divide std of advantages.

    Returns:
        dict with detached loss components for logging.
    """
    rewards = [float(t['reward']) for t in transitions]
    log_probs_old = torch.stack([t['log_prob'] for t in transitions])
    values_old = torch.stack([t['value'].view(-1) for t in transitions])

    advantages = compute_gae(rewards, values_old, gamma=gamma, lam=gae_lambda)
    value_targets = compute_value_targets(rewards, device=dev, dtype=values_old.dtype)

    if normalize_advantages and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_losses = []
    value_losses = []
    entropy_terms = []

    for t, trans in enumerate(transitions):
        log_prob_new, entropy, value_new = agent.evaluate(
            trans['graph'], dev, trans['feasible_op_id'], trans['action'])
        value_new = value_new.view(-1)

        ratio = torch.exp(log_prob_new - log_probs_old[t])
        surr1 = ratio * advantages[t]
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages[t]
        policy_losses.append(-torch.min(surr1, surr2))
        value_losses.append((value_new - value_targets[t]) ** 2)
        entropy_terms.append(entropy)

    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy_mean = torch.stack(entropy_terms).mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        'loss': loss.detach().item(),
        'policy_loss': policy_loss.detach().item(),
        'value_loss': value_loss.detach().item(),
        'entropy': entropy_mean.detach().item(),
        'return': sum(rewards),
        'num_steps': len(transitions),
    }
