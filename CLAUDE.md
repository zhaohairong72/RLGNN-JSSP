# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **PyTorch re-implementation** of the RL-GNN model from Park et al. (2021): *"Learning to schedule job-shop problems: representation and policy learning using graph neural network and reinforcement learning."* It solves the **Job-Shop Scheduling Problem (JSSP)** with a Graph Neural Network (GNN) policy trained via RL.

Status: Environment rollout + GNN inference + PPO training implemented.

## Key Dependencies

- **Python 3.9.9**, **PyTorch 1.10.1**, **PyTorch Geometric 2.0.2**, **CUDA 11.3**
- networkx (disjunctive graph construction), ortools, plotly, matplotlib

## File-by-File Reference

### Top-level files

| File | Contents |
|---|---|
| `model.py` | **GNN model definition** — `to_pyg()` (converts networkx disjunctive graph → three PyG `Data` graphs: pre, suc, dis), `MLP` (lightweight feed-forward), `RLGNNLayer` (one message-passing layer: aggregates along pre/suc/dis edges, processes with per-type MLPs, merges 6 terms including global-pooled embedding), `RLGNN` (stack of RLGNNLayers), `PolicyNet` (scores nodes → categorical over feasible ops → sampled action + log-prob + entropy; includes `evaluate()` for PPO re-forward passes), `CriticNet` (pooled graph → scalar value), `RLGNNAgent` (joint wrapper with `encode()`, `act()`, `evaluate()` for training), `count_parameters()`. `__main__` block is a gradient-flow smoke test. |
| `rollout.py` | **Rollout driver** — `rollout()` runs one full semi-MDP episode with GNN or random policy, returns `(p_list, elapsed, makespan)`. `collect_episode()` records PPO trajectories with utilization rewards, storing graphs + log_probs + values per decision step. `__main__` benchmarks run-time across problem sizes and saves `.npy` arrays. |
| `verify_rollout.py` | **Makespan verification** — `verify_env(n, m)` rolls out a random instance with a random policy, builds the disjunctive DAG from the realized schedule (precedence edges + machine-clique edges weighted by processing time), computes the longest path via `nx.dag_longest_path_length`, and compares it to the simulator's reported makespan. `adj_conjunction()` builds the precedence adjacency with dummy S/T nodes. |
| `ppo.py` | **PPO loss and advantage estimation** — `compute_gae()` (generalized advantage estimation from the paper, γ=1.0, λ=0.95), `compute_value_targets()` (undiscounted return-to-go for γ=1), `ppo_update()` (one gradient step per trajectory: re-forwards stored states through the agent, computes clipped surrogate objective + value MSE − entropy bonus, logs all loss components). |
| `train.py` | **PPO training driver** — `train()` loops over iterations: samples a random JSSP instance (m~U(5,9), n~U(m,9), PT~U(1,99) per the paper), collects one trajectory via `collect_episode()` with `reward='utilization'`, runs `ppo_update()`, and logs. Supports fixed-size instances, periodic checkpointing, and resampling schedule. CLI via argparse. |

### `semiMDP/` — Environment (semi-MDP)

| File | Contents |
|---|---|
| `__init__.py` | **Package re-exports** — imports `*` from utils, jobShopSamplers, machineHelpers, and simulators so callers can write `from semiMDP import Simulator`. |
| `configs.py` | **Global constants** — node signature enums (`NOT_START_NODE_SIG = -1`, `PROCESSING_NODE_SIG = 0`, `DONE_NODE_SIG = 1`, `DELAYED_NODE_SIG = 2`, `DUMMY_NODE_SIG = 3`), edge type enums (`CONJUNCTIVE_TYPE = 0` for intra-job, `DISJUNCTIVE_TYPE = 1` for cross-job), edge direction (`FORWARD = 0`, `BACKWARD = 1`), and text tokens for benchmark file parsing (`N_SEP = 1`, `SEP = ' '`, `NEW = '\n'`). |
| `simulators.py` | **Simulator classes** — `Simulator` (main semi-MDP: owns a `JobManager` and `MachineManager`, exposes `reset()`, `transit()`, `observe()`, `process_one_time()`, `flush_trivial_ops()`, `get_doable_ops()`, `plot_graph()`, `draw_gantt_chart()`). Has `from_path()` (load standard JSSP file) and `from_TA_path()` (load TA-format with separate machine/PT files) classmethods. `NodeProcessingTimeSimulator` (subclass using complete-ratio-weighted edges + explicit end node). |
| `operationHelpers.py` | **Operations and jobs** — `Operation` (lifecycle via `node_status`, feature dict via `.x` property), `Job` (linked chain of `Operation`s with `prev_op`/`next_op` pointers), `JobManager` (owns all jobs, builds the disjunctive graph in `observe()` with conjunctive and disjunctive edges, supports surrogate indexing). Variants: `NodeProcessingTimeOperation`, `NodeProcessingTimeJob`, `NodeProcessingTimeJobManager` for the complete-ratio graph flavour. Dummy/start/end sentinel nodes: `DummyOperation`, `StartOperation`, `EndOperation`, `NodeProcessingTimeEndOperation`. Helpers: `get_edge_color_map()`, `calc_positions()`, `get_node_color_map()`. Includes Plotly Gantt chart rendering via `draw_gantt_chart()`. |
| `machineHelpers.py` | **Machine dispatch** — `Machine` (single machine: backlog in `possible_ops`/`remain_ops`, `current_op`, `delayed_op` look-ahead reservation, `doable_ops()` with delayed vs no-delay modes, `load_op()`, `unload()`, `do_processing()`, `transit()`). `MachineManager` (owns all machines, routes `do_processing()`, `get_available_machines()`, `get_idle_machines()`, `cal_total_cost()` for utilization reward, `fab_stuck()` deadlock detection). `NodeProcessingTimeMachineManager` (subclass paired with node-processing-time operations). |
| `jobShopSamplers.py` | **Random instance generation** — `jssp_sampling(m, n, low=5, high=100)` returns `(machine_mat, process_time_mat)`: each job row is a random permutation of machines, processing times uniform in `[low, high)`. |
| `utils.py` | **Misc helpers** — `pprint_graph()` (print all node/edge attributes of a networkx graph), `check_equal()` (all array elements identical?), `instance_generate()` (random small problem size), `doable_ops_dr()` (no-delay doable ops for a machine), `Timer` (context manager for wall-clock timing). |
| `benchmarks.py` | **Hard-coded benchmarks** — `FT06()` (6×6 Fisher-Thompson, optimal makespan 55) and `FT10()` (10×10 Fisher-Thompson, optimal makespan 930). Each returns a pre-configured `Simulator` instance. |
| `benchmarks/` | **Reference instance files** — `.txt` files for FT (Fisher-Thompson), LA (Lawrence), and TA (Taillard) benchmarks in standard JSSP format. |

### `plt/` — Plotting

| File | Contents |
|---|---|
| `__init__.py` | Package marker — documents that the directory holds rollout `.npy` arrays and generated `.png` figures. |
| `plot_complexity.py` | **Complexity plotting** — loads per-instance wall-clock `.npy` arrays (original RL-GNN, this re-implementation, optional L2S baseline) and plots seconds vs number of machines (or jobs). Saves PNG to `plt/`. |

## How the System Fits Together

1. **Instance creation**: `Simulator` samples a random JSSP instance (`jobShopSamplers.jssp_sampling`) or loads one from a file/benchmark. The machine-order matrix and processing-time matrix define the problem.

2. **Simulation loop** (semi-MDP, in `rollout.rollout`):
   - When all machines are busy → advance the clock (`process_one_time`).
   - When some machines have work → flush forced moves (`flush_trivial_ops`, auto-executes machines with exactly one doable op).
   - At a real decision point → observe the disjunctive graph, forward it through `RLGNN` → `PolicyNet` to sample an action, then apply it via `transit`.

3. **Graph representation**: `JobManager.observe()` builds a networkx `OrderedDiGraph` where each operation is a node with features (type, processing_time, complete_ratio, remaining_ops, waiting_time, remain_time). Edges are conjunctive (intra-job precedence) or disjunctive (cross-job machine conflicts).

4. **GNN model** (`model.py`):
   - `to_pyg()` splits the networkx graph into three PyG `Data` graphs: `pre` (forward conjunctive), `suc` (backward conjunctive), `dis` (disjunctive).
   - `RLGNN` stacks `RLGNNLayer` blocks. Each layer runs message passing along all three edge sets, processes with per-type MLPs, then merges — including the global-pooled node embedding — into updated embeddings.
   - `PolicyNet` scores nodes with an MLP, restricts to feasible ops, produces a categorical distribution, and samples an action.
   - `CriticNet` pools node embeddings → scalar state value (for future actor-critic training).

5. **Two graph flavours**: The base `Simulator` uses processing-time-weighted conjunctive edges. `NodeProcessingTimeSimulator` uses complete-ratio-weighted edges with an explicit end node.

6. **PPO Training** (`train.py` + `ppo.py`):
   - Each iteration samples a random JSSP instance (m~U(5,9), n~U(m,9), PT~U(1,99)).
   - `collect_episode()` rolls out one trajectory under the current policy, recording every decision state, action, log-prob, and critic value. Rewards are the negative total queue length (`utilization` reward) accumulated between decisions.
   - `ppo_update()` runs one gradient step: recomputes log-probs and values via `RLGNNAgent.evaluate()`, computes GAE advantages (γ=1, λ=0.95), then applies the clipped surrogate objective with entropy bonus (coef β=0.01) and value MSE (coef α=0.5).
   - The agent is a single `RLGNNAgent` module containing GNN backbone + `PolicyNet` + `CriticNet`, trained end-to-end via one Adam optimizer (lr 2.5×10⁻⁴).

## Key Details & Gotchas

- Surrogate indices: Operations are keyed by flat integer `sur_id` (contiguous) for GNN indexing, mapped via `sur_index_dict` back to `(job_id, step_id)`.
- Delay mechanism: Machines can reserve a delayed (not-yet-processible) operation as a look-ahead — the machine waits for its predecessor to finish.
- Node features for finished ops are zeroed out so they don't contribute to message passing.
- The `verify_rollout.py` cross-checks correctness by building the disjunctive DAG from the realized schedule and comparing its longest path to the simulator's makespan.
- Machine ids are **1-based** internally (shifted +1 from the 0-based input matrices).

## Common Commands

```bash
# Verify environment rollout (random policy) produces correct makespan
python verify_rollout.py

# Rollout with GNN policy and benchmark run-time
python rollout.py

# Plot complexity results (loads .npy files from plt/)
python -m plt.plot_complexity

# Run model smoke test (gradient flow check)
python model.py

# Train RL-GNN with PPO (default: 200 iterations, random instances)
python train.py

# Train on a fixed-size instance (good for debugging)
python train.py --fixed-m 6 --fixed-n 6 --iterations 100

# Train with GPU, checkpointing every 50 iters, 1000 total iters
python train.py --iterations 1000 --checkpoint-every 50

# Full paper-style: resample fresh instances every 5 updates
python train.py --iterations 10000 --resample-every 5 --save-final final_weights.pt
```

### Running on CPU vs GPU

Set device in `rollout.py` `__main__`:
```python
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
```

### Loading a benchmark instance

```python
from semiMDP.benchmarks import FT06
sim = FT06(verbose=False)  # Same API as Simulator
```