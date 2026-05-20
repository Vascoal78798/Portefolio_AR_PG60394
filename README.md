# `mia_rl`

## Package organization

- `mia_rl/core/`
  - generic abstractions such as `Environment`, `Policy`, `Agent`, `Episode`, `Transition`
- `mia_rl/envs/`
  - interactive environments for model-free methods
- `mia_rl/mdps/`
  - known-model MDP abstractions for dynamic programming classes
- `mia_rl/policies/`
  - reusable policies
- `mia_rl/agents/`
  - learning algorithms grouped by task
- `mia_rl/plots/`
  - reusable plotting functions
- `mia_rl/experiments/`
  - rollout, training, evaluation, and experiment helpers
- `mia_rl/scripts/`
  - runnable experiment scripts
- `mia_rl/outputs/`
  - generated plots and experiment results

## New files for this class (Practical 9 — Planning: MCTS)

- `mia_rl/agents/planning/mcts.py` — `MCTSNode` + `MCTSAgent`: search tree, UCB1 selection, random rollout, backup
- `mia_rl/experiments/mcts_tictactoe.py` — policy wrapper, evaluation vs random, evaluation vs REINFORCE
- `mia_rl/notebooks/TicTacToe_MCTS.ipynb` — practical notebook

## Current practical TODOs (Practical 9)

1. **`mia_rl/agents/planning/mcts.py`** — `MCTSNode.backpropagate(value)` — implement the Backup phase:
   - Increment `visit_count` by 1
   - Add `value` to `value_sum` (outcome from `self.player`'s perspective)
   - Recurse to the parent with `-value` (sign flip: parent is the opponent)

2. **`mia_rl/agents/planning/mcts.py`** — `MCTSAgent._rollout(state, player)` body — implement the default rollout policy:
   - While the state is not terminal, pick a random available action, apply it, and switch player
   - Use the pure-function helpers `_apply`, `_available`, `_is_terminal` (never call `env.step`)

3. **`mia_rl/agents/planning/mcts.py`** — `MCTSAgent._rollout(state, player)` return — return the outcome from the original `player`'s perspective: `+1.0` win, `0.0` draw, `-1.0` loss

After implementing the TODOs, open `mia_rl/notebooks/TicTacToe_MCTS.ipynb` and run it.



