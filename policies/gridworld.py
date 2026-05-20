from __future__ import annotations

from typing import Literal

from envs.gridworld import ACTIONS, Gridworld, GridworldAction, GridworldState

GridworldStochasticPolicy = dict[GridworldState, dict[GridworldAction, float]]
GridworldGreedyAction = GridworldAction | Literal["."]
GridworldGreedyPolicy = dict[GridworldState, GridworldGreedyAction]


def uniform_random_policy(env: Gridworld) -> GridworldStochasticPolicy:
    policy: GridworldStochasticPolicy = {}
    for state in env.states():
        if env.is_terminal(state):
            policy[state] = {action: 0.0 for action in ACTIONS}
        else:
            prob = 1.0 / len(ACTIONS)
            policy[state] = {action: prob for action in ACTIONS}
    return policy
