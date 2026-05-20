from __future__ import annotations

import numpy as np

from envs.gridworld import ACTIONS, Gridworld, GridworldAction, GridworldState
from policies.gridworld import GridworldGreedyPolicy, GridworldStochasticPolicy, uniform_random_policy


def zeros_V(env: Gridworld) -> np.ndarray:
    return np.zeros((env.n_rows, env.n_cols), dtype=float)


def zeros_Q(env: Gridworld) -> np.ndarray:
    return np.zeros((env.n_rows, env.n_cols, len(ACTIONS)), dtype=float)


def bellman_expectation_update(
    env: Gridworld,
    V: np.ndarray,
    policy: GridworldStochasticPolicy,
    state: GridworldState,
    gamma: float,
) -> float:
    if env.is_terminal(state):
        return 0.0

    value = 0.0
    for action in ACTIONS:
        prob = policy[state][action]
        next_state, reward, _ = env.transition(state, action)
        value += prob * (reward + gamma * V[next_state])
    return value


def policy_evaluation(
    env: Gridworld,
    policy: GridworldStochasticPolicy,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> tuple[np.ndarray, int]:
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for state in env.states():
            v_new = bellman_expectation_update(env, V_old, policy, state, gamma)
            delta = max(delta, abs(v_new - V[state]))
            V[state] = v_new
        if delta < theta:
            return V, it + 1

    return V, max_iters


def bellman_optimality_update(
    env: Gridworld,
    V: np.ndarray,
    state: GridworldState,
    gamma: float,
) -> float:
    if env.is_terminal(state):
        return 0.0

    best = -np.inf
    for action in ACTIONS:
        next_state, reward, _ = env.transition(state, action)
        best = max(best, reward + gamma * V[next_state])
    return float(best)


def value_iteration(
    env: Gridworld,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> tuple[np.ndarray, int]:
    V = zeros_V(env)

    for it in range(max_iters):
        delta = 0.0
        V_old = V.copy()
        for state in env.states():
            v_new = bellman_optimality_update(env, V_old, state, gamma)
            delta = max(delta, abs(v_new - V[state]))
            V[state] = v_new
        if delta < theta:
            return V, it + 1

    return V, max_iters


def greedy_policy_from_V(
    env: Gridworld,
    V: np.ndarray,
    gamma: float,
) -> GridworldGreedyPolicy:
    policy: GridworldGreedyPolicy = {}

    for state in env.states():
        if env.is_terminal(state):
            policy[state] = "."
            continue

        best_action: GridworldAction | None = None
        best_q = -np.inf
        for action in ACTIONS:
            next_state, reward, _ = env.transition(state, action)
            q = reward + gamma * V[next_state]
            if q > best_q:
                best_action = action
                best_q = q
        policy[state] = best_action if best_action is not None else "."

    return policy


def greedy_action_from_V(
    env: Gridworld,
    V: np.ndarray,
    state: GridworldState,
    gamma: float,
) -> GridworldAction:
    best_action = ACTIONS[0]
    best_q = -np.inf

    for action in ACTIONS:
        next_state, reward, _ = env.transition(state, action)
        q = reward + gamma * V[next_state]
        if q > best_q:
            best_action = action
            best_q = q

    return best_action


def deterministic_policy_to_stochastic(
    env: Gridworld,
    policy: GridworldGreedyPolicy,
) -> GridworldStochasticPolicy:
    stochastic_policy: GridworldStochasticPolicy = {}

    for state in env.states():
        if env.is_terminal(state):
            stochastic_policy[state] = {action: 0.0 for action in ACTIONS}
            continue

        chosen_action = policy[state]
        stochastic_policy[state] = {action: float(action == chosen_action) for action in ACTIONS}

    return stochastic_policy


def policy_improvement(
    env: Gridworld,
    V: np.ndarray,
    old_policy_actions: GridworldGreedyPolicy | None = None,
    gamma: float = 0.9,
) -> tuple[GridworldGreedyPolicy, bool]:
    new_policy_actions: GridworldGreedyPolicy = {}
    stable = True

    for state in env.states():
        if env.is_terminal(state):
            new_policy_actions[state] = "."
            continue

        best_action = greedy_action_from_V(env, V, state, gamma)
        new_policy_actions[state] = best_action

        if old_policy_actions is not None and old_policy_actions.get(state) != best_action:
            stable = False

    return new_policy_actions, stable


def policy_iteration(
    env: Gridworld,
    gamma: float = 0.9,
    theta: float = 1e-8,
    max_outer: int = 100,
):
    policy_stochastic = uniform_random_policy(env)
    policy_actions: GridworldGreedyPolicy = {}
    history: list[tuple[int, int, np.ndarray, GridworldGreedyPolicy]] = []

    for outer in range(max_outer):
        V, iters = policy_evaluation(env, policy_stochastic, gamma=gamma, theta=theta)
        new_actions, stable = policy_improvement(env, V, old_policy_actions=policy_actions, gamma=gamma)
        history.append((outer, iters, V.copy(), new_actions.copy()))
        policy_actions = new_actions
        policy_stochastic = deterministic_policy_to_stochastic(env, policy_actions)

        if stable:
            return V, policy_actions, history

    return V, policy_actions, history


def policy_evaluation_Q(
    env: Gridworld,
    policy: GridworldStochasticPolicy,
    gamma: float,
    theta: float = 1e-6,
    max_iters: int = 10_000,
) -> tuple[np.ndarray, int]:
    Q = zeros_Q(env)

    for it in range(max_iters):
        delta = 0.0
        Q_old = Q.copy()

        for state in env.states():
            if env.is_terminal(state):
                Q[state] = 0.0
                continue

            for action_index, action in enumerate(ACTIONS):
                next_state, reward, _ = env.transition(state, action)
                expected_next = 0.0
                for next_action_index, next_action in enumerate(ACTIONS):
                    expected_next += (
                        policy[next_state][next_action] * Q_old[next_state][next_action_index]
                    )

                q_new = reward + gamma * expected_next
                delta = max(delta, abs(q_new - Q[state][action_index]))
                Q[state][action_index] = q_new

        if delta < theta:
            return Q, it + 1

    return Q, max_iters
