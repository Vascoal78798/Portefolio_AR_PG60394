from __future__ import annotations

import random
from collections import defaultdict

from mia_rl.agents.control.base import ActionT, ControlAgent, StateT
from mia_rl.core.base import Transition


class NStepSarsaControl(ControlAgent[StateT, ActionT]):
    """N-step SARSA control (Sutton & Barto, Section 7.2).

    Generalises one-step SARSA by using an n-step return as the update
    target:

        G_{t:t+n} = R_{t+1} + γ R_{t+2} + ... + γ^{n-1} R_{t+n}
                     + γ^n Q(S_{t+n}, A_{t+n})

    When n = 1 this reduces to standard (one-step) SARSA.

    Because the training loop in ``experiments/control.py`` calls
    ``update_transition`` after each step, this agent buffers transitions
    internally and performs the delayed n-step updates once enough
    experience has been accumulated.  Any remaining updates are flushed
    when ``end_episode`` is called at the episode boundary.
    """

    def __init__(
        self,
        actions: tuple[ActionT, ...],
        n_steps: int = 4,
        alpha: float = 0.5,
        epsilon: float = 0.1,
        gamma: float = 1.0,
        seed: int | None = None,
    ):
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._buffer: list[Transition[StateT, ActionT]] = []
        self._selected_actions: dict[StateT, ActionT] = {}

    # ── Action selection (ε-greedy) ───────────────────────────────────────

    def select_action(self, state: StateT) -> ActionT:
        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.actions)
        else:
            best_value = max(self.action_value_of(state, a) for a in self.actions)
            best_actions = [a for a in self.actions if self.action_value_of(state, a) == best_value]
            action = self.rng.choice(best_actions)
        self._selected_actions[state] = action
        return action

    # ── Learning ──────────────────────────────────────────────────────────

    def update_transition(self, transition: Transition[StateT, ActionT]) -> None:
        """Buffer the transition and perform the n-step update when ready."""
        self._buffer.append(transition)

        # We can update time step t once we have n more transitions (or the episode ends).
        # Current buffer length = t + n  →  update index = len(buffer) - n - 1
        # Only update if we have accumulated at least n+1 transitions (indices 0..n).
        t = len(self._buffer) - self.n_steps
        if t >= 0:
            self._update_step(t)

    def end_episode(self) -> None:
        """Flush remaining n-step updates at episode boundary."""
        # Update all time steps that haven't been updated yet.
        # Steps 0..len(buffer)-n_steps-1 were already updated online;
        # steps len(buffer)-n_steps..len(buffer)-1 still need updating.
        start = max(0, len(self._buffer) - self.n_steps + 1)
        # But some of those may already have been handled if n_steps > buffer length
        start = max(start, len(self._buffer) - self.n_steps)
        # Simplify: just update all remaining un-updated steps
        already_updated = max(0, len(self._buffer) - self.n_steps)
        for t in range(already_updated, len(self._buffer)):
            self._update_step(t)
        self._buffer.clear()
        self._selected_actions.clear()

    def _update_step(self, t: int) -> None:
        """Perform the n-step SARSA update for time step t."""
        T = len(self._buffer)

        # Build the n-step return: G_{t:t+n}
        G = 0.0
        end = min(t + self.n_steps, T)
        for k in range(t, end):
            G += (self.gamma ** (k - t)) * self._buffer[k].reward

        # Bootstrap from Q(S_{t+n}, A_{t+n}) if the episode hasn't ended
        last_transition = self._buffer[end - 1]
        if not last_transition.done and end < T:
            # The next transition at index `end` gives us S_{t+n} and A_{t+n}
            next_s = self._buffer[end].state
            next_a = self._buffer[end].action
            G += (self.gamma ** self.n_steps) * self.action_value_of(next_s, next_a)
        elif not last_transition.done and end == T and last_transition.next_state is not None:
            # We're at the buffer boundary but the episode isn't done —
            # bootstrap from the action that was already selected for next_state
            next_s = last_transition.next_state
            if next_s in self._selected_actions:
                next_a = self._selected_actions[next_s]
                G += (self.gamma ** (end - t)) * self.action_value_of(next_s, next_a)

        # Update Q(S_t, A_t)
        state = self._buffer[t].state
        action = self._buffer[t].action
        current_q = self.action_value_of(state, action)
        self.Q[(state, action)] = current_q + self.alpha * (G - current_q)

    # ── Value queries ─────────────────────────────────────────────────────

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda a: self.action_value_of(state, a))
