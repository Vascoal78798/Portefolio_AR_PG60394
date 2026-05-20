from __future__ import annotations

import random
from collections import defaultdict

from agents.control.base import ActionT, ControlAgent, StateT
from core.base import Transition


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
            raise ValueError("n_steps must be at least 1.")
        self.actions = actions
        self.n_steps = n_steps
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = random.Random(seed)
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.Q = defaultdict(float)
        self._selected_actions: dict[StateT, ActionT] = {}
        self._pending_transitions: list[Transition[StateT, ActionT]] = []

    # ── Action selection (ε-greedy) ───────────────────────────────────────

    def select_action(self, state: StateT) -> ActionT:
        """Choose an epsilon-greedy action and cache it for the n-step bootstrap.

        TODO:
        1. With probability `self.epsilon`, choose a random action from `self.actions`.
        2. Otherwise choose an action with the highest current action-value.
        3. Store the chosen action in `self._selected_actions[state]` and return it.
        """
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
        """Store the transition and update the oldest state-action when possible.

        TODO:
        1. Append each transition to `self._pending_transitions`.
        2. If the episode ended, keep updating and removing the oldest transition until the buffer is empty.
        3. Otherwise, once the buffer length reaches `self.n_steps`, update the oldest transition and remove it.
        4. Reuse `_update_oldest_transition()` for the actual target computation.
        """
        self._pending_transitions.append(transition)

        if transition.done:
            while self._pending_transitions:
                self._update_oldest_transition()
                self._pending_transitions.pop(0)
            return

        if len(self._pending_transitions) >= self.n_steps:
            self._update_oldest_transition()
            self._pending_transitions.pop(0)

    def end_episode(self) -> None:
        """Flush any leftover pending transitions at the end of the episode.

        TODO:
        1. If there are still transitions in `self._pending_transitions`, keep updating the oldest one.
        2. Remove each oldest transition after its update.
        3. Clear cached selected actions before the next episode starts.
        """
        while self._pending_transitions:
            self._update_oldest_transition()
            self._pending_transitions.pop(0)
        self._selected_actions.clear()

    def _update_oldest_transition(self) -> None:
        """Compute the n-step target for the oldest transition still in the buffer.

        TODO:
        1. Build a window with at most `self.n_steps` transitions starting from the oldest one.
        2. Sum the discounted rewards inside that window.
        3. If the window has exactly `self.n_steps` transitions and is non-terminal, bootstrap from
           `Q(last_step.next_state, cached_next_action)`.
        4. Apply the incremental update with `self.alpha` to the oldest `(state, action)` pair.
        """
        window = self._pending_transitions[: self.n_steps]
        G = 0.0
        for k, step in enumerate(window):
            G += (self.gamma ** k) * step.reward

        last_step = window[-1]
        if len(window) == self.n_steps and not last_step.done and last_step.next_state is not None:
            next_s = last_step.next_state
            next_a = self._selected_actions[next_s]
            G += (self.gamma ** self.n_steps) * self.action_value_of(next_s, next_a)

        oldest = self._pending_transitions[0]
        current_q = self.action_value_of(oldest.state, oldest.action)
        self.Q[(oldest.state, oldest.action)] = current_q + self.alpha * (G - current_q)

    # ── Value queries ─────────────────────────────────────────────────────

    def action_value_of(self, state: StateT, action: ActionT) -> float:
        return float(self.Q[(state, action)])

    def greedy_action(self, state: StateT) -> ActionT:
        return max(self.actions, key=lambda a: self.action_value_of(state, a))
