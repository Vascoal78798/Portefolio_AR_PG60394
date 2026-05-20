from __future__ import annotations

from collections import defaultdict

from core.base import Episode, PredictionAgent
from envs.blackjack import BlackjackAction, BlackjackState


class TD0Prediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, alpha: float = 0.05, gamma: float = 1.0):
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Update the value table using TD(0).

        TODO:
        For each transition, compute the TD target:
            reward + gamma * V(next_state)
        Use 0 as the bootstrap value on terminal transitions,
        then apply the incremental TD(0) update with self.alpha.
        """
        for transition in episode.transitions:
            bootstrap = 0.0 if transition.done or transition.next_state is None else self.V[transition.next_state]
            target = transition.reward + self.gamma * bootstrap
            self.V[transition.state] += self.alpha * (target - self.V[transition.state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
