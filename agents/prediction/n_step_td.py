from __future__ import annotations

from collections import defaultdict

from core.base import Episode, PredictionAgent
from envs.blackjack import BlackjackAction, BlackjackState


class NStepTDPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    """N-step TD prediction (Sutton & Barto, Section 7.1).

    Generalises TD(0) and First-Visit Monte Carlo by using an n-step
    return as the update target:

        G_{t:t+n} = R_{t+1} + γ R_{t+2} + ... + γ^{n-1} R_{t+n} + γ^n V(S_{t+n})

    When n = 1 this reduces to TD(0).
    When n ≥ episode length it behaves like Monte Carlo (bootstrap term vanishes).

    The update is applied at every time step t using the n-step return:

        V(S_t) += α · (G_{t:t+n} − V(S_t))

    All updates are performed within ``update_episode`` after the full
    episode has been collected, which avoids the complexity of an
    online n-step buffer while remaining faithful to the algorithm.
    """

    def __init__(self, n: int = 4, alpha: float = 0.05, gamma: float = 1.0):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float)

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        T = len(episode.transitions)
        if T == 0:
            return

        for t in range(T):
            # Build the n-step return G_{t:t+n}
            G = 0.0
            # Sum discounted rewards from t to min(t+n, T) - 1
            end = min(t + self.n, T)
            for k in range(t, end):
                G += (self.gamma ** (k - t)) * episode.transitions[k].reward

            # Bootstrap from V(S_{t+n}) if the episode hasn't ended by step t+n
            if end < T:
                bootstrap_state = episode.transitions[end].state
                G += (self.gamma ** self.n) * self.V[bootstrap_state]

            # Update V(S_t)
            state = episode.transitions[t].state
            self.V[state] += self.alpha * (G - self.V[state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
