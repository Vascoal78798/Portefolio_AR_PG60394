from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run k-armed bandit experiments.")
    parser.add_argument(
        "--experiment",
        choices=["epsilon-greedy", "optimistic-vs-ucb", "gradient-bandit", "all"],
        default="all",
        help="Which comparison to run.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps per run.")
    parser.add_argument("--runs", type=int, default=2000, help="Number of independent runs.")
    parser.add_argument("--k", type=int, default=10, help="Number of bandit arms.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/k_bandits",
        help="Directory where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def _run_epsilon_greedy(k: int, steps: int, runs: int, seed: int):
    from agents.bandits import EpsilonGreedy
    from envs.k_bandits import KArmedBandit
    from experiments.bandits import run_experiment

    curves: dict[str, object] = {}
    env = KArmedBandit(k=k, seed=seed)
    for eps in (0.0, 0.01, 0.1):
        agent = EpsilonGreedy(k=k, epsilon=eps, seed=seed)
        rewards, _ = run_experiment(agent, env, steps=steps, runs=runs)
        curves[f"epsilon={eps:g}"] = rewards
    return curves


def _run_optimistic_vs_ucb(k: int, steps: int, runs: int, seed: int):
    from agents.bandits import EpsilonGreedy, UCB
    from envs.k_bandits import KArmedBandit
    from experiments.bandits import run_experiment

    curves: dict[str, object] = {}
    env = KArmedBandit(k=k, seed=seed)
    agents = {
        "Optimistic greedy": EpsilonGreedy(k=k, epsilon=0.0, optimistic=5.0, seed=seed),
        "UCB c=2": UCB(k=k, c=2.0, seed=seed),
    }
    for label, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps=steps, runs=runs)
        curves[label] = rewards
    return curves


def _run_gradient_bandit(k: int, steps: int, runs: int, seed: int):
    from agents.bandits import GradientBandit
    from envs.k_bandits import KArmedBandit
    from experiments.bandits import run_experiment

    curves: dict[str, object] = {}
    env = KArmedBandit(k=k, seed=seed)
    agents = {
        "alpha=0.1 baseline": GradientBandit(k=k, alpha=0.1, baseline=True, seed=seed),
        "alpha=0.4 baseline": GradientBandit(k=k, alpha=0.4, baseline=True, seed=seed),
        "alpha=0.1 no baseline": GradientBandit(k=k, alpha=0.1, baseline=False, seed=seed),
    }
    for label, agent in agents.items():
        rewards, _ = run_experiment(agent, env, steps=steps, runs=runs)
        curves[label] = rewards
    return curves


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from plots.bandits import plot_reward_curves

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.experiment in ("epsilon-greedy", "all"):
        curves = _run_epsilon_greedy(args.k, args.steps, args.runs, args.seed)
        fig, _ = plot_reward_curves(curves, title="epsilon-greedy comparison")
        fig.savefig(output_dir / "epsilon_greedy.png", dpi=150, bbox_inches="tight")

    if args.experiment in ("optimistic-vs-ucb", "all"):
        curves = _run_optimistic_vs_ucb(args.k, args.steps, args.runs, args.seed)
        fig, _ = plot_reward_curves(curves, title="Optimistic initial values vs UCB")
        fig.savefig(output_dir / "optimistic_vs_ucb.png", dpi=150, bbox_inches="tight")

    if args.experiment in ("gradient-bandit", "all"):
        curves = _run_gradient_bandit(args.k, args.steps, args.runs, args.seed)
        fig, _ = plot_reward_curves(curves, title="Gradient bandit methods")
        fig.savefig(output_dir / "gradient_bandit.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
