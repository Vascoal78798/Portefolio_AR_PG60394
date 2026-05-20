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
    parser = argparse.ArgumentParser(description="Run dynamic-programming GridWorld experiments.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--theta", type=float, default=1e-8, help="Convergence threshold.")
    parser.add_argument("--max-iters", type=int, default=10000, help="Maximum number of iterations.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gridworld_dp",
        help="Directory where plots will be saved.",
    )
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from envs.gridworld import Gridworld
    from experiments.gridworld_dp import (
        greedy_policy_from_V,
        policy_evaluation,
        policy_evaluation_Q,
        value_iteration,
    )
    from plots.gridworld import plot_grid_values_and_policy
    from policies.gridworld import uniform_random_policy

    env = Gridworld()
    policy = uniform_random_policy(env)

    V_pi, eval_iters = policy_evaluation(
        env,
        policy,
        gamma=args.gamma,
        theta=args.theta,
        max_iters=args.max_iters,
    )
    V_star, vi_iters = value_iteration(
        env,
        gamma=args.gamma,
        theta=args.theta,
        max_iters=args.max_iters,
    )
    pi_star = greedy_policy_from_V(env, V_star, gamma=args.gamma)
    _, q_iters = policy_evaluation_Q(
        env,
        policy,
        gamma=args.gamma,
        theta=args.theta,
        max_iters=args.max_iters,
    )

    print(f"Policy evaluation converged in {eval_iters} iterations")
    print(f"Value iteration converged in {vi_iters} iterations")
    print(f"Q^pi evaluation converged in {q_iters} iterations")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_grid_values_and_policy(
        env,
        V_pi,
        None,
        title="Policy evaluation: V^pi (uniform random policy)",
    )
    fig.savefig(output_dir / "gridworld_v_pi.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_grid_values_and_policy(
        env,
        V_star,
        pi_star,
        title="Value iteration: V* and greedy policy",
    )
    fig.savefig(output_dir / "gridworld_v_star.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
