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
    parser = argparse.ArgumentParser(description="Run GridWorld policy-improvement and policy-iteration experiments.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--theta", type=float, default=1e-8, help="Convergence threshold.")
    parser.add_argument("--max-outer", type=int, default=100, help="Maximum policy-iteration outer loops.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gridworld_policy_iteration",
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
    from experiments.gridworld_dp import policy_evaluation, policy_improvement, policy_iteration
    from plots.gridworld import plot_grid_values_and_policy
    from policies.gridworld import uniform_random_policy

    env = Gridworld()
    pi0 = uniform_random_policy(env)
    V_pi0, eval_iters = policy_evaluation(env, pi0, gamma=args.gamma, theta=args.theta)
    pi1_actions, _ = policy_improvement(env, V_pi0, old_policy_actions=None, gamma=args.gamma)
    V_star, pi_star_actions, hist = policy_iteration(
        env,
        gamma=args.gamma,
        theta=args.theta,
        max_outer=args.max_outer,
    )

    print(f"Policy evaluation converged in iterations: {eval_iters}")
    print(f"Policy iteration outer loops: {len(hist)}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_grid_values_and_policy(
        env,
        V_pi0,
        None,
        title="Policy evaluation: V^pi (uniform random policy)",
    )
    fig.savefig(output_dir / "gridworld_policy_eval.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_grid_values_and_policy(
        env,
        V_pi0,
        pi1_actions,
        title="Greedy policy w.r.t. V^pi",
    )
    fig.savefig(output_dir / "gridworld_policy_improvement.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_grid_values_and_policy(
        env,
        V_star,
        pi_star_actions,
        title="Policy Iteration: V* and greedy policy",
    )
    fig.savefig(output_dir / "gridworld_policy_iteration_final.png", dpi=150, bbox_inches="tight")

    if hist:
        num_plots = len(hist)
        fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 6, 6))
        if num_plots == 1:
            axes = [axes]

        for idx, (outer_iter, pe_iters, V_hist, pi_actions_hist) in enumerate(hist):
            plot_grid_values_and_policy(
                env,
                V_hist,
                pi_actions_hist,
                title=f"Outer Loop {outer_iter}\nInner Iters: {pe_iters}",
                ax=axes[idx],
            )
        fig.tight_layout()
        fig.savefig(output_dir / "gridworld_policy_iteration_history.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
