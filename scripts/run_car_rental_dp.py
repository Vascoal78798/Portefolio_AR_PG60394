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
    parser = argparse.ArgumentParser(description="Run Jack's Car Rental dynamic-programming experiments.")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor.")
    parser.add_argument("--theta", type=float, default=1e-4, help="Convergence threshold.")
    parser.add_argument("--max-outer", type=int, default=20, help="Maximum policy-iteration outer loops.")
    parser.add_argument("--max-iters", type=int, default=10000, help="Maximum value-iteration sweeps.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/car_rental_dp",
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

    from experiments.car_rental_dp import policy_iteration, value_iteration
    from mdps.car_rental import CarRentalMDP, CarRentalParams
    from plots.car_rental import plot_poisson_distributions, plot_policy, plot_values

    params = CarRentalParams()
    mdp = CarRentalMDP(params)

    V_pi, pi_pi, hist = policy_iteration(
        mdp,
        gamma=args.gamma,
        theta=args.theta,
        max_outer=args.max_outer,
    )
    V_vi, pi_vi, it_vi = value_iteration(
        mdp,
        gamma=args.gamma,
        theta=args.theta,
        max_iters=args.max_iters,
    )

    print(f"Policy Iteration outer loops: {len(hist)}")
    print(f"Value Iteration iterations: {it_vi}")

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, _ = plot_poisson_distributions(params)
    fig.savefig(output_dir / "poisson_distributions.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_policy(mdp, pi_pi, title="Policy Iteration: final policy (cars moved 1->2)")
    fig.savefig(output_dir / "policy_iteration_policy.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_values(mdp, V_pi, title="Policy Iteration: V^pi")
    fig.savefig(output_dir / "policy_iteration_values.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_policy(mdp, pi_vi, title="Value Iteration: greedy policy from V*")
    fig.savefig(output_dir / "value_iteration_policy.png", dpi=150, bbox_inches="tight")

    fig, _ = plot_values(mdp, V_vi, title="Value Iteration: V*")
    fig.savefig(output_dir / "value_iteration_values.png", dpi=150, bbox_inches="tight")

    print(f"Saved plots to {output_dir}")

    if args.no_show:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()
