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
    parser = argparse.ArgumentParser(description="Run Blackjack model-free prediction experiments.")
    parser.add_argument("--episodes", type=int, default=20000, help="Number of episodes for each algorithm.")
    parser.add_argument("--td-alpha", type=float, default=0.05, help="Step-size for TD(0).")
    parser.add_argument("--n-step", type=int, default=4, help="Number of steps for n-step TD prediction.")
    parser.add_argument("--n-step-alpha", type=float, default=0.05, help="Step-size for n-step TD prediction.")
    parser.add_argument("--threshold", type=int, default=20, help="Policy threshold: hit below this sum.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument("--output-dir", type=str, default="outputs/blackjack_prediction", help="Directory where plots will be saved.")
    parser.add_argument("--no-show", action="store_true", help="Disable interactive plot display.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    from agents.prediction import FirstVisitMonteCarloPrediction, NStepTDPrediction, TD0Prediction
    from envs.blackjack import BlackjackEnv
    from experiments.training import generate_episode, train_prediction_agent
    from plots.blackjack import plot_value_difference, plot_value_function
    from policies.blackjack import ThresholdPolicy

    policy = ThresholdPolicy(threshold=args.threshold)

    try:
        sample_env = BlackjackEnv(seed=args.seed)
        sample_episode = generate_episode(sample_env, policy)
        print(f"Sample episode length: {len(sample_episode.transitions)}")
        print("First transitions:")
        for transition in sample_episode.transitions[:5]:
            print(transition)

        mc_env = BlackjackEnv(seed=args.seed)
        td_env = BlackjackEnv(seed=args.seed)
        n_step_env = BlackjackEnv(seed=args.seed)
        mc_agent = FirstVisitMonteCarloPrediction(gamma=1.0)
        td_agent = TD0Prediction(alpha=args.td_alpha, gamma=1.0)
        n_step_agent = NStepTDPrediction(n=args.n_step, alpha=args.n_step_alpha, gamma=1.0)

        checkpoints = sorted({cp for cp in (1000, 5000, args.episodes) if cp <= args.episodes})

        print(f"Training First-Visit Monte Carlo for {args.episodes} episodes...")
        mc_history = train_prediction_agent(mc_env, policy, mc_agent, args.episodes, checkpoints=checkpoints)

        print(f"Training TD(0) for {args.episodes} episodes...")
        td_history = train_prediction_agent(td_env, policy, td_agent, args.episodes, checkpoints=checkpoints)
        print(f"Training {args.n_step}-step TD for {args.episodes} episodes...")
        n_step_history = train_prediction_agent(n_step_env, policy, n_step_agent, args.episodes, checkpoints=checkpoints)
        final_mc = mc_history[args.episodes]
        final_td = td_history[args.episodes]
        final_n_step = n_step_history[args.episodes]

        fig_mc, _ = plot_value_function(final_mc, title=f"First-Visit MC after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
        fig_td, _ = plot_value_function(final_td, title=f"TD(0) after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
        fig_n_step, _ = plot_value_function(final_n_step, title=f"{args.n_step}-step TD after {args.episodes} episodes", vmin=-1.0, vmax=1.0)
        fig_td_minus_mc, _ = plot_value_difference(final_td, final_mc, title="TD(0) - First-Visit MC", vmin=-1.0, vmax=1.0)
        fig_n_step_minus_mc, _ = plot_value_difference(final_n_step, final_mc, title=f"{args.n_step}-step TD - First-Visit MC", vmin=-1.0, vmax=1.0)
        fig_n_step_minus_td, _ = plot_value_difference(final_n_step, final_td, title=f"{args.n_step}-step TD - TD(0)", vmin=-1.0, vmax=1.0)

        output_dir = PROJECT_ROOT / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_mc.savefig(output_dir / "blackjack_mc.png", dpi=150, bbox_inches="tight")
        fig_td.savefig(output_dir / "blackjack_td0.png", dpi=150, bbox_inches="tight")
        fig_n_step.savefig(output_dir / f"blackjack_n_step_td_n{args.n_step}.png", dpi=150, bbox_inches="tight")
        fig_td_minus_mc.savefig(output_dir / "blackjack_td_minus_mc.png", dpi=150, bbox_inches="tight")
        fig_n_step_minus_mc.savefig(output_dir / f"blackjack_n_step_td_n{args.n_step}_minus_mc.png", dpi=150, bbox_inches="tight")
        fig_n_step_minus_td.savefig(output_dir / f"blackjack_n_step_td_n{args.n_step}_minus_td0.png", dpi=150, bbox_inches="tight")
        print(f"Saved plots to {output_dir}")

        if args.no_show:
            plt.close("all")
        else:
            plt.show()
    except NotImplementedError as exc:
        print("\nThis practical is not complete yet.")
        print("Please finish the TODOs in:")
        print("- envs/blackjack.py")
        print("- agents/prediction/monte_carlo.py")
        print("- agents/prediction/td.py")
        print(f"\nOriginal message: {exc}")
        return


if __name__ == "__main__":
    main()
