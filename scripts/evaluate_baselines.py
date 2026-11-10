#!/usr/bin/env python3
"""
evaluate_baselines.py — Evaluate rule-based and random baselines.

Evaluates both policies on 1000 fixed goals and reports:
  • Success rate, average turns, average reward (± std)
  • Breakdown by goal complexity (simple / medium / complex)
  • Publication-quality comparison plots

Usage:
  cd "5. RL"
  python3 scripts/evaluate_baselines.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── project imports ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, set_seed,
    SYSTEM_ACTIONS, ACTION_TO_IDX, IDX_TO_ACTION, NUM_SYSTEM_ACTIONS,
    INFORMABLE_SLOTS, REQUESTABLE_SLOTS, BOOKING_SLOTS, STATE_DIM,
)
from src.environment import RestaurantBookingEnv
from src.policies.rule_based import RuleBasedPolicy, RandomPolicy


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation harness
# ═══════════════════════════════════════════════════════════════════════════

def generate_eval_goals(
    all_goals: list,
    n: int = 1000,
    seed: int = 42,
) -> list:
    """
    Sample N fixed evaluation goals from the goal pool.

    Uses a fixed seed so all policies are tested on the exact same goals.
    Saves the goals for reproducibility.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_goals), size=n, replace=True)
    return [all_goals[int(i)] for i in indices]


def classify_goal(goal: dict) -> str:
    """Classify a goal by complexity for breakdown analysis."""
    has_reqt = bool(goal.get("reqt", []))
    has_book = bool(goal.get("book", {}))

    if has_book and has_reqt:
        return "complex"
    elif has_book:
        return "booking"
    elif has_reqt:
        return "requests"
    else:
        return "simple"


def evaluate_policy(
    policy,
    goals: list,
    db: pd.DataFrame,
    max_turns: int,
    reward_config: dict,
    noise_level: float = 0.0,
    seed: int = 42,
) -> dict:
    """
    Run a policy on a list of goals and collect results.

    Returns a dict with per-episode results and aggregate stats.
    """
    episodes = []

    for i, goal in enumerate(goals):
        # Fresh env + rng per episode for independence
        env = RestaurantBookingEnv(
            goals=[goal],
            db=db,
            noise_level=noise_level,
            max_turns=max_turns,
            reward_config=reward_config,
            rng=np.random.default_rng(seed + i),
        )
        obs, info = env.reset(options={"goal": goal})

        ep_reward = 0.0
        for t in range(max_turns):
            action = policy.select_action(obs=obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                break

        success = bool(info.get("success", False))
        episodes.append({
            "goal_idx": i,
            "success": success,
            "turns": t + 1,
            "reward": ep_reward,
            "complexity": classify_goal(goal),
        })
        env.close()

    # Aggregate
    df = pd.DataFrame(episodes)
    stats = {
        "n": len(df),
        "success_rate": df["success"].mean(),
        "avg_turns": df["turns"].mean(),
        "std_turns": df["turns"].std(),
        "avg_reward": df["reward"].mean(),
        "std_reward": df["reward"].std(),
        "episodes": episodes,
        "df": df,
    }
    return stats


def print_results(name: str, stats: dict) -> None:
    """Pretty-print evaluation results with complexity breakdown."""
    df = stats["df"]

    print(f"\n{'━' * 55}")
    print(f"  {name}")
    print(f"{'━' * 55}")
    print(f"  {'Metric':<25s}  {'Value':>15s}")
    print(f"  {'─' * 45}")
    print(f"  {'Episodes':<25s}  {stats['n']:>15d}")
    print(f"  {'Success rate':<25s}  {stats['success_rate']:>14.1%}")
    print(f"  {'Avg turns':<25s}  {stats['avg_turns']:>10.2f} ± {stats['std_turns']:.2f}")
    print(f"  {'Avg reward':<25s}  {stats['avg_reward']:>10.2f} ± {stats['std_reward']:.2f}")

    # Complexity breakdown
    print(f"\n  {'Complexity':<15s}  {'N':>5s}  {'Success':>8s}  {'Turns':>10s}  {'Reward':>12s}")
    print(f"  {'─' * 55}")
    for cplx in ["simple", "requests", "booking", "complex"]:
        subset = df[df["complexity"] == cplx]
        if len(subset) == 0:
            continue
        sr = subset["success"].mean()
        at = subset["turns"].mean()
        ar = subset["reward"].mean()
        print(f"  {cplx:<15s}  {len(subset):>5d}  {sr:>7.1%}  {at:>10.2f}  {ar:>12.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════

def make_comparison_plots(
    results: dict,
    plot_dir: Path,
) -> None:
    """Generate publication-quality comparison plots."""
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette = sns.color_palette("viridis", len(results))

    names = list(results.keys())

    # ── Figure 1: Overall comparison (3 panels) ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1a. Success rate
    success_rates = [results[n]["success_rate"] * 100 for n in names]
    bars = axes[0].bar(names, success_rates, color=palette, edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 105)
    for bar, v in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     f"{v:.1f}%", ha="center", fontweight="bold")

    # 1b. Average turns
    avg_turns = [results[n]["avg_turns"] for n in names]
    std_turns = [results[n]["std_turns"] for n in names]
    axes[1].bar(names, avg_turns, yerr=std_turns, color=palette,
                edgecolor="black", capsize=5)
    axes[1].set_ylabel("Average Turns")
    axes[1].set_title("Dialogue Length")

    # 1c. Average reward
    avg_rewards = [results[n]["avg_reward"] for n in names]
    std_rewards = [results[n]["std_reward"] for n in names]
    axes[2].bar(names, avg_rewards, yerr=std_rewards, color=palette,
                edgecolor="black", capsize=5)
    axes[2].set_ylabel("Average Reward")
    axes[2].set_title("Total Reward")

    plt.suptitle("Baseline Policy Evaluation (1000 Goals)", y=1.02,
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "baseline_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ baseline_comparison.png")

    # ── Figure 2: Complexity breakdown ──────────────────────────────────
    complexities = ["simple", "requests", "booking", "complex"]
    cplx_labels = ["Simple\n(info only)", "Requests\n(info+reqt)",
                   "Booking\n(info+book)", "Complex\n(info+reqt+book)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(complexities))
    width = 0.35

    for i, (name, color) in enumerate(zip(names, palette)):
        df = results[name]["df"]
        rates = []
        for cplx in complexities:
            subset = df[df["complexity"] == cplx]
            rates.append(subset["success"].mean() * 100 if len(subset) > 0 else 0)
        bars = ax.bar(x + i * width, rates, width, label=name,
                      color=color, edgecolor="black")
        for bar, v in zip(bars, rates):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                        f"{v:.0f}%", ha="center", fontsize=9)

    ax.set_xlabel("Goal Complexity")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Goal Complexity", fontweight="bold")
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(cplx_labels)
    ax.set_ylim(0, 115)
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "baseline_complexity_breakdown.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ baseline_complexity_breakdown.png")

    # ── Figure 3: Reward distribution ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, color in zip(names, palette):
        rewards = [e["reward"] for e in results[name]["episodes"]]
        ax.hist(rewards, bins=40, alpha=0.6, label=name, color=color,
                edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Count")
    ax.set_title("Reward Distribution", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(plot_dir / "baseline_reward_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ baseline_reward_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# Results table (for the report)
# ═══════════════════════════════════════════════════════════════════════════

def save_results_table(results: dict, output_path: Path) -> None:
    """Save results as a clean JSON for the final report."""
    table = {}
    for name, stats in results.items():
        table[name] = {
            "success_rate": round(stats["success_rate"], 4),
            "avg_turns": round(stats["avg_turns"], 2),
            "std_turns": round(stats["std_turns"], 2),
            "avg_reward": round(stats["avg_reward"], 2),
            "std_reward": round(stats["std_reward"], 2),
        }
    with open(output_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"  ✓ Results saved to {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    set_seed(cfg["training"]["seed"])

    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    plot_dir = PROJECT_ROOT / "results" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    max_turns = cfg["environment"]["max_turns"]
    reward_config = cfg["environment"]["reward"]
    n_eval = cfg["evaluation"]["num_test_goals"]   # 1000

    # ── load data ───────────────────────────────────────────────────────
    print("Loading data …")
    db = pd.read_csv(processed_dir / "restaurant_db.csv")
    with open(processed_dir / "user_goals_test.json") as f:
        test_goals = json.load(f)
    with open(processed_dir / "user_goals_train.json") as f:
        train_goals = json.load(f)

    # ── generate fixed evaluation goals ─────────────────────────────────
    all_goals = test_goals + train_goals
    eval_goals = generate_eval_goals(all_goals, n=n_eval, seed=42)

    # Save for reproducibility
    eval_path = processed_dir / "eval_goals_1000.json"
    with open(eval_path, "w") as f:
        json.dump(eval_goals, f, indent=2)
    print(f"  ✓ {n_eval} eval goals saved to {eval_path}")

    # Show complexity distribution
    cplx_counts = defaultdict(int)
    for g in eval_goals:
        cplx_counts[classify_goal(g)] += 1
    print(f"  Complexity: {dict(cplx_counts)}")

    # ── create policies ─────────────────────────────────────────────────
    policies = {
        "Rule-Based": RuleBasedPolicy(),
        "Random": RandomPolicy(rng=np.random.default_rng(42)),
    }

    # ── evaluate ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE EVALUATION — 1000 Fixed Goals")
    print("=" * 60)

    results = {}
    for name, policy in policies.items():
        print(f"\nEvaluating {name} …")
        stats = evaluate_policy(
            policy=policy,
            goals=eval_goals,
            db=db,
            max_turns=max_turns,
            reward_config=reward_config,
            noise_level=0.0,
            seed=42,
        )
        results[name] = stats
        print_results(name, stats)

    # ── plots ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating plots …")
    print("=" * 60)
    make_comparison_plots(results, plot_dir)
    save_results_table(results, processed_dir / "baseline_results.json")

    print(f"\n✅  Step 4 complete!")


if __name__ == "__main__":
    main()

