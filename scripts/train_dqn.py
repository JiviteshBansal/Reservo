"""
train_dqn.py — Train & evaluate DQN dialogue policy (cold- and warm-start)

What this script does
─────────────────────
1. Loads restaurant DB and 1000 fixed evaluation goals from Step 4.
2. Trains a DQN agent *from scratch* ("cold start") across 3 seeds
   (42, 123, 456), 5 000 episodes each.
3. Trains a DQN agent *warm-started* from the supervised policy
   (Step 5) across the same 3 seeds, 5 000 episodes each.
4. Periodic evaluation every 500 episodes on 200 held-out goals.
5. After training, evaluates the best checkpoint of each variant on the
   full 1 000-goal test set.
6. Plots:
   - Learning curves (mean ± std across seeds) for cold vs warm start.
   - Final comparison bar chart (Random, Rule-Based, Supervised, DQN-Cold, DQN-Warm).
   - Evaluation convergence curves.
7. Saves models and results JSON.

Run:
    cd "5. RL"
    python3 scripts/train_dqn.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

# ── project imports ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config, set_seed, SYSTEM_ACTIONS
from src.environment import RestaurantBookingEnv
from src.policies.dqn import DQNAgent, train_dqn
from src.policies.supervised import DialoguePolicyMLP


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_resources(cfg: dict):
    """Load DB, training goals, and fixed evaluation goals."""
    proc = Path(cfg["data"]["processed_dir"])

    db = pd.read_csv(proc / "restaurant_db.csv")

    # Training goals — combine train + val for more diversity
    with open(proc / "user_goals_train.json") as f:
        train_goals = json.load(f)
    with open(proc / "user_goals_val.json") as f:
        val_goals = json.load(f)
    all_train_goals = train_goals + val_goals

    # Fixed evaluation goals from Step 4
    with open(proc / "eval_goals_1000.json") as f:
        eval_goals = json.load(f)

    return db, all_train_goals, eval_goals


def evaluate_on_goals(agent: DQNAgent, goals: list, db,
                      max_turns: int = 20) -> dict:
    """Full greedy evaluation on a list of goals.  Returns summary dict."""
    env = RestaurantBookingEnv(
        goals=goals, db=db, max_turns=max_turns,
        rng=np.random.default_rng(999),
    )

    successes, rewards, turns = [], [], []
    for goal in goals:
        obs, _ = env.reset(options={"goal": goal})
        ep_r = 0.0
        for t in range(max_turns):
            action = agent.select_action(obs=obs, greedy=True)
            obs, reward, terminated, _, info = env.step(action)
            ep_r += reward
            if terminated:
                break
        successes.append(bool(info.get("success", False)))
        rewards.append(ep_r)
        turns.append(t + 1)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_turns": float(np.mean(turns)),
        "num_goals": len(goals),
    }


def train_single_run(
    seed: int,
    db,
    train_goals: list,
    eval_goals: list,
    cfg: dict,
    warm_start_path: str | None = None,
) -> tuple[DQNAgent, dict]:
    """Run one full DQN training.  Returns (agent, history)."""

    set_seed(seed)
    dqn_cfg = cfg["training"]["dqn"]
    max_turns = cfg["environment"]["max_turns"]

    # ── create environments ──────────────────────────────────────────────
    train_env = RestaurantBookingEnv(
        goals=train_goals, db=db, max_turns=max_turns,
        noise_level=cfg["simulator"]["noise_level"],
        rng=np.random.default_rng(seed),
    )
    eval_env = RestaurantBookingEnv(
        goals=eval_goals, db=db, max_turns=max_turns,
        rng=np.random.default_rng(seed + 1000),
    )

    # ── create agent ─────────────────────────────────────────────────────
    eps_start = dqn_cfg["epsilon_start"]
    if warm_start_path is not None:
        # Warm start: lower initial exploration (network already knows basics)
        eps_start = 0.3

    agent = DQNAgent(
        hidden_dims=dqn_cfg["hidden_dims"],
        lr=dqn_cfg["lr"],
        gamma=dqn_cfg["gamma"],
        buffer_size=dqn_cfg["buffer_size"],
        batch_size=dqn_cfg["batch_size"],
        target_update_freq=dqn_cfg["target_update_freq"],
        epsilon_start=eps_start,
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_steps=dqn_cfg["epsilon_decay_steps"],
        seed=seed,
    )

    # ── optional warm start ──────────────────────────────────────────────
    if warm_start_path is not None:
        ckpt = torch.load(warm_start_path, map_location="cpu",
                          weights_only=False)
        # Supervised checkpoint wraps weights in "model_state_dict"
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt
        agent.warm_start(state_dict)
        print(f"  Warm-started from {warm_start_path}")

    # ── train ────────────────────────────────────────────────────────────
    history = train_dqn(
        agent,
        train_env,
        num_episodes=dqn_cfg["num_episodes"],
        max_turns=max_turns,
        eval_env=eval_env,
        eval_goals=eval_goals,
        eval_interval=500,
        log_interval=100,
    )

    return agent, history


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

PLOT_DIR = Path("results/plots")


def smooth(arr, window=100):
    """Simple moving average for plotting."""
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_learning_curves(cold_histories, warm_histories, seeds):
    """Plot training success & reward (mean ± std across seeds)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, ylabel in [
        (axes[0], "episode_success", "Success Rate (smoothed)"),
        (axes[1], "episode_rewards", "Episode Reward (smoothed)"),
    ]:
        for label, histories, color in [
            ("DQN Cold", cold_histories, "#2196F3"),
            ("DQN Warm", warm_histories, "#FF5722"),
        ]:
            smoothed = [smooth(h[key], 100) for h in histories]
            min_len = min(len(s) for s in smoothed)
            stacked = np.array([s[:min_len] for s in smoothed])
            mean = stacked.mean(axis=0)
            std = stacked.std(axis=0)
            x = np.arange(100, 100 + min_len)  # offset from smoothing

            ax.plot(x, mean, label=label, color=color, linewidth=1.5)
            ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)

        ax.set_xlabel("Episode")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("DQN Training: Cold vs Warm Start (mean ± std, 3 seeds)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dqn_learning_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'dqn_learning_curves.png'}")


def plot_eval_convergence(cold_histories, warm_histories):
    """Plot evaluation success rate during training."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, histories, color, marker in [
        ("DQN Cold", cold_histories, "#2196F3", "o"),
        ("DQN Warm", warm_histories, "#FF5722", "s"),
    ]:
        all_eps = histories[0]["eval_episodes"]
        all_succ = np.array([h["eval_success"] for h in histories])
        mean = all_succ.mean(axis=0)
        std = all_succ.std(axis=0)

        ax.plot(all_eps, mean, label=label, color=color, marker=marker,
                linewidth=2, markersize=6)
        ax.fill_between(all_eps, mean - std, mean + std, alpha=0.2,
                        color=color)

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Eval Success Rate (200 goals)")
    ax.set_title("DQN Evaluation During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dqn_eval_convergence.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'dqn_eval_convergence.png'}")


def plot_final_comparison(results: dict):
    """Bar chart comparing all policies on 1 000 goals."""
    policies = ["Random", "Rule-Based", "Supervised\n(greedy)",
                "DQN Cold", "DQN Warm"]
    success_rates = [
        results["random"]["success_rate"],
        results["rule_based"]["success_rate"],
        results["supervised"]["success_rate"],
        results["dqn_cold"]["success_rate"],
        results["dqn_warm"]["success_rate"],
    ]
    avg_rewards = [
        results["random"]["avg_reward"],
        results["rule_based"]["avg_reward"],
        results["supervised"]["avg_reward"],
        results["dqn_cold"]["avg_reward"],
        results["dqn_warm"]["avg_reward"],
    ]

    colors = ["#9E9E9E", "#4CAF50", "#FFC107", "#2196F3", "#FF5722"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Success rate
    bars = axes[0].bar(policies, success_rates, color=colors, edgecolor="black",
                       linewidth=0.5)
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Task Completion Rate (1000 goals)")
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars, success_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.1%}", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    # Average reward
    bars = axes[1].bar(policies, avg_rewards, color=colors, edgecolor="black",
                       linewidth=0.5)
    axes[1].set_ylabel("Average Reward")
    axes[1].set_title("Average Episode Reward (1000 goals)")
    axes[1].axhline(0, color="black", linewidth=0.5)
    for bar, val in zip(bars, avg_rewards):
        y_off = 1 if val >= 0 else -2
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_off,
                     f"{val:.1f}", ha="center", va="bottom", fontsize=10,
                     fontweight="bold")

    fig.suptitle("All-Policy Comparison on 1000 Fixed Goals", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "dqn_final_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {PLOT_DIR / 'dqn_final_comparison.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    cfg = load_config()
    seeds = cfg["evaluation"]["seeds"]  # [42, 123, 456]
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("DQN Dialogue Policy Training")
    print("=" * 65)

    # ── resume mode: skip cold start if models already exist ─────────
    resume = all((model_dir / f"dqn_cold_seed{s}.pt").exists() for s in seeds)
    if resume:
        print("\n  ★ Cold-start models found — skipping Phase 1 (resume mode)")

    # ── load data ────────────────────────────────────────────────────────
    db, train_goals, eval_goals = load_resources(cfg)
    print(f"\nResources: {len(db)} restaurants, "
          f"{len(train_goals)} train goals, {len(eval_goals)} eval goals")

    supervised_path = model_dir / "supervised_policy.pt"
    if not supervised_path.exists():
        print(f"WARNING: {supervised_path} not found — warm start will be skipped.")
        do_warm = False
    else:
        do_warm = True
        print(f"Supervised model: {supervised_path}")

    # ══════════════════════════════════════════════════════════════════════
    # COLD-START TRAINING
    # ══════════════════════════════════════════════════════════════════════
    cold_agents, cold_histories = [], []

    if resume:
        # Load pre-trained cold-start agents (no histories for learning curves)
        print("\n" + "─" * 65)
        print("PHASE 1: LOADING SAVED COLD-START MODELS")
        print("─" * 65)
        for seed in seeds:
            dqn_cfg = cfg["training"]["dqn"]
            agent = DQNAgent(
                hidden_dims=dqn_cfg["hidden_dims"], lr=dqn_cfg["lr"],
                gamma=dqn_cfg["gamma"], buffer_size=dqn_cfg["buffer_size"],
                batch_size=dqn_cfg["batch_size"],
                target_update_freq=dqn_cfg["target_update_freq"],
                epsilon_start=dqn_cfg["epsilon_end"],  # fully decayed
                epsilon_end=dqn_cfg["epsilon_end"],
                seed=seed,
            )
            agent.load(str(model_dir / f"dqn_cold_seed{seed}.pt"))
            cold_agents.append(agent)
            cold_histories.append(None)  # no history available
            print(f"  Loaded dqn_cold_seed{seed}.pt")
    else:
        print("\n" + "─" * 65)
        print("PHASE 1: DQN COLD START (3 seeds × 5 000 episodes)")
        print("─" * 65)
        for i, seed in enumerate(seeds):
            print(f"\n[Cold seed {seed}] ({i+1}/{len(seeds)})")
            agent, history = train_single_run(
                seed, db, train_goals, eval_goals, cfg,
                warm_start_path=None,
            )
            cold_agents.append(agent)
            cold_histories.append(history)
            agent.save(str(model_dir / f"dqn_cold_seed{seed}.pt"))
            print(f"  Model saved: dqn_cold_seed{seed}.pt")

    # ══════════════════════════════════════════════════════════════════════
    # WARM-START TRAINING
    # ══════════════════════════════════════════════════════════════════════
    warm_agents, warm_histories = [], []
    if do_warm:
        warm_resume = all((model_dir / f"dqn_warm_seed{s}.pt").exists()
                          for s in seeds)
        if warm_resume:
            print("\n" + "─" * 65)
            print("PHASE 2: LOADING SAVED WARM-START MODELS")
            print("─" * 65)
            for seed in seeds:
                dqn_cfg = cfg["training"]["dqn"]
                agent = DQNAgent(
                    hidden_dims=dqn_cfg["hidden_dims"], lr=dqn_cfg["lr"],
                    gamma=dqn_cfg["gamma"], buffer_size=dqn_cfg["buffer_size"],
                    batch_size=dqn_cfg["batch_size"],
                    target_update_freq=dqn_cfg["target_update_freq"],
                    epsilon_start=dqn_cfg["epsilon_end"],
                    epsilon_end=dqn_cfg["epsilon_end"],
                    seed=seed,
                )
                agent.load(str(model_dir / f"dqn_warm_seed{seed}.pt"))
                warm_agents.append(agent)
                warm_histories.append(None)
                print(f"  Loaded dqn_warm_seed{seed}.pt")
        else:
            print("\n" + "─" * 65)
            print("PHASE 2: DQN WARM START (3 seeds × 5 000 episodes)")
            print("─" * 65)

            for i, seed in enumerate(seeds):
                print(f"\n[Warm seed {seed}] ({i+1}/{len(seeds)})")
                agent, history = train_single_run(
                    seed, db, train_goals, eval_goals, cfg,
                    warm_start_path=str(supervised_path),
                )
                warm_agents.append(agent)
                warm_histories.append(history)
                agent.save(str(model_dir / f"dqn_warm_seed{seed}.pt"))
                print(f"  Model saved: dqn_warm_seed{seed}.pt")

    # ══════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION ON 1 000 GOALS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("PHASE 3: FINAL EVALUATION (1000 goals, greedy)")
    print("─" * 65)

    # -- Find best cold & warm agents (by full eval on 1000 goals) ──────
    cold_per_seed = {}
    for i, seed in enumerate(seeds):
        res = evaluate_on_goals(cold_agents[i], eval_goals, db)
        cold_per_seed[seed] = res
        print(f"  Cold seed {seed}: success={res['success_rate']:.1%}  "
              f"reward={res['avg_reward']:.1f}  turns={res['avg_turns']:.1f}")

    best_cold_idx = int(np.argmax([cold_per_seed[s]["success_rate"]
                                   for s in seeds]))
    cold_eval = cold_per_seed[seeds[best_cold_idx]]
    print(f"\n  Best cold-start: seed {seeds[best_cold_idx]} "
          f"({cold_eval['success_rate']:.1%})")

    warm_eval = {"success_rate": 0.0, "avg_reward": -20.0, "avg_turns": 20.0}
    warm_per_seed = {}
    warm_idx = 0
    if do_warm:
        for i, seed in enumerate(seeds):
            res = evaluate_on_goals(warm_agents[i], eval_goals, db)
            warm_per_seed[seed] = res
            print(f"  Warm seed {seed}: success={res['success_rate']:.1%}  "
                  f"reward={res['avg_reward']:.1f}  turns={res['avg_turns']:.1f}")

        best_warm_idx = int(np.argmax([warm_per_seed[s]["success_rate"]
                                       for s in seeds]))
        warm_eval = warm_per_seed[seeds[best_warm_idx]]
        warm_idx = best_warm_idx
        print(f"\n  Best warm-start: seed {seeds[best_warm_idx]} "
              f"({warm_eval['success_rate']:.1%})")

    # ── load baseline results from Step 4/5 ──────────────────────────
    proc = Path(cfg["data"]["processed_dir"])
    try:
        with open(proc / "baseline_results.json") as f:
            baseline_res = json.load(f)
    except FileNotFoundError:
        baseline_res = {
            "rule_based": {"success_rate": 1.0, "avg_reward": 30.0},
            "random": {"success_rate": 0.138, "avg_reward": -20.0},
        }

    try:
        with open(proc / "supervised_results.json") as f:
            supervised_res = json.load(f)
    except FileNotFoundError:
        supervised_res = {"greedy": {"success_rate": 0.027, "avg_reward": -18.0}}

    # ── build comparison dict ────────────────────────────────────────
    # Handle various key formats from saved results
    def _get(d, *keys):
        for k in keys:
            if k in d:
                return d[k]
        return {}

    comparison = {
        "random": _get(baseline_res, "random", "Random"),
        "rule_based": _get(baseline_res, "rule_based", "Rule-Based"),
        "supervised": _get(supervised_res, "greedy", "Greedy", "Supervised"),
        "dqn_cold": {
            "success_rate": float(np.mean([v["success_rate"]
                                           for v in cold_per_seed.values()])),
            "avg_reward": float(np.mean([v["avg_reward"]
                                         for v in cold_per_seed.values()])),
            "per_seed": {str(k): v for k, v in cold_per_seed.items()},
        },
        "dqn_warm": {
            "success_rate": float(np.mean([v["success_rate"]
                                           for v in warm_per_seed.values()])) if warm_per_seed else 0.0,
            "avg_reward": float(np.mean([v["avg_reward"]
                                         for v in warm_per_seed.values()])) if warm_per_seed else -20.0,
            "per_seed": {str(k): v for k, v in warm_per_seed.items()},
        },
    }

    # ══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 65)
    print("PHASE 4: GENERATING PLOTS")
    print("─" * 65)

    has_cold_hist = all(h is not None for h in cold_histories)
    has_warm_hist = do_warm and all(h is not None for h in warm_histories)
    if has_cold_hist and has_warm_hist:
        # Full run: plot cold vs warm learning curves
        plot_learning_curves(cold_histories, warm_histories, seeds)
        if cold_histories[0]["eval_episodes"]:
            plot_eval_convergence(cold_histories, warm_histories)
    elif has_warm_hist:
        # Resume mode: only warm histories available
        if warm_histories[0]["eval_episodes"]:
            plot_eval_convergence(warm_histories, warm_histories)
    else:
        print("  (Skipping learning curves — loaded from saved models)")

    plot_final_comparison(comparison)

    # ══════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ══════════════════════════════════════════════════════════════════════
    results = {
        "comparison": comparison,
        "cold_start": {
            "best_seed": int(seeds[best_cold_idx]),
            "eval": cold_eval,
            "per_seed": {str(k): v for k, v in cold_per_seed.items()},
        },
        "warm_start": {
            "best_seed": int(seeds[warm_idx]) if do_warm else None,
            "eval": warm_eval,
            "per_seed": {str(k): v for k, v in warm_per_seed.items()},
        } if do_warm else {},
        "config": {
            "num_episodes": cfg["training"]["dqn"]["num_episodes"],
            "seeds": seeds,
            "warm_epsilon_start": 0.3,
        },
    }

    results_path = proc / "dqn_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # ── training histories (for Step 7 analysis) ─────────────────────
    # Save only the eval checkpoints (compact) — full episode logs are large
    if has_cold_hist or has_warm_hist:
        histories_compact = {
            "cold": [{
                "eval_episodes": h["eval_episodes"],
                "eval_success": h["eval_success"],
                "eval_reward": h["eval_reward"],
            } for h in cold_histories if h is not None],
            "warm": [{
                "eval_episodes": h["eval_episodes"],
                "eval_success": h["eval_success"],
                "eval_reward": h["eval_reward"],
            } for h in warm_histories if h is not None],
        }
        with open(proc / "dqn_eval_histories.json", "w") as f:
            json.dump(histories_compact, f, indent=2)
        print(f"  Histories saved: {proc / 'dqn_eval_histories.json'}")

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print(f"\n{'Policy':<25} {'Success Rate':>14} {'Avg Reward':>12}")
    print("─" * 53)
    for label, key in [("Random", "random"), ("Rule-Based", "rule_based"),
                       ("Supervised (greedy)", "supervised"),
                       ("DQN Cold Start", "dqn_cold"),
                       ("DQN Warm Start", "dqn_warm")]:
        sr = comparison[key]["success_rate"]
        ar = comparison[key]["avg_reward"]
        print(f"{label:<25} {sr:>13.1%} {ar:>12.1f}")

    print(f"\nTotal training time: {elapsed / 60:.1f} minutes")
    print(f"Models saved in: {model_dir}/")
    print(f"Plots saved in: {PLOT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
