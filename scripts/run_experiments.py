"""
run_experiments.py — Ablation studies & analysis

Four experiments:
  (a) Reward Shaping: default rewards vs. no shaping (step penalty only)
  (b) Simulator Noise: train DQN at noise = 0%, 10%, 20%
  (c) Cold vs Warm Start: convergence comparison (uses Step 6 data)
  (d) Goal Complexity: evaluate best DQN on "requests" vs "booking" goals

Reuses Step 6 models wherever possible to save training time.
Only trains ~3 new models (no-shaping, noise-10%, noise-20%).

Run:
    cd "5. RL"
    python3 scripts/run_experiments.py
"""

from __future__ import annotations

import json
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
from src.policies.rule_based import RuleBasedPolicy, RandomPolicy
from src.policies.supervised import SupervisedPolicy

PLOT_DIR = Path("results/plots")
MODEL_DIR = Path("results/models")


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_resources(cfg):
    proc = Path(cfg["data"]["processed_dir"])
    db = pd.read_csv(proc / "restaurant_db.csv")
    with open(proc / "user_goals_train.json") as f:
        train_goals = json.load(f)
    with open(proc / "user_goals_val.json") as f:
        val_goals = json.load(f)
    with open(proc / "eval_goals_1000.json") as f:
        eval_goals = json.load(f)
    return db, train_goals + val_goals, eval_goals


def evaluate_agent(agent, goals, db, max_turns=20):
    """Evaluate a DQN agent on given goals, returns dict with metrics."""
    env = RestaurantBookingEnv(
        goals=goals, db=db, max_turns=max_turns,
        rng=np.random.default_rng(999),
    )
    successes, rewards, turns_list = [], [], []
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
        turns_list.append(t + 1)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_turns": float(np.mean(turns_list)),
        "successes": successes,
        "rewards": rewards,
        "turns": turns_list,
    }


def evaluate_policy(policy, goals, db, max_turns=20):
    """Evaluate a rule-based or supervised policy."""
    env = RestaurantBookingEnv(
        goals=goals, db=db, max_turns=max_turns,
        rng=np.random.default_rng(999),
    )
    successes, rewards, turns_list = [], [], []
    for goal in goals:
        obs, _ = env.reset(options={"goal": goal})
        ep_r = 0.0
        for t in range(max_turns):
            action = policy.select_action(obs=obs, env=env)
            obs, reward, terminated, _, info = env.step(action)
            ep_r += reward
            if terminated:
                break
        successes.append(bool(info.get("success", False)))
        rewards.append(ep_r)
        turns_list.append(t + 1)

    return {
        "success_rate": float(np.mean(successes)),
        "avg_reward": float(np.mean(rewards)),
        "avg_turns": float(np.mean(turns_list)),
        "successes": successes,
        "rewards": rewards,
        "turns": turns_list,
    }


def train_dqn_experiment(db, train_goals, eval_goals, cfg, seed=42,
                          num_episodes=3000, noise_level=0.0,
                          reward_config=None, label=""):
    """Train a single DQN with optional overrides. Returns (agent, history)."""
    set_seed(seed)
    dqn_cfg = cfg["training"]["dqn"]
    max_turns = cfg["environment"]["max_turns"]

    train_env = RestaurantBookingEnv(
        goals=train_goals, db=db, max_turns=max_turns,
        noise_level=noise_level,
        reward_config=reward_config,
        rng=np.random.default_rng(seed),
    )
    eval_env = RestaurantBookingEnv(
        goals=eval_goals, db=db, max_turns=max_turns,
        noise_level=0.0,  # always eval at noise=0
        rng=np.random.default_rng(seed + 1000),
    )

    agent = DQNAgent(
        hidden_dims=dqn_cfg["hidden_dims"],
        lr=dqn_cfg["lr"],
        gamma=dqn_cfg["gamma"],
        buffer_size=dqn_cfg["buffer_size"],
        batch_size=dqn_cfg["batch_size"],
        target_update_freq=dqn_cfg["target_update_freq"],
        epsilon_start=dqn_cfg["epsilon_start"],
        epsilon_end=dqn_cfg["epsilon_end"],
        epsilon_decay_steps=dqn_cfg["epsilon_decay_steps"],
        seed=seed,
    )

    print(f"\n  Training: {label} (seed={seed}, eps={num_episodes}, "
          f"noise={noise_level})")

    history = train_dqn(
        agent, train_env,
        num_episodes=num_episodes,
        max_turns=max_turns,
        eval_env=eval_env,
        eval_goals=eval_goals,
        eval_interval=500,
        log_interval=500,
    )
    return agent, history


def smooth(arr, window=100):
    if len(arr) < window:
        return arr
    return np.convolve(arr, np.ones(window) / window, mode="valid")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment A — Reward Shaping Ablation
# ═══════════════════════════════════════════════════════════════════════════

def experiment_a(db, train_goals, eval_goals, cfg):
    """Compare default reward shaping vs. no shaping."""
    print("\n" + "═" * 65)
    print("EXPERIMENT A: REWARD SHAPING ABLATION")
    print("═" * 65)

    rc = cfg["environment"]["reward"]
    max_turns = cfg["environment"]["max_turns"]

    # ── Default rewards (from config) ────────────────────────────────
    default_config = {
        "step_penalty": rc["step_penalty"],
        "success_multiplier": rc["success_multiplier"],
        "failure_multiplier": rc["failure_multiplier"],
        "redundant_penalty": rc["redundant_penalty"],
        "invalid_penalty": rc["invalid_penalty"],
    }

    # ── No shaping: only step penalty + success/failure ──────────────
    no_shaping_config = {
        "step_penalty": rc["step_penalty"],
        "success_multiplier": rc["success_multiplier"],
        "failure_multiplier": rc["failure_multiplier"],
        "redundant_penalty": 0.0,  # removed
        "invalid_penalty": 0.0,    # removed
    }

    # Check if default model already trained (reuse Step 6)
    default_path = MODEL_DIR / "dqn_cold_seed42.pt"
    default_curve_path = MODEL_DIR / "dqn_default_curve_seed42.pt"
    if default_path.exists():
        print("  Reusing Step 6 cold-start seed 42 for default rewards")
        dqn_cfg = cfg["training"]["dqn"]
        default_agent = DQNAgent(
            hidden_dims=dqn_cfg["hidden_dims"], seed=42,
            epsilon_start=dqn_cfg["epsilon_end"],
            epsilon_end=dqn_cfg["epsilon_end"],
        )
        default_agent.load(str(default_path))
        if default_curve_path.exists():
            print("  Reusing saved default-curve model (skip re-training)")
            default_hist = None
        else:
            # Train a fresh one just for learning curve comparison
            print("  Training default-rewards model for learning curve comparison...")
            _, default_hist = train_dqn_experiment(
                db, train_goals, eval_goals, cfg,
                seed=42, num_episodes=3000, reward_config=default_config,
                label="Default Rewards",
            )
            # Save so we don't re-train on next run
            _.save(str(default_curve_path))
    else:
        default_agent_fresh, default_hist = train_dqn_experiment(
            db, train_goals, eval_goals, cfg,
            seed=42, num_episodes=3000, reward_config=default_config,
            label="Default Rewards",
        )
        default_agent = default_agent_fresh

    # ── Train no-shaping model ───────────────────────────────────────
    noshape_path = MODEL_DIR / "dqn_noshaping_seed42.pt"
    if noshape_path.exists():
        print("  Loading existing no-shaping model")
        dqn_cfg = cfg["training"]["dqn"]
        noshape_agent = DQNAgent(
            hidden_dims=dqn_cfg["hidden_dims"], seed=42,
            epsilon_start=dqn_cfg["epsilon_end"],
            epsilon_end=dqn_cfg["epsilon_end"],
        )
        noshape_agent.load(str(noshape_path))
        noshape_hist = None
    else:
        noshape_agent, noshape_hist = train_dqn_experiment(
            db, train_goals, eval_goals, cfg,
            seed=42, num_episodes=3000, reward_config=no_shaping_config,
            label="No Shaping",
        )
        noshape_agent.save(str(noshape_path))

    # ── Evaluate both ────────────────────────────────────────────────
    # For default, use the Step 6 model (better trained)
    default_eval = evaluate_agent(default_agent, eval_goals, db)
    noshape_eval = evaluate_agent(noshape_agent, eval_goals, db)

    print(f"\n  Default rewards: success={default_eval['success_rate']:.1%}  "
          f"reward={default_eval['avg_reward']:.1f}  "
          f"turns={default_eval['avg_turns']:.1f}")
    print(f"  No shaping:      success={noshape_eval['success_rate']:.1%}  "
          f"reward={noshape_eval['avg_reward']:.1f}  "
          f"turns={noshape_eval['avg_turns']:.1f}")

    # ── Plot ─────────────────────────────────────────────────────────
    if default_hist and noshape_hist:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax, key, ylabel in [
            (axes[0], "episode_success", "Success Rate (smoothed)"),
            (axes[1], "episode_rewards", "Episode Reward (smoothed)"),
        ]:
            for label, hist, color in [
                ("Default (shaped)", default_hist, "#2196F3"),
                ("No shaping", noshape_hist, "#FF9800"),
            ]:
                y = smooth(hist[key], 100)
                x = np.arange(100, 100 + len(y))
                ax.plot(x, y, label=label, color=color, linewidth=1.5)

            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle("Experiment A: Reward Shaping Ablation", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "exp_a_reward_shaping.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {PLOT_DIR / 'exp_a_reward_shaping.png'}")

    # ── Eval convergence comparison ──────────────────────────────────
    if default_hist and noshape_hist:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, hist, color, marker in [
            ("Default (shaped)", default_hist, "#2196F3", "o"),
            ("No shaping", noshape_hist, "#FF9800", "s"),
        ]:
            ax.plot(hist["eval_episodes"], hist["eval_success"],
                    label=label, color=color, marker=marker,
                    linewidth=2, markersize=6)
        ax.set_xlabel("Training Episode")
        ax.set_ylabel("Eval Success Rate (200 goals)")
        ax.set_title("Reward Shaping: Evaluation Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "exp_a_reward_eval.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {PLOT_DIR / 'exp_a_reward_eval.png'}")

    return {
        "default": {k: v for k, v in default_eval.items()
                     if k not in ("successes", "rewards", "turns")},
        "no_shaping": {k: v for k, v in noshape_eval.items()
                        if k not in ("successes", "rewards", "turns")},
    }


# ═══════════════════════════════════════════════════════════════════════════
# Experiment B — Simulator Noise Robustness
# ═══════════════════════════════════════════════════════════════════════════

def experiment_b(db, train_goals, eval_goals, cfg):
    """Train DQN at different noise levels, evaluate at noise=0."""
    print("\n" + "═" * 65)
    print("EXPERIMENT B: SIMULATOR NOISE ROBUSTNESS")
    print("═" * 65)

    noise_levels = [0.0, 0.1, 0.2]
    results = {}
    agents = {}
    histories = {}

    for noise in noise_levels:
        tag = f"noise{int(noise * 100)}"
        model_path = MODEL_DIR / f"dqn_{tag}_seed42.pt"

        if noise == 0.0 and (MODEL_DIR / "dqn_cold_seed42.pt").exists():
            # Reuse Step 6 model for noise=0
            print(f"\n  noise={noise:.0%}: Reusing Step 6 cold-start seed 42")
            dqn_cfg = cfg["training"]["dqn"]
            agent = DQNAgent(
                hidden_dims=dqn_cfg["hidden_dims"], seed=42,
                epsilon_start=dqn_cfg["epsilon_end"],
                epsilon_end=dqn_cfg["epsilon_end"],
            )
            agent.load(str(MODEL_DIR / "dqn_cold_seed42.pt"))
            agents[noise] = agent
            histories[noise] = None
        elif model_path.exists():
            print(f"\n  noise={noise:.0%}: Loading existing model")
            dqn_cfg = cfg["training"]["dqn"]
            agent = DQNAgent(
                hidden_dims=dqn_cfg["hidden_dims"], seed=42,
                epsilon_start=dqn_cfg["epsilon_end"],
                epsilon_end=dqn_cfg["epsilon_end"],
            )
            agent.load(str(model_path))
            agents[noise] = agent
            histories[noise] = None
        else:
            agent, hist = train_dqn_experiment(
                db, train_goals, eval_goals, cfg,
                seed=42, num_episodes=3000, noise_level=noise,
                label=f"Noise {noise:.0%}",
            )
            agents[noise] = agent
            histories[noise] = hist
            agent.save(str(model_path))

        # Evaluate at noise=0 (standard test)
        ev = evaluate_agent(agents[noise], eval_goals, db)
        results[f"noise_{int(noise*100)}pct"] = {
            k: v for k, v in ev.items()
            if k not in ("successes", "rewards", "turns")
        }
        print(f"  noise={noise:.0%} → eval: success={ev['success_rate']:.1%}  "
              f"reward={ev['avg_reward']:.1f}  turns={ev['avg_turns']:.1f}")

    # ── Also evaluate noisy-trained models on noisy test ─────────────
    print("\n  Cross-evaluation (train noise → test noise):")
    cross_eval = {}
    for train_noise in noise_levels:
        for test_noise in [0.0, 0.1, 0.2]:
            test_env_goals = eval_goals
            env = RestaurantBookingEnv(
                goals=test_env_goals, db=db,
                max_turns=cfg["environment"]["max_turns"],
                noise_level=test_noise,
                rng=np.random.default_rng(999),
            )
            successes = []
            for goal in eval_goals[:200]:  # subset for speed
                obs, _ = env.reset(options={"goal": goal})
                for t in range(20):
                    action = agents[train_noise].select_action(obs=obs, greedy=True)
                    obs, _, terminated, _, info = env.step(action)
                    if terminated:
                        break
                successes.append(bool(info.get("success", False)))
            sr = float(np.mean(successes))
            key = f"train{int(train_noise*100)}_test{int(test_noise*100)}"
            cross_eval[key] = sr

    # Print cross-eval table
    print(f"\n  {'Train \\ Test':<15} {'0%':>8} {'10%':>8} {'20%':>8}")
    print("  " + "─" * 41)
    for tn in noise_levels:
        row = f"  {f'{int(tn*100)}%':<15}"
        for te in [0.0, 0.1, 0.2]:
            key = f"train{int(tn*100)}_test{int(te*100)}"
            row += f" {cross_eval[key]:>7.1%}"
        print(row)

    results["cross_eval"] = cross_eval

    # ── Plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: performance at noise=0 test
    noise_labels = ["0%", "10%", "20%"]
    colors_noise = ["#4CAF50", "#FF9800", "#F44336"]
    sr_vals = [results[f"noise_{int(n*100)}pct"]["success_rate"]
               for n in noise_levels]
    rw_vals = [results[f"noise_{int(n*100)}pct"]["avg_reward"]
               for n in noise_levels]

    bars = axes[0].bar(noise_labels, sr_vals, color=colors_noise,
                       edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Training Noise Level")
    axes[0].set_ylabel("Success Rate (test at 0% noise)")
    axes[0].set_title("Effect of Training Noise on Performance")
    axes[0].set_ylim(0, 1.1)
    for bar, val in zip(bars, sr_vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{val:.1%}", ha="center", fontweight="bold")

    # Heatmap: cross-evaluation
    cross_matrix = np.zeros((3, 3))
    for i, tn in enumerate(noise_levels):
        for j, te in enumerate([0.0, 0.1, 0.2]):
            key = f"train{int(tn*100)}_test{int(te*100)}"
            cross_matrix[i, j] = cross_eval[key]

    im = axes[1].imshow(cross_matrix, cmap="RdYlGn", vmin=0, vmax=1,
                        aspect="auto")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["0%", "10%", "20%"])
    axes[1].set_yticks([0, 1, 2])
    axes[1].set_yticklabels(["0%", "10%", "20%"])
    axes[1].set_xlabel("Test Noise Level")
    axes[1].set_ylabel("Training Noise Level")
    axes[1].set_title("Cross-Evaluation Success Rate")
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f"{cross_matrix[i, j]:.1%}",
                         ha="center", va="center", fontsize=12,
                         fontweight="bold",
                         color="white" if cross_matrix[i, j] < 0.5 else "black")
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle("Experiment B: Simulator Noise Robustness", fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp_b_noise.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {PLOT_DIR / 'exp_b_noise.png'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment C — Cold vs Warm Start (formal analysis)
# ═══════════════════════════════════════════════════════════════════════════

def experiment_c(db, eval_goals, cfg):
    """Formal comparison of cold vs warm start using Step 6 models."""
    print("\n" + "═" * 65)
    print("EXPERIMENT C: COLD vs WARM START ANALYSIS")
    print("═" * 65)

    seeds = cfg["evaluation"]["seeds"]
    dqn_cfg = cfg["training"]["dqn"]
    results = {"cold": {}, "warm": {}}

    for variant in ["cold", "warm"]:
        all_successes, all_rewards, all_turns = [], [], []
        for seed in seeds:
            path = MODEL_DIR / f"dqn_{variant}_seed{seed}.pt"
            agent = DQNAgent(
                hidden_dims=dqn_cfg["hidden_dims"], seed=seed,
                epsilon_start=dqn_cfg["epsilon_end"],
                epsilon_end=dqn_cfg["epsilon_end"],
            )
            agent.load(str(path))
            ev = evaluate_agent(agent, eval_goals, db)
            results[variant][str(seed)] = {
                "success_rate": ev["success_rate"],
                "avg_reward": ev["avg_reward"],
                "avg_turns": ev["avg_turns"],
            }
            all_successes.append(ev["success_rate"])
            all_rewards.append(ev["avg_reward"])
            all_turns.append(ev["avg_turns"])
            print(f"  {variant} seed {seed}: success={ev['success_rate']:.1%}  "
                  f"reward={ev['avg_reward']:.1f}  turns={ev['avg_turns']:.1f}")

        results[variant]["mean"] = {
            "success_rate": float(np.mean(all_successes)),
            "avg_reward": float(np.mean(all_rewards)),
            "avg_turns": float(np.mean(all_turns)),
            "std_reward": float(np.std(all_rewards)),
            "std_turns": float(np.std(all_turns)),
        }

    # ── Plot: per-seed comparison ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(seeds))
    width = 0.35

    for ax, metric, ylabel in [
        (axes[0], "avg_reward", "Average Reward"),
        (axes[1], "avg_turns", "Average Turns"),
    ]:
        cold_vals = [results["cold"][str(s)][metric] for s in seeds]
        warm_vals = [results["warm"][str(s)][metric] for s in seeds]

        bars1 = ax.bar(x - width/2, cold_vals, width, label="Cold Start",
                       color="#2196F3", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x + width/2, warm_vals, width, label="Warm Start",
                       color="#FF5722", edgecolor="black", linewidth=0.5)

        ax.set_xlabel("Seed")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seeds])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f"{bar.get_height():.1f}", ha="center",
                        fontsize=9, fontweight="bold")

    fig.suptitle("Experiment C: Cold vs Warm Start (per seed, 1000 goals)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp_c_cold_warm.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {PLOT_DIR / 'exp_c_cold_warm.png'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment D — Goal Complexity Breakdown
# ═══════════════════════════════════════════════════════════════════════════

def experiment_d(db, eval_goals, cfg):
    """Evaluate all policies on different goal types."""
    print("\n" + "═" * 65)
    print("EXPERIMENT D: GOAL COMPLEXITY BREAKDOWN")
    print("═" * 65)

    # ── Categorise goals ─────────────────────────────────────────────
    # Booking: non-empty "book" dict means booking is required
    request_goals = [g for g in eval_goals if not g.get("book")]
    booking_goals = [g for g in eval_goals if g.get("book")]

    # Split by number of informable constraints (in "info" dict)
    simple_goals = [g for g in eval_goals
                    if len(g.get("info", {})) <= 1]
    medium_goals = [g for g in eval_goals
                    if len(g.get("info", {})) == 2]
    complex_goals = [g for g in eval_goals
                     if len(g.get("info", {})) >= 3]

    categories = {
        "All (1000)": eval_goals,
        "Requests only": request_goals,
        "Booking required": booking_goals,
    }
    if simple_goals:
        categories["Simple (≤1 constraint)"] = simple_goals
    if medium_goals:
        categories["Medium (2 constraints)"] = medium_goals
    if complex_goals:
        categories["Complex (3 constraints)"] = complex_goals

    print(f"  Goal breakdown:")
    for name, goals in categories.items():
        print(f"    {name}: {len(goals)} goals")

    # ── Load policies ────────────────────────────────────────────────
    dqn_cfg = cfg["training"]["dqn"]

    # Best DQN (cold, seed 456 had highest reward)
    best_dqn = DQNAgent(
        hidden_dims=dqn_cfg["hidden_dims"], seed=456,
        epsilon_start=dqn_cfg["epsilon_end"],
        epsilon_end=dqn_cfg["epsilon_end"],
    )
    best_dqn.load(str(MODEL_DIR / "dqn_cold_seed456.pt"))

    # Best DQN warm
    best_warm = DQNAgent(
        hidden_dims=dqn_cfg["hidden_dims"], seed=123,
        epsilon_start=dqn_cfg["epsilon_end"],
        epsilon_end=dqn_cfg["epsilon_end"],
    )
    best_warm.load(str(MODEL_DIR / "dqn_warm_seed123.pt"))

    rule_based = RuleBasedPolicy()
    random_pol = RandomPolicy(rng=np.random.default_rng(42))

    policies = {
        "Random": ("policy", random_pol),
        "Rule-Based": ("policy", rule_based),
        "DQN Cold": ("agent", best_dqn),
        "DQN Warm": ("agent", best_warm),
    }

    # ── Evaluate ─────────────────────────────────────────────────────
    results = {}
    for cat_name, goals in categories.items():
        if len(goals) == 0:
            continue
        results[cat_name] = {}
        print(f"\n  {cat_name} ({len(goals)} goals):")
        for pol_name, (pol_type, pol) in policies.items():
            if pol_type == "agent":
                ev = evaluate_agent(pol, goals, db)
            else:
                ev = evaluate_policy(pol, goals, db)
            results[cat_name][pol_name] = {
                "success_rate": ev["success_rate"],
                "avg_reward": ev["avg_reward"],
                "avg_turns": ev["avg_turns"],
            }
            print(f"    {pol_name:<15} success={ev['success_rate']:.1%}  "
                  f"reward={ev['avg_reward']:.1f}  turns={ev['avg_turns']:.1f}")

    # ── Plot: grouped bar chart ──────────────────────────────────────
    # Focus on Request vs Booking breakdown
    if request_goals and booking_goals:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        pol_names = ["Random", "Rule-Based", "DQN Cold", "DQN Warm"]
        colors_pol = ["#9E9E9E", "#4CAF50", "#2196F3", "#FF5722"]
        x = np.arange(len(pol_names))
        width = 0.35

        for ax, cat1, cat2, ylabel in [
            (axes[0], "Requests only", "Booking required", "Success Rate"),
        ]:
            v1 = [results[cat1][p]["success_rate"] for p in pol_names]
            v2 = [results[cat2][p]["success_rate"] for p in pol_names]

            bars1 = ax.bar(x - width/2, v1, width,
                           label=f"Requests ({len(request_goals)})",
                           color=[c + "88" for c in colors_pol],
                           edgecolor="black", linewidth=0.5)
            bars2 = ax.bar(x + width/2, v2, width,
                           label=f"Booking ({len(booking_goals)})",
                           color=colors_pol,
                           edgecolor="black", linewidth=0.5)

            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(pol_names)
            ax.legend()
            ax.set_ylim(0, 1.15)
            ax.grid(True, alpha=0.3, axis="y")

            for bars in [bars1, bars2]:
                for bar in bars:
                    if bar.get_height() > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.02,
                                f"{bar.get_height():.0%}", ha="center",
                                fontsize=8, fontweight="bold")

        # Turns comparison
        ax2 = axes[1]
        v1 = [results["Requests only"][p]["avg_turns"] for p in pol_names]
        v2 = [results["Booking required"][p]["avg_turns"] for p in pol_names]

        bars1 = ax2.bar(x - width/2, v1, width,
                        label=f"Requests ({len(request_goals)})",
                        color=[c + "88" for c in colors_pol],
                        edgecolor="black", linewidth=0.5)
        bars2 = ax2.bar(x + width/2, v2, width,
                        label=f"Booking ({len(booking_goals)})",
                        color=colors_pol,
                        edgecolor="black", linewidth=0.5)
        ax2.set_ylabel("Average Turns")
        ax2.set_xticks(x)
        ax2.set_xticklabels(pol_names)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.1,
                         f"{bar.get_height():.1f}", ha="center",
                         fontsize=8, fontweight="bold")

        fig.suptitle("Experiment D: Goal Complexity Breakdown", fontsize=13)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "exp_d_complexity.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved {PLOT_DIR / 'exp_d_complexity.png'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Summary Table Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_summary_table(all_results):
    """Create a publication-quality summary figure."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    # Table data
    headers = ["Experiment", "Condition", "Success Rate", "Avg Reward",
               "Avg Turns"]
    rows = []

    # Exp A
    for cond, key in [("Default (shaped)", "default"),
                      ("No shaping", "no_shaping")]:
        r = all_results["exp_a"].get(key, {})
        rows.append(["A: Reward Shaping", cond,
                     f"{r.get('success_rate', 0):.1%}",
                     f"{r.get('avg_reward', 0):.1f}",
                     f"{r.get('avg_turns', 0):.1f}"])

    # Exp B
    for noise_pct in [0, 10, 20]:
        key = f"noise_{noise_pct}pct"
        r = all_results["exp_b"].get(key, {})
        rows.append(["B: Noise", f"Train {noise_pct}%",
                     f"{r.get('success_rate', 0):.1%}",
                     f"{r.get('avg_reward', 0):.1f}",
                     f"{r.get('avg_turns', 0):.1f}"])

    # Exp C
    for variant in ["cold", "warm"]:
        m = all_results["exp_c"].get(variant, {}).get("mean", {})
        rows.append(["C: Init", variant.title() + " Start",
                     f"{m.get('success_rate', 0):.1%}",
                     f"{m.get('avg_reward', 0):.1f} ± {m.get('std_reward', 0):.1f}",
                     f"{m.get('avg_turns', 0):.1f} ± {m.get('std_turns', 0):.1f}"])

    table = ax.table(cellText=rows, colLabels=headers, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(len(headers)):
        table[0, j].set_facecolor("#2196F3")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colours
    for i in range(1, len(rows) + 1):
        color = "#F5F5F5" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i, j].set_facecolor(color)

    fig.suptitle("Summary of Ablation Studies", fontsize=13, y=0.98)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "exp_summary_table.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved {PLOT_DIR / 'exp_summary_table.png'}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    cfg = load_config()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Ablation Studies")
    print("=" * 65)

    db, train_goals, eval_goals = load_resources(cfg)
    print(f"\nResources: {len(db)} restaurants, "
          f"{len(train_goals)} train goals, {len(eval_goals)} eval goals")

    all_results = {}

    # ── Run experiments ──────────────────────────────────────────────
    all_results["exp_a"] = experiment_a(db, train_goals, eval_goals, cfg)
    all_results["exp_b"] = experiment_b(db, train_goals, eval_goals, cfg)
    all_results["exp_c"] = experiment_c(db, eval_goals, cfg)
    all_results["exp_d"] = experiment_d(db, eval_goals, cfg)

    # ── Summary table ────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("GENERATING SUMMARY")
    print("═" * 65)
    plot_summary_table(all_results)

    # ── Save all results ─────────────────────────────────────────────
    proc = Path(cfg["data"]["processed_dir"])
    results_path = proc / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved: {results_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f"All done — {elapsed / 60:.1f} minutes")
    print(f"{'=' * 65}")
    print(f"Plots saved in: {PLOT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()

