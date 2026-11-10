#!/usr/bin/env python3
"""
train_supervised.py — Train and evaluate a behaviour-cloning MLP policy.

Pipeline:
  1. Load (state, action) pairs from Step 1.
  2. Train an MLP with class-weighted cross-entropy.
  3. Evaluate on the held-out SL test set (accuracy).
  4. Deploy in the Gym environment and compare with baselines.
  5. Generate learning curves and comparison plots.

Usage:
  cd "5. RL"
  python3 scripts/train_supervised.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── project imports ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    load_config, set_seed,
    SYSTEM_ACTIONS, ACTION_TO_IDX, NUM_SYSTEM_ACTIONS, STATE_DIM,
)
from src.environment import RestaurantBookingEnv
from src.policies.supervised import (
    DialoguePolicyMLP, SupervisedPolicy,
    load_sl_data, compute_class_weights, train_model,
)
from src.policies.rule_based import RuleBasedPolicy, RandomPolicy


# ═══════════════════════════════════════════════════════════════════════════
# Evaluation in the Gym environment
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_in_env(
    policy,
    goals: list,
    db: pd.DataFrame,
    max_turns: int,
    reward_config: dict,
    noise_level: float = 0.0,
    seed: int = 42,
) -> dict:
    """Evaluate a policy on goals in the Gym environment."""
    successes, turns_list, rewards_list = [], [], []

    for i, goal in enumerate(goals):
        env = RestaurantBookingEnv(
            goals=[goal], db=db, noise_level=noise_level,
            max_turns=max_turns, reward_config=reward_config,
            rng=np.random.default_rng(seed + i),
        )
        obs, _ = env.reset(options={"goal": goal})

        ep_reward = 0.0
        for t in range(max_turns):
            action = policy.select_action(obs=obs, env=env)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated:
                break

        successes.append(bool(info.get("success", False)))
        turns_list.append(t + 1)
        rewards_list.append(ep_reward)
        env.close()

    return {
        "success_rate": np.mean(successes),
        "avg_turns": np.mean(turns_list),
        "std_turns": np.std(turns_list),
        "avg_reward": np.mean(rewards_list),
        "std_reward": np.std(rewards_list),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_learning_curves(history: dict, plot_dir: Path) -> None:
    """Plot training and validation loss/accuracy curves."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train", markersize=3)
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val", markersize=3)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train", markersize=3)
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val", markersize=3)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Classification Accuracy")
    axes[1].legend()

    plt.suptitle("Behaviour Cloning — Learning Curves", y=1.02,
                 fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "supervised_learning_curves.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ supervised_learning_curves.png")


def plot_comparison(results: dict, plot_dir: Path) -> None:
    """Bar chart comparing all policies so far."""
    sns.set_theme(style="whitegrid", font_scale=1.15)
    palette = sns.color_palette("viridis", len(results))

    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Success rate
    success = [results[n]["success_rate"] * 100 for n in names]
    bars = axes[0].bar(names, success, color=palette, edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 110)
    for bar, v in zip(bars, success):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                     f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)

    # Turns
    turns = [results[n]["avg_turns"] for n in names]
    stds = [results[n]["std_turns"] for n in names]
    axes[1].bar(names, turns, yerr=stds, color=palette, edgecolor="black",
                capsize=4)
    axes[1].set_ylabel("Avg Turns")
    axes[1].set_title("Dialogue Length")

    # Reward
    rewards = [results[n]["avg_reward"] for n in names]
    rstds = [results[n]["std_reward"] for n in names]
    axes[2].bar(names, rewards, yerr=rstds, color=palette, edgecolor="black",
                capsize=4)
    axes[2].set_ylabel("Avg Reward")
    axes[2].set_title("Total Reward")

    plt.suptitle("Policy Comparison (1000 Goals)", y=1.02,
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(plot_dir / "supervised_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ supervised_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    set_seed(cfg["training"]["seed"])
    device = "cpu"

    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    model_dir = PROJECT_ROOT / "results" / "models"
    plot_dir = PROJECT_ROOT / "results" / "plots"
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    sl_cfg = cfg["training"]["supervised"]
    max_turns = cfg["environment"]["max_turns"]
    reward_config = cfg["environment"]["reward"]

    # ── 1. Load SL data ─────────────────────────────────────────────────
    print("=" * 60)
    print("  Step 5a: Load supervised-learning data")
    print("=" * 60)
    train_ds, val_ds, test_ds = load_sl_data(processed_dir)

    # Class weights
    weights = compute_class_weights(train_ds)
    print(f"\n  Class weights (per action):")
    for i, (name, w) in enumerate(zip(SYSTEM_ACTIONS, weights)):
        marker = "⚠" if w == 0 else " "
        print(f"   {marker} {name:25s}  weight={w:.3f}")

    # ── 2. Build & train model ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 5b: Train behaviour-cloning MLP")
    print("=" * 60)

    model = DialoguePolicyMLP(
        input_dim=STATE_DIM,
        output_dim=NUM_SYSTEM_ACTIONS,
        hidden_dims=sl_cfg["hidden_dims"],
        dropout=0.2,
    )
    print(f"  Model: {model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    history = train_model(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        lr=sl_cfg["lr"],
        batch_size=sl_cfg["batch_size"],
        epochs=sl_cfg["epochs"],
        device=device,
        patience=10,
        class_weights=weights,
    )

    # ── 3. Test set evaluation ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 5c: Test set evaluation (classification)")
    print("=" * 60)

    from src.policies.supervised import eval_epoch
    from torch.utils.data import DataLoader
    import torch.nn as nn

    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  Test accuracy: {test_acc:.3f}")

    # Per-class accuracy
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for states, actions in test_loader:
            logits = model(states.to(device))
            preds = logits.argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(actions)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    print(f"\n  {'Action':<25s}  {'N':>5s}  {'Correct':>7s}  {'Acc':>6s}")
    print(f"  {'─' * 50}")
    for i, name in enumerate(SYSTEM_ACTIONS):
        mask = all_labels == i
        n = mask.sum().item()
        if n == 0:
            print(f"  {name:<25s}  {n:>5d}  {'—':>7s}  {'—':>6s}")
        else:
            correct = ((all_preds == i) & mask).sum().item()
            acc = correct / n
            print(f"  {name:<25s}  {n:>5d}  {correct:>7d}  {acc:>5.1%}")

    # ── 4. Save model ───────────────────────────────────────────────────
    model_path = model_dir / "supervised_policy.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "hidden_dims": sl_cfg["hidden_dims"],
        "history": history,
        "test_acc": test_acc,
    }, model_path)
    print(f"\n  ✓ Model saved to {model_path}")

    # ── 5. Evaluate in Gym environment ──────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 5d: Evaluate in Gym environment (1000 goals)")
    print("=" * 60)

    # Load eval goals
    db = pd.read_csv(processed_dir / "restaurant_db.csv")
    with open(processed_dir / "eval_goals_1000.json") as f:
        eval_goals = json.load(f)

    # Supervised policy
    sl_policy = SupervisedPolicy(model, device=device, greedy=True)

    results = {}

    print("\n  Evaluating Supervised (greedy) …")
    results["Supervised"] = evaluate_in_env(
        sl_policy, eval_goals, db, max_turns, reward_config, seed=42
    )

    # Also evaluate stochastic version
    sl_policy_stoch = SupervisedPolicy(model, device=device, greedy=False)
    print("  Evaluating Supervised (stochastic) …")
    results["Supervised\n(stochastic)"] = evaluate_in_env(
        sl_policy_stoch, eval_goals, db, max_turns, reward_config, seed=42
    )

    # Load baseline results for comparison
    baseline_path = processed_dir / "baseline_results.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        for name, stats in baseline.items():
            results[name] = stats

    # Print results
    print(f"\n  {'Policy':<25s}  {'Success':>8s}  {'Turns':>10s}  {'Reward':>12s}")
    print(f"  {'─' * 60}")
    for name, stats in results.items():
        sr = stats["success_rate"]
        at = stats["avg_turns"]
        st = stats["std_turns"]
        ar = stats["avg_reward"]
        disp_name = name.replace("\n", " ")
        print(f"  {disp_name:<25s}  {sr:>7.1%}  {at:>6.2f}±{st:<4.2f}  {ar:>12.2f}")

    # ── 6. Plots ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating plots …")
    print("=" * 60)
    plot_learning_curves(history, plot_dir)
    plot_comparison(results, plot_dir)

    # Save results
    save_results = {
        k.replace("\n", " "): {
            "success_rate": round(v["success_rate"], 4),
            "avg_turns": round(v["avg_turns"], 2),
            "std_turns": round(v["std_turns"], 2),
            "avg_reward": round(v["avg_reward"], 2),
            "std_reward": round(v.get("std_reward", 0), 2),
        }
        for k, v in results.items()
    }
    with open(processed_dir / "supervised_results.json", "w") as f:
        json.dump(save_results, f, indent=2)

    print(f"\n✅  Step 5 complete!")
    print(f"  Model: {model_path}")
    print(f"  Test accuracy: {test_acc:.1%}")
    print(f"  Env success rate (greedy): {results['Supervised']['success_rate']:.1%}")


if __name__ == "__main__":
    main()

