#!/usr/bin/env python3
"""
test_simulator.py — Test the agenda-based user simulator.

Tests:
  1. Print 3 sample dialogues with a hand-coded scripted policy.
  2. Batch evaluation: 1000 dialogues with the scripted policy.
  3. Batch evaluation: 1000 dialogues with a random policy.
  4. Noise-level sweep: measure success rate at 0%, 10%, 20%, 30%.
  5. Generate summary plots.

Usage:
  cd "5. RL"
  python3 scripts/test_simulator.py
"""

import sys
import json
from pathlib import Path
from collections import Counter

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
    INFORMABLE_SLOTS, REQUESTABLE_SLOTS, BOOKING_SLOTS,
)
from src.data_loader import query_db
from src.user_simulator import AgendaBasedSimulator, run_dialogue


# ═══════════════════════════════════════════════════════════════════════════
# Scripted policy (simple hand-coded rules for testing)
# ═══════════════════════════════════════════════════════════════════════════

def scripted_policy(state: dict) -> int:
    """
    A simple rule-based policy for testing the simulator.

    Strategy:
      1. Collect all informable slots (request if missing).
      2. Once enough info gathered, offer a venue.
      3. If user has pending requests, inform them.
      4. If booking needed, ask for booking details then book.
      5. Say bye.

    This is a *testing* policy — Step 4 will build a proper rule-based one.
    """
    filled = state["filled_slots"]
    db_count = state["db_match_count"]
    pending = state["pending_requests"]
    offer_made = state["offer_made"]
    is_booked = state["booked"]
    user_act = state["user_action"]

    # 1. If an informable slot is missing, request it
    for slot in INFORMABLE_SLOTS:
        if slot not in filled or not filled[slot]:
            return ACTION_TO_IDX[f"request_{slot}"]

    # 2. If we haven't offered a venue yet, offer one
    if not offer_made:
        return ACTION_TO_IDX["offer_venue"]

    # 3. If user has pending info requests, fulfil them
    for slot in REQUESTABLE_SLOTS:
        if slot in pending:
            return ACTION_TO_IDX[f"inform_{slot}"]

    # 4. If user says bye, say bye
    if user_act == "bye":
        return ACTION_TO_IDX["bye"]

    # 5. Booking flow: ask for each booking slot, then book
    # Check if user has provided booking info by looking at their slots
    venue = state.get("offered_venue", {})
    if venue and not is_booked:
        # Try requesting booking details, then book
        # We don't track booking slots in this simple state, so just book
        # after offering
        if state["turn"] > 0:
            return ACTION_TO_IDX["book"]
        return ACTION_TO_IDX["request_people"]

    # 6. Default: say bye
    return ACTION_TO_IDX["bye"]


def scripted_policy_with_booking(state: dict) -> int:
    """
    Enhanced scripted policy that handles the booking flow properly.

    Keeps track of booking stage via state["turn"] as a rough proxy.
    """
    filled = state["filled_slots"]
    db_count = state["db_match_count"]
    pending = state["pending_requests"]
    offer_made = state["offer_made"]
    is_booked = state["booked"]
    user_act = state["user_action"]
    turn = state["turn"]

    # If user says bye, reciprocate
    if user_act == "bye":
        return ACTION_TO_IDX["bye"]

    # 1. Collect informable slots
    for slot in INFORMABLE_SLOTS:
        if slot not in filled or not filled[slot]:
            return ACTION_TO_IDX[f"request_{slot}"]

    # 2. Offer venue
    if not offer_made:
        return ACTION_TO_IDX["offer_venue"]

    # 3. Fulfil pending info requests
    for slot in REQUESTABLE_SLOTS:
        if slot in pending:
            return ACTION_TO_IDX[f"inform_{slot}"]

    # 4. Handle booking
    if not is_booked and offer_made:
        # Ask for booking details in sequence
        # We use user's last action to determine what's been provided
        if user_act == "accept" or user_act.startswith("inform_") or user_act.startswith("request_"):
            # Cycle through booking slot requests
            booking_request_sequence = ["request_people", "request_day", "request_time"]
            # Use turn number as a rough stage indicator
            booking_stage = max(0, turn - 3)  # subtract turns used for info gathering
            if booking_stage < len(booking_request_sequence):
                return ACTION_TO_IDX[booking_request_sequence[booking_stage]]
            else:
                return ACTION_TO_IDX["book"]

    # 5. Bye
    return ACTION_TO_IDX["bye"]


def random_policy(state: dict) -> int:
    """Uniformly random action selection."""
    return np.random.randint(NUM_SYSTEM_ACTIONS)


# ═══════════════════════════════════════════════════════════════════════════
# Batch evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_policy(
    policy_fn,
    simulator: AgendaBasedSimulator,
    db: pd.DataFrame,
    goals: list,
    n_episodes: int = 1000,
    max_turns: int = 20,
    verbose_episodes: int = 0,
) -> dict:
    """
    Evaluate a policy over many episodes.

    Returns dict with aggregate statistics.
    """
    results = {"success": [], "turns": [], "reward": []}

    # Use a fixed subset of goals for reproducibility
    test_goals = goals[:n_episodes] if len(goals) >= n_episodes else goals

    for i, goal in enumerate(test_goals):
        verbose = (i < verbose_episodes)
        result = run_dialogue(
            simulator=simulator,
            policy_fn=policy_fn,
            db=db,
            goal=goal,
            max_turns=max_turns,
            verbose=verbose,
        )
        results["success"].append(result["success"])
        results["turns"].append(result["turns"])
        results["reward"].append(result["reward"])

    # Aggregate
    n = len(results["success"])
    stats = {
        "n_episodes": n,
        "success_rate": np.mean(results["success"]),
        "avg_turns": np.mean(results["turns"]),
        "std_turns": np.std(results["turns"]),
        "avg_reward": np.mean(results["reward"]),
        "std_reward": np.std(results["reward"]),
        "success_list": results["success"],
        "turns_list": results["turns"],
        "reward_list": results["reward"],
    }
    return stats


def print_eval_results(name: str, stats: dict) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Episodes:      {stats['n_episodes']}")
    print(f"  Success rate:  {stats['success_rate']:.1%}")
    print(f"  Avg turns:     {stats['avg_turns']:.2f} ± {stats['std_turns']:.2f}")
    print(f"  Avg reward:    {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")


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

    # ── load data ───────────────────────────────────────────────────────
    print("Loading processed data …")
    db = pd.read_csv(processed_dir / "restaurant_db.csv")
    with open(processed_dir / "user_goals_train.json") as f:
        train_goals = json.load(f)
    with open(processed_dir / "user_goals_test.json") as f:
        test_goals = json.load(f)

    print(f"  DB: {len(db)} restaurants")
    print(f"  Train goals: {len(train_goals)},  Test goals: {len(test_goals)}")

    # ── create simulator ────────────────────────────────────────────────
    rng = np.random.default_rng(cfg["training"]["seed"])
    sim = AgendaBasedSimulator(
        goals=train_goals,
        db=db,
        noise_level=0.0,
        max_turns=max_turns,
        rng=rng,
    )

    # ═════════════════════════════════════════════════════════════════════
    # Test 1: Print 3 sample dialogues with scripted policy
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 1: Sample dialogues with scripted policy")
    print("=" * 60)

    # Pick 3 goals with different complexities
    sample_goals = []
    # Simple: only info constraints, no booking, no requests
    for g in test_goals:
        if g["info"] and not g["book"] and not g["reqt"]:
            sample_goals.append(g)
            break
    # Medium: info + requests, no booking
    for g in test_goals:
        if g["info"] and g["reqt"] and not g["book"]:
            sample_goals.append(g)
            break
    # Complex: info + requests + booking
    for g in test_goals:
        if g["info"] and g["reqt"] and g["book"]:
            sample_goals.append(g)
            break

    # Fall back to first 3 if filtering didn't find enough
    while len(sample_goals) < 3 and len(test_goals) > len(sample_goals):
        sample_goals.append(test_goals[len(sample_goals)])

    for i, goal in enumerate(sample_goals):
        complexity = ["Simple", "Medium", "Complex"][i] if i < 3 else f"Goal {i}"
        print(f"\n{'▸'*3} Dialogue {i+1} ({complexity}) {'◂'*3}")
        sim_test = AgendaBasedSimulator(
            goals=[goal], db=db, noise_level=0.0,
            max_turns=max_turns, rng=np.random.default_rng(42),
        )
        run_dialogue(sim_test, scripted_policy_with_booking, db,
                     goal=goal, max_turns=max_turns, verbose=True)

    # ═════════════════════════════════════════════════════════════════════
    # Test 2: Batch evaluation — scripted policy (no noise)
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 2: Batch evaluation")
    print("=" * 60)

    n_eval = min(len(test_goals), 437)

    sim_eval = AgendaBasedSimulator(
        goals=test_goals, db=db, noise_level=0.0,
        max_turns=max_turns, rng=np.random.default_rng(42),
    )

    scripted_stats = evaluate_policy(
        scripted_policy_with_booking, sim_eval, db, test_goals, n_eval, max_turns
    )
    print_eval_results("Scripted Policy (noise=0%)", scripted_stats)

    # Random policy
    set_seed(42)
    sim_rand = AgendaBasedSimulator(
        goals=test_goals, db=db, noise_level=0.0,
        max_turns=max_turns, rng=np.random.default_rng(42),
    )
    random_stats = evaluate_policy(
        random_policy, sim_rand, db, test_goals, n_eval, max_turns
    )
    print_eval_results("Random Policy (noise=0%)", random_stats)

    # ═════════════════════════════════════════════════════════════════════
    # Test 3: Noise-level sweep
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 3: Noise-level sweep (scripted policy)")
    print("=" * 60)

    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    noise_results = {}

    for noise in noise_levels:
        sim_noisy = AgendaBasedSimulator(
            goals=test_goals, db=db, noise_level=noise,
            max_turns=max_turns, rng=np.random.default_rng(42),
        )
        stats = evaluate_policy(
            scripted_policy_with_booking, sim_noisy, db, test_goals,
            n_eval, max_turns,
        )
        noise_results[noise] = stats
        print_eval_results(f"Scripted Policy (noise={noise:.0%})", stats)

    # ═════════════════════════════════════════════════════════════════════
    # Test 4: Goal complexity breakdown
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TEST 4: Success rate by goal complexity")
    print("=" * 60)

    # Categorise goals
    simple_goals = [g for g in test_goals if not g["book"] and not g["reqt"]]
    medium_goals = [g for g in test_goals if g["reqt"] and not g["book"]]
    complex_goals = [g for g in test_goals if g["book"]]

    for name, goals_subset in [("Simple (info only)", simple_goals),
                               ("Medium (info+reqt)", medium_goals),
                               ("Complex (info+book)", complex_goals)]:
        if not goals_subset:
            print(f"  {name}: no goals in this category")
            continue
        sim_cat = AgendaBasedSimulator(
            goals=goals_subset, db=db, noise_level=0.0,
            max_turns=max_turns, rng=np.random.default_rng(42),
        )
        stats = evaluate_policy(
            scripted_policy_with_booking, sim_cat, db, goals_subset,
            min(len(goals_subset), 500), max_turns,
        )
        print_eval_results(f"{name}  (n={len(goals_subset)})", stats)

    # ═════════════════════════════════════════════════════════════════════
    # Plots
    # ═════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  Generating plots …")
    print("=" * 60)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── Plot 1: Noise-level impact ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    x = [f"{n:.0%}" for n in noise_levels]
    success_rates = [noise_results[n]["success_rate"] * 100 for n in noise_levels]
    avg_turns_list = [noise_results[n]["avg_turns"] for n in noise_levels]
    avg_rewards = [noise_results[n]["avg_reward"] for n in noise_levels]

    axes[0].bar(x, success_rates, color=sns.color_palette("viridis", len(x)),
                edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_xlabel("Noise Level")
    axes[0].set_title("Success vs Noise")
    axes[0].set_ylim(0, 100)

    axes[1].bar(x, avg_turns_list, color=sns.color_palette("mako", len(x)),
                edgecolor="black")
    axes[1].set_ylabel("Avg Turns")
    axes[1].set_xlabel("Noise Level")
    axes[1].set_title("Turns vs Noise")

    axes[2].bar(x, avg_rewards, color=sns.color_palette("rocket", len(x)),
                edgecolor="black")
    axes[2].set_ylabel("Avg Reward")
    axes[2].set_xlabel("Noise Level")
    axes[2].set_title("Reward vs Noise")

    plt.suptitle("Simulator Noise-Level Analysis (Scripted Policy)", y=1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "simulator_noise_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ simulator_noise_sweep.png")

    # ── Plot 2: Turn distribution (scripted vs random) ──────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(scripted_stats["turns_list"], bins=range(1, max_turns + 2),
                 edgecolor="black", alpha=0.8, color=sns.color_palette("viridis")[2],
                 label="Scripted")
    axes[0].hist(random_stats["turns_list"], bins=range(1, max_turns + 2),
                 edgecolor="black", alpha=0.5, color=sns.color_palette("viridis")[4],
                 label="Random")
    axes[0].set_xlabel("Dialogue Length (turns)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Turn Distribution")
    axes[0].legend()

    axes[1].hist(scripted_stats["reward_list"], bins=30, edgecolor="black",
                 alpha=0.8, color=sns.color_palette("mako")[2], label="Scripted")
    axes[1].hist(random_stats["reward_list"], bins=30, edgecolor="black",
                 alpha=0.5, color=sns.color_palette("mako")[4], label="Random")
    axes[1].set_xlabel("Episode Reward")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Reward Distribution")
    axes[1].legend()

    plt.suptitle("Scripted vs Random Policy", y=1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "simulator_policy_comparison.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ simulator_policy_comparison.png")

    print(f"\n✅  Step 2 complete!  Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()

