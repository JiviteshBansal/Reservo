#!/usr/bin/env python3
"""
test_environment.py — Test the Gymnasium environment.

Tests:
  1. Smoke test: reset, step, observation shape & range.
  2. Sample episodes with a state-dict scripted policy.
  3. Sample episodes with random policy.
  4. Reward structure validation (check penalties fire correctly).
  5. Batch evaluation & statistics.

Usage:
  cd "5. RL"
  python3 scripts/test_environment.py
"""

import sys
import json
from pathlib import Path

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


# ═══════════════════════════════════════════════════════════════════════════
# Policies that work with the Gym environment
# ═══════════════════════════════════════════════════════════════════════════

def rule_based_policy(env: RestaurantBookingEnv) -> int:
    """
    A rule-based policy that uses env.get_state_info() for decisions.

    Strategy (same logical flow as the scripted policy in Step 2):
      1. Gather all informable slots.
      2. Offer a venue once we have enough info (≥1 DB match).
      3. Fulfil pending info requests.
      4. Handle booking flow.
      5. Say bye.
    """
    s = env.get_state_info()

    filled = s["filled_slots"]
    db_count = s["db_match_count"]
    pending = s["pending_requests"]
    offer_made = s["offer_made"]
    is_booked = s["booked"]
    user_act = s["user_action"]
    booking = s["booking_state"]

    # If user says bye, reciprocate
    if user_act == "bye":
        return ACTION_TO_IDX["bye"]

    # 1. Collect all informable slots
    for slot in INFORMABLE_SLOTS:
        if not filled.get(slot, ""):
            return ACTION_TO_IDX[f"request_{slot}"]

    # 2. Offer a venue (only once)
    if not offer_made:
        return ACTION_TO_IDX["offer_venue"]

    # 3. Fulfil pending info requests
    for slot in REQUESTABLE_SLOTS:
        if slot in pending:
            return ACTION_TO_IDX[f"inform_{slot}"]

    # 4. Booking flow
    if not is_booked and offer_made:
        # Ask for each booking slot we don't have
        for slot in BOOKING_SLOTS:
            if not booking.get(slot, ""):
                return ACTION_TO_IDX[f"request_{slot}"]
        # All booking slots collected → book
        return ACTION_TO_IDX["book"]

    # 5. Default: bye
    return ACTION_TO_IDX["bye"]


# ═══════════════════════════════════════════════════════════════════════════
# Test functions
# ═══════════════════════════════════════════════════════════════════════════

def test_smoke(env: RestaurantBookingEnv) -> None:
    """Basic sanity checks on the environment."""
    print("\n" + "=" * 60)
    print("  TEST 1: Smoke test")
    print("=" * 60)

    obs, info = env.reset()

    # Check observation
    assert obs.shape == (STATE_DIM,), f"Wrong obs shape: {obs.shape}"
    assert obs.dtype == np.float32, f"Wrong obs dtype: {obs.dtype}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), \
        f"Obs out of [0,1]: min={obs.min()}, max={obs.max()}"

    # Check spaces
    assert env.observation_space.shape == (STATE_DIM,)
    assert env.action_space.n == NUM_SYSTEM_ACTIONS

    # Take a few random steps
    total_reward = 0.0
    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (STATE_DIM,), f"Wrong obs shape after step: {obs.shape}"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

        total_reward += reward
        if terminated:
            break

    print(f"  ✓ Observation shape: {obs.shape}")
    print(f"  ✓ Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  ✓ Action space: Discrete({env.action_space.n})")
    print(f"  ✓ Step reward type: float")
    print(f"  ✓ All assertions passed!")


def test_verbose_episodes(env_factory, goals: list, n: int = 3) -> None:
    """Run N episodes with the rule-based policy, printed to stdout."""
    print("\n" + "=" * 60)
    print(f"  TEST 2: {n} verbose episodes (rule-based policy)")
    print("=" * 60)

    # Pick diverse goals: one with booking, one with requests, one complex
    sample_goals = []
    for g in goals:
        if g["info"] and g["book"] and not g.get("reqt"):
            sample_goals.append(g); break
    for g in goals:
        if g["info"] and g.get("reqt") and not g["book"]:
            sample_goals.append(g); break
    for g in goals:
        if g["info"] and g.get("reqt") and g["book"]:
            sample_goals.append(g); break
    while len(sample_goals) < n:
        sample_goals.append(goals[len(sample_goals)])

    for i in range(min(n, len(sample_goals))):
        env = env_factory(render_mode="human")
        obs, info = env.reset(options={"goal": sample_goals[i]})

        for _ in range(env.max_turns):
            action = rule_based_policy(env)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                break

        env.close()


def test_reward_structure(env_factory) -> None:
    """Verify reward penalties fire correctly."""
    print("\n" + "=" * 60)
    print("  TEST 3: Reward structure validation")
    print("=" * 60)

    env = env_factory()
    obs, _ = env.reset()

    # Action before offering → should be invalid
    # inform_address before offer
    _, r_invalid, _, _, _ = env.step(ACTION_TO_IDX["inform_address"])
    expected_invalid = env.step_penalty + env.invalid_penalty
    print(f"  inform_address before offer: reward = {r_invalid:.1f} "
          f"(expected ≈ {expected_invalid:.1f})")
    assert r_invalid <= env.step_penalty, "Invalid action should be penalised"

    # Request booking before offer → invalid
    env2 = env_factory()
    env2.reset()
    _, r_booking_inv, _, _, _ = env2.step(ACTION_TO_IDX["request_people"])
    print(f"  request_people before offer: reward = {r_booking_inv:.1f} "
          f"(expected ≈ {expected_invalid:.1f})")
    assert r_booking_inv <= env.step_penalty

    # Normal request (valid) → just step penalty
    env3 = env_factory()
    env3.reset()
    _, r_normal, _, _, _ = env3.step(ACTION_TO_IDX["request_food"])
    print(f"  request_food (valid):        reward = {r_normal:.1f} "
          f"(expected ≈ {env.step_penalty:.1f})")

    # Check that request for already-filled slot is redundant
    # (food was probably informed by user in opening)
    env4 = env_factory()
    env4.reset()
    state = env4.get_state_info()
    # Find a slot that's already filled
    for slot in INFORMABLE_SLOTS:
        if state["filled_slots"].get(slot):
            _, r_redund, _, _, _ = env4.step(ACTION_TO_IDX[f"request_{slot}"])
            print(f"  request_{slot} (redundant):  reward = {r_redund:.1f} "
                  f"(expected ≈ {env.step_penalty + env.redundant_penalty:.1f})")
            assert r_redund < env.step_penalty, "Redundant action should be penalised"
            break

    print(f"\n  ✓ All reward structure checks passed!")


def test_batch_evaluation(env_factory, goals, n_episodes=500) -> dict:
    """Batch evaluation with rule-based and random policies."""
    print("\n" + "=" * 60)
    print("  TEST 4: Batch evaluation")
    print("=" * 60)

    results = {}

    for policy_name, policy_fn in [("Rule-Based", "rule"), ("Random", "random")]:
        successes, turns_list, rewards_list = [], [], []

        test_goals = goals[:n_episodes]
        for goal in test_goals:
            env = env_factory()
            obs, _ = env.reset(options={"goal": goal})

            ep_reward = 0.0
            for t in range(env.max_turns):
                if policy_fn == "rule":
                    action = rule_based_policy(env)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                if terminated:
                    break

            successes.append(bool(info.get("success", False)))
            turns_list.append(t + 1)
            rewards_list.append(ep_reward)
            env.close()

        stats = {
            "success_rate": np.mean(successes),
            "avg_turns": np.mean(turns_list),
            "std_turns": np.std(turns_list),
            "avg_reward": np.mean(rewards_list),
            "std_reward": np.std(rewards_list),
            "turns_list": turns_list,
            "rewards_list": rewards_list,
        }
        results[policy_name] = stats

        print(f"\n  {policy_name} Policy ({len(test_goals)} episodes)")
        print(f"  {'─' * 40}")
        print(f"    Success rate:  {stats['success_rate']:.1%}")
        print(f"    Avg turns:     {stats['avg_turns']:.2f} ± {stats['std_turns']:.2f}")
        print(f"    Avg reward:    {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}")

    return results


def make_plots(results: dict, plot_dir: Path) -> None:
    """Generate summary plots."""
    sns.set_theme(style="whitegrid", font_scale=1.1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Success rate comparison
    names = list(results.keys())
    success_rates = [results[n]["success_rate"] * 100 for n in names]
    colors = sns.color_palette("viridis", len(names))
    axes[0].bar(names, success_rates, color=colors, edgecolor="black")
    axes[0].set_ylabel("Success Rate (%)")
    axes[0].set_title("Success Rate")
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(success_rates):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=10)

    # 2. Turn distributions
    for name, color in zip(names, colors):
        axes[1].hist(results[name]["turns_list"], bins=range(1, 22),
                     alpha=0.6, label=name, edgecolor="black", color=color)
    axes[1].set_xlabel("Turns")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Turn Distribution")
    axes[1].legend()

    # 3. Reward distributions
    for name, color in zip(names, colors):
        axes[2].hist(results[name]["rewards_list"], bins=30,
                     alpha=0.6, label=name, edgecolor="black", color=color)
    axes[2].set_xlabel("Episode Reward")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Reward Distribution")
    axes[2].legend()

    plt.suptitle("Gymnasium Environment — Policy Evaluation", y=1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "env_policy_evaluation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ env_policy_evaluation.png")


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

    # ── load data ───────────────────────────────────────────────────────
    print("Loading processed data …")
    db = pd.read_csv(processed_dir / "restaurant_db.csv")
    with open(processed_dir / "user_goals_train.json") as f:
        train_goals = json.load(f)
    with open(processed_dir / "user_goals_test.json") as f:
        test_goals = json.load(f)

    print(f"  DB: {len(db)} restaurants")
    print(f"  Train goals: {len(train_goals)},  Test goals: {len(test_goals)}")

    # ── factory for creating fresh envs ─────────────────────────────────
    def make_env(render_mode=None, noise=0.0):
        return RestaurantBookingEnv(
            goals=train_goals,
            db=db,
            noise_level=noise,
            max_turns=max_turns,
            reward_config=reward_config,
            render_mode=render_mode,
            rng=np.random.default_rng(cfg["training"]["seed"]),
        )

    # ── Test 1: Smoke test ──────────────────────────────────────────────
    test_smoke(make_env())

    # ── Test 2: Verbose episodes ────────────────────────────────────────
    test_verbose_episodes(make_env, test_goals, n=3)

    # ── Test 3: Reward structure ────────────────────────────────────────
    test_reward_structure(make_env)

    # ── Test 4: Batch evaluation ────────────────────────────────────────
    n_eval = min(len(test_goals), 437)
    results = test_batch_evaluation(make_env, test_goals, n_eval)

    # ── Plots ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Generating plots …")
    print("=" * 60)
    make_plots(results, plot_dir)

    print(f"\n✅  Step 3 complete!")
    print(f"  State dim: {STATE_DIM}")
    print(f"  Action space: {NUM_SYSTEM_ACTIONS}")
    print(f"  Plots saved to {plot_dir}")


if __name__ == "__main__":
    main()

