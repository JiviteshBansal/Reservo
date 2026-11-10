#!/usr/bin/env python3
"""
prepare_data.py — Download MultiWOZ 2.1 and extract restaurant-domain data.

Outputs (saved to data/processed/):
  • restaurant_db.csv          — restaurant database
  • user_goals_{split}.json    — user goals per split
  • sl_pairs_{split}.npz       — (state, action) arrays for supervised learning
  • data_statistics.json       — summary statistics

Usage:
  cd "5. RL"
  python3 scripts/prepare_data.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns

# ── add project root to path so `from src…` works ──────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import load_config, set_seed, SYSTEM_ACTIONS, INFORMABLE_SLOTS
from src.data_loader import (
    download_multiwoz,
    load_raw_data,
    load_restaurant_db,
    get_restaurant_dialogues,
    extract_user_goals,
    extract_state_act_pairs,
    compute_statistics,
    print_statistics,
)


def save_processed_data(
    db: pd.DataFrame,
    goals: dict,
    pairs: dict,
    stats: dict,
    output_dir: Path,
) -> None:
    """Save all extracted data to the processed directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── restaurant DB ───────────────────────────────────────────────────
    db_path = output_dir / "restaurant_db.csv"
    db.to_csv(db_path, index=False)
    print(f"  ✓ DB saved to {db_path}")

    # ── user goals ──────────────────────────────────────────────────────
    for split, split_goals in goals.items():
        path = output_dir / f"user_goals_{split}.json"
        with open(path, "w") as f:
            json.dump(split_goals, f, indent=2)
        print(f"  ✓ {len(split_goals)} goals saved to {path}")

    # ── (state, action) pairs as .npz for fast loading ──────────────────
    for split, split_pairs in pairs.items():
        if not split_pairs:
            continue
        states = np.stack([p[0] for p in split_pairs])
        actions = np.array([p[1] for p in split_pairs], dtype=np.int64)
        path = output_dir / f"sl_pairs_{split}.npz"
        np.savez_compressed(path, states=states, actions=actions)
        print(f"  ✓ {len(split_pairs)} pairs saved to {path}  "
              f"(states shape: {states.shape})")

    # ── statistics ──────────────────────────────────────────────────────
    stats_path = output_dir / "data_statistics.json"
    # Convert any numpy types for JSON serialisation
    serialisable = json.loads(json.dumps(stats, default=str))
    with open(stats_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  ✓ Statistics saved to {stats_path}")


def make_plots(
    db: pd.DataFrame,
    goals: dict,
    stats: dict,
    plot_dir: Path,
) -> None:
    """Generate publication-quality plots summarising the dataset."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # ── 1. Action distribution ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    action_dist = stats["action_distribution"]
    actions = list(action_dist.keys())
    counts = list(action_dist.values())
    colours = sns.color_palette("viridis", len(actions))
    ax.barh(actions, counts, color=colours)
    ax.set_xlabel("Count")
    ax.set_title("System Action Distribution in Training Data")
    ax.invert_yaxis()
    for i, v in enumerate(counts):
        ax.text(v + 5, i, str(v), va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(plot_dir / "action_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  ✓ action_distribution.png")

    # ── 2. DB slot distributions (3 subplots) ───────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, slot in zip(axes, INFORMABLE_SLOTS):
        dist = db[slot].value_counts().sort_values(ascending=True)
        dist.plot.barh(ax=ax, color=sns.color_palette("mako", len(dist)))
        ax.set_title(f"DB: {slot}")
        ax.set_xlabel("Count")
    plt.suptitle("Restaurant Database Slot Distributions", y=1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "db_distributions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ db_distributions.png")

    # ── 3. Goal complexity ──────────────────────────────────────────────
    all_goals = goals.get("train", []) + goals.get("val", []) + goals.get("test", [])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 3a. Number of informable constraints
    n_info = [len(g["info"]) for g in all_goals]
    axes[0].hist(n_info, bins=range(0, 5), edgecolor="black", alpha=0.8,
                 color=sns.color_palette("viridis")[2])
    axes[0].set_xlabel("# Informable Constraints")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Goal Constraints")
    axes[0].set_xticks(range(0, 4))

    # 3b. Number of requested slots
    n_reqt = [len(g["reqt"]) for g in all_goals]
    axes[1].hist(n_reqt, bins=range(0, 8), edgecolor="black", alpha=0.8,
                 color=sns.color_palette("viridis")[3])
    axes[1].set_xlabel("# Requested Slots")
    axes[1].set_title("Goal Requests")
    axes[1].set_xticks(range(0, 7))

    # 3c. Booking vs no-booking
    n_book = sum(1 for g in all_goals if g["book"])
    n_nobook = len(all_goals) - n_book
    axes[2].bar(["With Booking", "Without"], [n_book, n_nobook],
                color=[sns.color_palette("viridis")[1],
                       sns.color_palette("viridis")[4]],
                edgecolor="black")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Booking Requirement")

    plt.suptitle("User Goal Complexity", y=1.02)
    plt.tight_layout()
    fig.savefig(plot_dir / "goal_complexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ goal_complexity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    cfg = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    set_seed(cfg["training"]["seed"])

    raw_dir = str(PROJECT_ROOT / cfg["data"]["raw_dir"])
    processed_dir = PROJECT_ROOT / cfg["data"]["processed_dir"]
    plot_dir = PROJECT_ROOT / "results" / "plots"

    # ── 1. Download & locate data ───────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 1a: Download / locate MultiWOZ 2.1")
    print("─" * 60)
    multiwoz_dir = download_multiwoz(raw_dir, cfg["data"]["multiwoz_url"])
    print(f"  Data directory: {multiwoz_dir}")

    # ── 2. Load raw data ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 1b: Load raw data")
    print("─" * 60)
    data, val_ids, test_ids = load_raw_data(multiwoz_dir)

    # ── 3. Restaurant DB ────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 1c: Load restaurant database")
    print("─" * 60)
    db = load_restaurant_db(multiwoz_dir)

    # ── 4. Filter restaurant dialogues ──────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 1d: Filter restaurant-domain dialogues")
    print("─" * 60)
    train_dials, val_dials, test_dials = get_restaurant_dialogues(
        data, val_ids, test_ids
    )

    # ── 5. Extract user goals ───────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 1e: Extract user goals")
    print("─" * 60)
    goals = {
        "train": extract_user_goals(train_dials),
        "val":   extract_user_goals(val_dials),
        "test":  extract_user_goals(test_dials),
    }

    # ── 6. Extract (state, action) pairs ────────────────────────────────
    print("\n" + "─" * 60)
    print("Step 1f: Extract (state, action) pairs for supervised learning")
    print("─" * 60)
    max_turns = cfg["environment"]["max_turns"]
    pairs = {
        "train": extract_state_act_pairs(train_dials, db, max_turns),
        "val":   extract_state_act_pairs(val_dials,   db, max_turns),
        "test":  extract_state_act_pairs(test_dials,  db, max_turns),
    }

    # ── 7. Statistics ───────────────────────────────────────────────────
    stats = compute_statistics(db, goals, pairs)
    print_statistics(stats)

    # ── 8. Save everything ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Saving processed data …")
    print("─" * 60)
    save_processed_data(db, goals, pairs, stats, processed_dir)

    # ── 9. Plots ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("Generating plots …")
    print("─" * 60)
    make_plots(db, goals, stats, plot_dir)

    print("\n✅  Step 1 complete! All outputs in:")
    print(f"    {processed_dir}")
    print(f"    {plot_dir}")


if __name__ == "__main__":
    main()

