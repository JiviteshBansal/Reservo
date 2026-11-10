"""
utils.py — Shared constants, seed management, and helper functions.

Design note:
  All slot names, action spaces, and state dimensions are defined here
  so that every module imports from a single source of truth.
"""

import random
import numpy as np
import torch
import yaml
from pathlib import Path
from typing import Dict


# ─── reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ─── domain slot definitions ────────────────────────────────────────────────

INFORMABLE_SLOTS = ["food", "area", "pricerange"]
REQUESTABLE_SLOTS = ["address", "phone", "postcode"]
BOOKING_SLOTS = ["people", "day", "time"]
ALL_REQUESTABLE = REQUESTABLE_SLOTS + ["reference"]   # users can also ask for booking ref


# ─── system action space (15 discrete actions) ──────────────────────────────
#
# These are the atomic dialogue acts the *system* can perform.
# Multi-act turns in the real data get mapped to the single highest-priority
# action (see data_loader.map_system_acts_to_action).

SYSTEM_ACTIONS = [
    "request_food",        # 0  — ask user for food type
    "request_area",        # 1  — ask user for area
    "request_pricerange",  # 2  — ask user for price range
    "confirm_food",        # 3  — confirm food type with user
    "confirm_area",        # 4  — confirm area with user
    "confirm_pricerange",  # 5  — confirm price range with user
    "offer_venue",         # 6  — recommend / offer a restaurant
    "inform_address",      # 7  — provide restaurant address
    "inform_phone",        # 8  — provide restaurant phone
    "inform_postcode",     # 9  — provide restaurant postcode
    "request_people",      # 10 — ask how many people for booking
    "request_day",         # 11 — ask which day for booking
    "request_time",        # 12 — ask what time for booking
    "book",                # 13 — confirm the booking
    "bye",                 # 14 — end conversation
]
NUM_SYSTEM_ACTIONS = len(SYSTEM_ACTIONS)   # 15
ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(SYSTEM_ACTIONS)}
IDX_TO_ACTION: Dict[int, str] = {i: a for i, a in enumerate(SYSTEM_ACTIONS)}


# ─── user action space (for state encoding) ─────────────────────────────────

USER_ACTIONS = [
    "inform_food",         # 0
    "inform_area",         # 1
    "inform_pricerange",   # 2
    "request_address",     # 3
    "request_phone",       # 4
    "request_postcode",    # 5
    "request_reference",   # 6
    "accept",              # 7
    "deny",                # 8
    "inform_people",       # 9
    "inform_day",          # 10
    "inform_time",         # 11
    "bye",                 # 12
    "other",               # 13
]
NUM_USER_ACTIONS = len(USER_ACTIONS)       # 14
USER_ACTION_TO_IDX: Dict[str, int] = {a: i for i, a in enumerate(USER_ACTIONS)}


# ─── state vector layout ────────────────────────────────────────────────────
#
# [0:3]   filled flags for informable slots  (food, area, pricerange)
# [3:7]   one-hot DB match bin               (0, 1, 2–5, 6+)
# [7:21]  one-hot user's last act
# [21:25] pending-request flags              (address, phone, postcode, reference)
# [25:27] status flags                       (offer_made, booked)
# [27]    normalised turn count              (turn / max_turns)

NUM_INFORMABLE = 3
NUM_DB_BINS = 4
NUM_PENDING_REQ = 4        # address, phone, postcode, reference
NUM_FLAGS = 2              # offer_made, booked
NUM_TURN = 1

STATE_DIM = (NUM_INFORMABLE + NUM_DB_BINS + NUM_USER_ACTIONS
             + NUM_PENDING_REQ + NUM_FLAGS + NUM_TURN)
# 3 + 4 + 14 + 4 + 2 + 1 = 28


def bin_db_matches(count: int) -> np.ndarray:
    """Convert a DB match count to a one-hot bin vector [0, 1, 2–5, 6+]."""
    vec = np.zeros(NUM_DB_BINS, dtype=np.float32)
    if count == 0:
        vec[0] = 1.0
    elif count == 1:
        vec[1] = 1.0
    elif count <= 5:
        vec[2] = 1.0
    else:
        vec[3] = 1.0
    return vec

