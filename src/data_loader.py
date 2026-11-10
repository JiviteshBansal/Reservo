"""
data_loader.py — MultiWOZ 2.1 Restaurant-Domain Data Extraction

Responsibilities:
  1. Download & unzip the MultiWOZ 2.1 dataset.
  2. Load the restaurant database into a DataFrame.
  3. Extract user goals from dialogue goal annotations.
  4. Map each (user-turn, system-turn) pair to a (state_vector, action_index)
     for supervised / behaviour-cloning training.

Design decisions:
  • We only consider dialogues whose goal includes the restaurant domain.
  • Multi-act system turns are reduced to a single action via a fixed priority
    (book > offer > request_info > request_booking > inform > confirm > bye).
  • "dontcare" slot values count as *filled*: the user has expressed a preference.
  • The state vector is 28-d (see utils.py for the exact layout).
"""

import json
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.utils import (
    INFORMABLE_SLOTS, REQUESTABLE_SLOTS, BOOKING_SLOTS, ALL_REQUESTABLE,
    SYSTEM_ACTIONS, ACTION_TO_IDX, USER_ACTIONS, USER_ACTION_TO_IDX,
    NUM_USER_ACTIONS, STATE_DIM,
    bin_db_matches,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Download helpers
# ═══════════════════════════════════════════════════════════════════════════

def download_multiwoz(data_dir: str, url: str) -> Path:
    """
    Download MultiWOZ 2.1 zip if not already present and extract it.

    Returns the path to the extracted data directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "MultiWOZ_2.1.zip"

    # ── download ────────────────────────────────────────────────────────
    if not zip_path.exists():
        print(f"Downloading MultiWOZ 2.1 from:\n  {url}")
        urllib.request.urlretrieve(url, str(zip_path))
        print(f"  → saved to {zip_path}  ({zip_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Zip already exists: {zip_path}")

    # ── extract ─────────────────────────────────────────────────────────
    # The zip may contain a top-level folder (e.g. "MULTIWOZ2.1/").
    # We detect it and return whichever directory holds data.json.
    extracted_marker = data_dir / ".extracted"
    if not extracted_marker.exists():
        print("Extracting …")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        extracted_marker.touch()
        print("  → done")
    else:
        print("Already extracted.")

    # Find the directory containing data.json
    for candidate in [data_dir, *data_dir.iterdir()]:
        if candidate.is_dir() and (candidate / "data.json").exists():
            return candidate
    # Broader search
    for root, _dirs, files in os.walk(data_dir):
        if "data.json" in files:
            return Path(root)
    raise FileNotFoundError(
        f"Could not find data.json anywhere under {data_dir}. "
        "Please download MultiWOZ 2.1 manually."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Load raw data
# ═══════════════════════════════════════════════════════════════════════════

def load_raw_data(multiwoz_dir: Path) -> Tuple[dict, list, list]:
    """
    Load the three essential files:
      • data.json       — all dialogues with annotations
      • valListFile.json (or .txt)  — validation dialogue IDs
      • testListFile.json (or .txt) — test dialogue IDs

    Returns (data_dict, val_ids, test_ids).
    """
    # ── main data ───────────────────────────────────────────────────────
    data_path = multiwoz_dir / "data.json"
    print(f"Loading {data_path.name} …")
    with open(data_path, "r") as f:
        data = json.load(f)
    print(f"  → {len(data)} total dialogues loaded")

    # ── split lists ─────────────────────────────────────────────────────
    val_ids = _load_id_list(multiwoz_dir, "valListFile")
    test_ids = _load_id_list(multiwoz_dir, "testListFile")
    print(f"  → val: {len(val_ids)}, test: {len(test_ids)} dialogue IDs")

    return data, val_ids, test_ids


def _load_id_list(directory: Path, prefix: str) -> list:
    """Try .json then .txt for a split-list file."""
    for ext in [".json", ".txt"]:
        path = directory / f"{prefix}{ext}"
        if path.exists():
            with open(path, "r") as f:
                content = f.read()
            # May be a JSON list or newline-separated IDs
            try:
                ids = json.loads(content)
            except json.JSONDecodeError:
                ids = [line.strip() for line in content.strip().split("\n") if line.strip()]
            return ids
    print(f"  ⚠ {prefix} file not found; will treat all as train")
    return []


# ═══════════════════════════════════════════════════════════════════════════
# 3. Restaurant database
# ═══════════════════════════════════════════════════════════════════════════

def load_restaurant_db(multiwoz_dir: Path) -> pd.DataFrame:
    """
    Load restaurant_db.json into a DataFrame.

    Each row is a restaurant with columns:
      name, food, area, pricerange, address, phone, postcode, id, …
    """
    db_path = multiwoz_dir / "restaurant_db.json"
    with open(db_path, "r") as f:
        entries = json.load(f)

    df = pd.DataFrame(entries)
    # Normalise text to lower-case for matching (skip non-string columns)
    for col in df.columns:
        if df[col].dtype == object:
            # Some "object" columns may contain mixed types; convert safely
            df[col] = df[col].astype(str).str.strip().str.lower()
    print(f"Restaurant DB: {len(df)} entries, columns = {list(df.columns)}")
    return df


def query_db(db: pd.DataFrame, constraints: Dict[str, str]) -> pd.DataFrame:
    """
    Query the restaurant DB with the given slot constraints.
    'dontcare' values are ignored (they match everything).
    """
    mask = pd.Series(True, index=db.index)
    for slot, value in constraints.items():
        if value and value != "dontcare" and slot in db.columns:
            mask &= (db[slot] == value.lower().strip())
    return db[mask]


# ═══════════════════════════════════════════════════════════════════════════
# 4. Extract user goals
# ═══════════════════════════════════════════════════════════════════════════

def get_restaurant_dialogues(
    data: dict, val_ids: list, test_ids: list
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, dict]]:
    """
    Filter dialogues that include the restaurant domain.

    Returns three dicts: train, val, test — each mapping dialogue_id → dialogue.
    """
    val_set = set(val_ids)
    test_set = set(test_ids)
    train, val, test = {}, {}, {}

    for dial_id, dial in data.items():
        # A dialogue is relevant if its goal has a non-empty restaurant sub-goal
        goal = dial.get("goal", {})
        rest_goal = goal.get("restaurant", {})
        if not rest_goal or not rest_goal.get("info", {}):
            continue  # no restaurant constraints → skip

        if dial_id in test_set:
            test[dial_id] = dial
        elif dial_id in val_set:
            val[dial_id] = dial
        else:
            train[dial_id] = dial

    print(f"Restaurant dialogues — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


def extract_user_goals(dialogues: Dict[str, dict]) -> List[dict]:
    """
    Extract structured user goals from restaurant dialogue annotations.

    Each goal dict has:
      info      : dict of informable slot→value  (food, area, pricerange)
      reqt      : list of requested slot names     (address, phone, …)
      book      : dict of booking slot→value       (people, day, time) or {}
      fail_info : dict of fallback constraints     (when first query yields nothing)
      dial_id   : source dialogue ID
    """
    goals = []
    for dial_id, dial in dialogues.items():
        rest_goal = dial["goal"]["restaurant"]

        # -- informable constraints -----------------------------------------
        info = {}
        for slot in INFORMABLE_SLOTS:
            val = rest_goal.get("info", {}).get(slot, "")
            if val:
                info[slot] = val.lower().strip()

        # -- requestable slots ----------------------------------------------
        reqt = rest_goal.get("reqt", [])
        if isinstance(reqt, dict):
            reqt = list(reqt.keys())
        reqt = [r.lower().strip() for r in reqt]

        # -- booking --------------------------------------------------------
        book_raw = rest_goal.get("book", {})
        book = {}
        for slot in BOOKING_SLOTS:
            val = book_raw.get(slot, "")
            if val:
                book[slot] = val.lower().strip()

        # -- fallback constraints -------------------------------------------
        fail_info = {}
        for slot in INFORMABLE_SLOTS:
            val = rest_goal.get("fail_info", {}).get(slot, "")
            if val:
                fail_info[slot] = val.lower().strip()

        goals.append({
            "info": info,
            "reqt": reqt,
            "book": book,
            "fail_info": fail_info,
            "dial_id": dial_id,
        })
    return goals


# ═══════════════════════════════════════════════════════════════════════════
# 5. Map dialogue acts → our action space
# ═══════════════════════════════════════════════════════════════════════════

def _normalise_act_key(act_key: str) -> Tuple[str, str]:
    """
    Split 'Restaurant-Inform' → ('restaurant', 'inform'),
          'Booking-Request'   → ('booking', 'request'),
          'general-bye'       → ('general', 'bye').
    """
    parts = act_key.lower().replace("_", "-").split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], ""


def map_system_acts_to_action(dialog_acts: dict) -> Optional[int]:
    """
    Map a system turn's MultiWOZ dialog_act dict to a single action index
    from our 15-action space.

    Returns None if no recognisable action is found (turn is skipped for SL).

    Priority order ensures the most dialogue-advancing act is chosen:
      book > offer_venue > request_info > request_booking >
      inform_requestable > confirm > bye
    """
    if not dialog_acts:
        return None

    candidates = []   # list of (priority, action_name)

    for act_key, slot_value_pairs in dialog_acts.items():
        domain, act_type = _normalise_act_key(act_key)
        slots = {sv[0].lower() for sv in slot_value_pairs} if slot_value_pairs else set()

        # ── booking confirmation (highest priority) ─────────────────────
        if act_type in ("book", "offerbooked"):
            candidates.append((0, "book"))

        # ── booking info with reference → also counts as book ──────────
        elif act_type == "inform" and domain == "booking" and "ref" in slots:
            candidates.append((0, "book"))

        # ── offer / recommend a venue ───────────────────────────────────
        elif act_type in ("recommend", "select"):
            candidates.append((1, "offer_venue"))
        elif act_type == "inform" and domain == "restaurant" and "name" in slots:
            candidates.append((1, "offer_venue"))
        elif act_type == "nooffer":
            candidates.append((1, "offer_venue"))  # no-offer is still an "offer attempt"

        # ── request informable slots ────────────────────────────────────
        elif act_type == "request" and domain == "restaurant":
            for slot in INFORMABLE_SLOTS:
                if slot in slots:
                    candidates.append((2, f"request_{slot}"))

        # ── request booking details ─────────────────────────────────────
        elif act_type == "request" and domain == "booking":
            for slot in BOOKING_SLOTS:
                if slot in slots:
                    candidates.append((3, f"request_{slot}"))
        elif act_type == "offerbook":
            # System offering to book — typically asks for booking details
            candidates.append((3, "request_people"))

        # ── inform requestable slots ────────────────────────────────────
        elif act_type == "inform" and domain == "restaurant":
            for slot in REQUESTABLE_SLOTS:
                if slot in slots:
                    candidates.append((4, f"inform_{slot}"))

        # ── confirm (rare — we look for select-style acts) ──────────────
        # Restaurant-Inform with choice count is sometimes a soft confirm
        # We don't explicitly map confirms from data; RL learns them.

        # ── bye ─────────────────────────────────────────────────────────
        elif act_type in ("bye", "reqmore"):
            candidates.append((6, "bye"))

    if not candidates:
        return None

    # Pick highest-priority (lowest number) candidate
    candidates.sort(key=lambda x: x[0])
    action_name = candidates[0][1]
    return ACTION_TO_IDX.get(action_name)


def map_user_acts_to_action(dialog_acts: dict) -> str:
    """
    Map a user turn's dialog_act dict to a user-action string
    (used for the one-hot part of the state vector).
    """
    if not dialog_acts:
        return "other"

    # Collect all candidate user actions with priorities
    candidates = []

    for act_key, slot_value_pairs in dialog_acts.items():
        domain, act_type = _normalise_act_key(act_key)
        slots = {sv[0].lower() for sv in slot_value_pairs} if slot_value_pairs else set()

        if act_type == "inform" and domain == "restaurant":
            for slot in INFORMABLE_SLOTS:
                if slot in slots:
                    candidates.append((0, f"inform_{slot}"))

        elif act_type == "inform" and domain == "booking":
            for slot in BOOKING_SLOTS:
                if slot in slots:
                    candidates.append((0, f"inform_{slot}"))

        elif act_type == "request" and domain == "restaurant":
            for slot in REQUESTABLE_SLOTS:
                if slot in slots:
                    candidates.append((1, f"request_{slot}"))
            if "ref" in slots or "reference" in slots:
                candidates.append((1, "request_reference"))

        elif act_type == "bye":
            candidates.append((3, "bye"))

        elif act_type in ("affirm", "thankyou"):
            candidates.append((2, "accept"))

        elif act_type in ("negate", "deny"):
            candidates.append((2, "deny"))

    if not candidates:
        return "other"
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


# ═══════════════════════════════════════════════════════════════════════════
# 6. Build state vectors & extract (state, action) pairs
# ═══════════════════════════════════════════════════════════════════════════

def build_state_vector(
    filled_slots: Dict[str, bool],
    db_match_count: int,
    user_act: str,
    pending_requests: Set[str],
    offer_made: bool,
    booked: bool,
    turn_num: int,
    max_turns: int,
) -> np.ndarray:
    """
    Construct the 28-dimensional state vector.

    Layout (see utils.py for detailed docs):
      [0:3]   filled informable slot flags
      [3:7]   one-hot DB match bin
      [7:21]  one-hot user last act
      [21:25] pending request flags
      [25:27] status flags  (offer_made, booked)
      [27]    normalised turn count
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0

    # filled informable slots
    for slot in INFORMABLE_SLOTS:
        state[idx] = 1.0 if filled_slots.get(slot, False) else 0.0
        idx += 1

    # DB match bin
    state[idx: idx + 4] = bin_db_matches(db_match_count)
    idx += 4

    # user last act (one-hot)
    ua_idx = USER_ACTION_TO_IDX.get(user_act, USER_ACTION_TO_IDX["other"])
    state[idx + ua_idx] = 1.0
    idx += NUM_USER_ACTIONS

    # pending request flags
    for slot in ALL_REQUESTABLE:
        state[idx] = 1.0 if slot in pending_requests else 0.0
        idx += 1

    # status flags
    state[idx] = 1.0 if offer_made else 0.0
    idx += 1
    state[idx] = 1.0 if booked else 0.0
    idx += 1

    # normalised turn count
    state[idx] = min(turn_num / max_turns, 1.0)

    assert idx == STATE_DIM - 1, f"State index mismatch: {idx} vs {STATE_DIM - 1}"
    return state


def extract_state_act_pairs(
    dialogues: Dict[str, dict],
    db: pd.DataFrame,
    max_turns: int = 20,
) -> List[Tuple[np.ndarray, int]]:
    """
    Walk through each dialogue turn-by-turn and extract (state, action_idx)
    pairs suitable for supervised learning.

    For each system turn we:
      1. Read the belief state from metadata to know which slots are filled.
      2. Query the DB with current constraints to get the match count.
      3. Build the 28-d state vector from tracked conversation state.
      4. Map the system's dialog_act to our action space.
      5. Store (state_vector, action_index).

    Turns that can't be mapped are skipped (logged at debug level).
    """
    pairs = []
    skipped = 0

    for dial_id, dial in dialogues.items():
        log = dial["log"]

        # ── per-dialogue tracking ───────────────────────────────────────
        user_act = "other"
        pending_requests: Set[str] = set()
        offer_made = False
        booked = False
        turn_num = 0   # increments per *system* turn (dialogue moves)

        for i, turn in enumerate(log):
            is_user_turn = (i % 2 == 0)
            dialog_acts = turn.get("dialog_act", {})

            if is_user_turn:
                # ── user turn: update tracking ──────────────────────────
                user_act = map_user_acts_to_action(dialog_acts)

                # Track newly requested slots
                for act_key, svs in dialog_acts.items():
                    _, act_type = _normalise_act_key(act_key)
                    if act_type == "request":
                        for sv in (svs or []):
                            slot = sv[0].lower()
                            if slot in ALL_REQUESTABLE:
                                pending_requests.add(slot)

            else:
                # ── system turn: build state & extract action ───────────
                # 1) Belief state from metadata
                meta = turn.get("metadata", {})
                rest_meta = meta.get("restaurant", {})
                semi = rest_meta.get("semi", {})

                filled_slots = {}
                constraints = {}
                for slot in INFORMABLE_SLOTS:
                    val = semi.get(slot, "")
                    filled_slots[slot] = bool(val)
                    if val and val != "not mentioned":
                        constraints[slot] = val

                # Booking state
                book_meta = rest_meta.get("book", {})
                if book_meta.get("booked", []):
                    booked = True

                # 2) DB match count
                matches = query_db(db, constraints)
                db_count = len(matches)

                # 3) Build state vector
                state_vec = build_state_vector(
                    filled_slots=filled_slots,
                    db_match_count=db_count,
                    user_act=user_act,
                    pending_requests=pending_requests,
                    offer_made=offer_made,
                    booked=booked,
                    turn_num=turn_num,
                    max_turns=max_turns,
                )

                # 4) Map system action
                action_idx = map_system_acts_to_action(dialog_acts)

                if action_idx is not None:
                    pairs.append((state_vec, action_idx))

                    # Update tracking based on system action
                    action_name = SYSTEM_ACTIONS[action_idx]

                    if action_name == "offer_venue":
                        offer_made = True
                    elif action_name == "book":
                        booked = True
                    elif action_name.startswith("inform_"):
                        informed_slot = action_name.replace("inform_", "")
                        pending_requests.discard(informed_slot)
                else:
                    skipped += 1

                turn_num += 1

    print(f"Extracted {len(pairs)} (state, action) pairs  "
          f"({skipped} system turns skipped — unmappable acts)")
    return pairs


# ═══════════════════════════════════════════════════════════════════════════
# 7. Statistics
# ═══════════════════════════════════════════════════════════════════════════

def compute_statistics(
    db: pd.DataFrame,
    goals: Dict[str, List[dict]],
    pairs: Dict[str, List[Tuple[np.ndarray, int]]],
) -> dict:
    """
    Compute and return summary statistics for the extracted data.

    Parameters:
      db    : restaurant DataFrame
      goals : {"train": [...], "val": [...], "test": [...]}
      pairs : {"train": [...], "val": [...], "test": [...]}

    Returns a dict of statistics for display / logging.
    """
    stats = {}

    # ── DB stats ────────────────────────────────────────────────────────
    stats["db_size"] = len(db)
    stats["db_food_types"] = db["food"].nunique() if "food" in db.columns else 0
    stats["db_areas"] = db["area"].nunique() if "area" in db.columns else 0
    stats["db_priceranges"] = db["pricerange"].nunique() if "pricerange" in db.columns else 0
    stats["db_food_dist"] = db["food"].value_counts().to_dict() if "food" in db.columns else {}
    stats["db_area_dist"] = db["area"].value_counts().to_dict() if "area" in db.columns else {}
    stats["db_price_dist"] = db["pricerange"].value_counts().to_dict() if "pricerange" in db.columns else {}

    # ── goal stats ──────────────────────────────────────────────────────
    all_goals = []
    for split in ["train", "val", "test"]:
        split_goals = goals.get(split, [])
        stats[f"num_goals_{split}"] = len(split_goals)
        all_goals.extend(split_goals)

    stats["num_goals_total"] = len(all_goals)

    # Goal complexity: number of informable constraints
    n_constraints = [len(g["info"]) for g in all_goals]
    stats["goal_constraints_dist"] = dict(Counter(n_constraints))

    # How many goals require booking?
    n_booking = sum(1 for g in all_goals if g["book"])
    stats["goals_with_booking"] = n_booking
    stats["goals_with_booking_pct"] = (
        100.0 * n_booking / len(all_goals) if all_goals else 0
    )

    # How many goals have requestable slots?
    n_reqt = [len(g["reqt"]) for g in all_goals]
    stats["goal_reqt_dist"] = dict(Counter(n_reqt))

    # How many goals have fallback constraints?
    n_fail = sum(1 for g in all_goals if g["fail_info"])
    stats["goals_with_fallback"] = n_fail

    # ── (state, action) pair stats ──────────────────────────────────────
    for split in ["train", "val", "test"]:
        split_pairs = pairs.get(split, [])
        stats[f"num_pairs_{split}"] = len(split_pairs)

    all_pairs = pairs.get("train", []) + pairs.get("val", []) + pairs.get("test", [])
    stats["num_pairs_total"] = len(all_pairs)

    # Action distribution
    action_counts = Counter(p[1] for p in all_pairs)
    stats["action_distribution"] = {
        SYSTEM_ACTIONS[k]: v for k, v in sorted(action_counts.items())
    }

    return stats


def print_statistics(stats: dict) -> None:
    """Pretty-print the computed statistics."""
    print("\n" + "=" * 70)
    print("  DATA STATISTICS — MultiWOZ 2.1 Restaurant Domain")
    print("=" * 70)

    print(f"\n📦 Restaurant Database")
    print(f"  Entries:       {stats['db_size']}")
    print(f"  Food types:    {stats['db_food_types']}")
    print(f"  Areas:         {stats['db_areas']}")
    print(f"  Price ranges:  {stats['db_priceranges']}")

    print(f"\n  Food distribution (top 10):")
    for food, count in sorted(stats["db_food_dist"].items(),
                              key=lambda x: -x[1])[:10]:
        print(f"    {food:20s}  {count}")

    print(f"\n  Area distribution:")
    for area, count in sorted(stats["db_area_dist"].items(),
                              key=lambda x: -x[1]):
        print(f"    {area:20s}  {count}")

    print(f"\n  Price range distribution:")
    for pr, count in sorted(stats["db_price_dist"].items(),
                            key=lambda x: -x[1]):
        print(f"    {pr:20s}  {count}")

    print(f"\n🎯 User Goals")
    print(f"  Total:             {stats['num_goals_total']}")
    print(f"    Train:           {stats['num_goals_train']}")
    print(f"    Validation:      {stats['num_goals_val']}")
    print(f"    Test:            {stats['num_goals_test']}")
    print(f"  With booking:      {stats['goals_with_booking']}  "
          f"({stats['goals_with_booking_pct']:.1f}%)")
    print(f"  With fallback:     {stats['goals_with_fallback']}")

    print(f"\n  Informable constraints per goal:")
    for n, count in sorted(stats["goal_constraints_dist"].items()):
        print(f"    {n} constraints:  {count}")

    print(f"\n  Requested slots per goal:")
    for n, count in sorted(stats["goal_reqt_dist"].items()):
        print(f"    {n} slots:        {count}")

    print(f"\n📊 Supervised Learning Pairs (state, action)")
    print(f"  Total:             {stats['num_pairs_total']}")
    print(f"    Train:           {stats['num_pairs_train']}")
    print(f"    Validation:      {stats['num_pairs_val']}")
    print(f"    Test:            {stats['num_pairs_test']}")

    print(f"\n  Action distribution:")
    total = sum(stats["action_distribution"].values())
    for action, count in sorted(stats["action_distribution"].items(),
                                key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0
        bar = "█" * int(pct / 2)
        print(f"    {action:25s}  {count:5d}  ({pct:5.1f}%)  {bar}")

    print("=" * 70)
