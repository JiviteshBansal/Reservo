"""
rule_based.py — Hand-coded and random baseline policies.

Provides two baselines for the restaurant booking environment:

  RuleBasedPolicy : A hand-coded state-machine policy that follows a
      sensible information-gathering → offer → fulfil-requests → book
      → bye strategy, with a fallback confirm step when DB returns 0
      matches (possible noise corruption).

  RandomPolicy : Uniformly random action selection (lower-bound baseline).

Both share a common `select_action(obs, env)` interface so they can be
swapped in with the neural-network policies from Steps 5–6.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.utils import (
    INFORMABLE_SLOTS,
    REQUESTABLE_SLOTS,
    BOOKING_SLOTS,
    ACTION_TO_IDX,
    NUM_SYSTEM_ACTIONS,
)


class RuleBasedPolicy:
    """
    Hand-coded dialogue policy.

    Uses ``env.get_state_info()`` for full access to the belief state.
    The logic follows a priority-ordered rule list:

    Phase 0  React to user bye.
    Phase 1  *Gather*: request any missing informable slot.
    Phase 2  *Resolve*: if 0 DB matches, confirm a filled slot
             (the user may have been misunderstood under noise).
    Phase 3  *Offer*: recommend a venue from the DB.
    Phase 4  *Inform*: fulfil any pending user requests
             (address / phone / postcode).
    Phase 5  *Book*: collect booking details, then confirm the booking.
    Phase 6  *Bye*.
    """

    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        env=None,
    ) -> int:
        """
        Choose an action using the environment's structured state.

        Parameters
        ----------
        obs : np.ndarray or None
            Not used (rule-based reads env.get_state_info() instead).
        env : RestaurantBookingEnv
            The environment instance.
        """
        s = env.get_state_info()

        filled = s["filled_slots"]           # slot → value or ""
        db_count = s["db_match_count"]
        pending = s["pending_requests"]       # set of slot names
        offer_made = s["offer_made"]
        booked = s["booked"]
        user_act = s["user_action"]
        booking = s["booking_state"]          # slot → value or ""

        # ── Phase 0: reciprocate goodbye ────────────────────────────────
        if user_act == "bye":
            return ACTION_TO_IDX["bye"]

        # ── Phase 1: gather informable slots ────────────────────────────
        for slot in INFORMABLE_SLOTS:
            if not filled.get(slot, ""):
                return ACTION_TO_IDX[f"request_{slot}"]

        # ── Phase 2: resolve zero matches (possible noise) ─────────────
        if db_count == 0 and not offer_made:
            # Try confirming each filled slot; the user may correct a
            # noisy value.  Cycle through slots using turn count.
            turn = s["turn"]
            slot_idx = turn % len(INFORMABLE_SLOTS)
            slot = INFORMABLE_SLOTS[slot_idx]
            return ACTION_TO_IDX[f"confirm_{slot}"]

        # ── Phase 3: offer a venue ──────────────────────────────────────
        if not offer_made:
            return ACTION_TO_IDX["offer_venue"]

        # ── Phase 4: fulfil pending info requests ───────────────────────
        for slot in REQUESTABLE_SLOTS:
            if slot in pending:
                return ACTION_TO_IDX[f"inform_{slot}"]

        # ── Phase 5: booking flow ───────────────────────────────────────
        if not booked and offer_made:
            for slot in BOOKING_SLOTS:
                if not booking.get(slot, ""):
                    return ACTION_TO_IDX[f"request_{slot}"]
            # All booking details collected → book
            return ACTION_TO_IDX["book"]

        # ── Phase 6: done ───────────────────────────────────────────────
        return ACTION_TO_IDX["bye"]


class RandomPolicy:
    """
    Uniformly random action selection.

    Serves as a lower bound — any trained policy should beat this.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None):
        self.rng = rng if rng is not None else np.random.default_rng()

    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        env=None,
    ) -> int:
        return int(self.rng.integers(NUM_SYSTEM_ACTIONS))

