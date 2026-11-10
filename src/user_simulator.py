"""
user_simulator.py — Agenda-Based User Simulator for Restaurant Booking

Based on Schatzmann et al., "Statistical User Simulation with a Hidden
Agenda" (SIGdial 2007).  Simplified for the restaurant domain.

Design overview:
  The simulator maintains a *goal* (what the user wants) and an *agenda*
  (a stack of dialogue acts still to perform).  On each turn it receives
  a system action + context, updates its internal state, and returns a
  user response.

  Noise is injected by sometimes providing a wrong or missing slot value
  when the system asks for information, controlled by `noise_level`.

Key interface:
  reset(goal=None)   → initial user act dict
  step(sys_action, context)  → user response dict
  is_success()       → bool (were all goal criteria met?)
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils import (
    INFORMABLE_SLOTS,
    REQUESTABLE_SLOTS,
    BOOKING_SLOTS,
    ALL_REQUESTABLE,
    SYSTEM_ACTIONS,
    IDX_TO_ACTION,
    USER_ACTIONS,
    USER_ACTION_TO_IDX,
)
from src.data_loader import query_db


# ═══════════════════════════════════════════════════════════════════════════
# Simulator
# ═══════════════════════════════════════════════════════════════════════════

class AgendaBasedSimulator:
    """
    Agenda-based user simulator for the restaurant booking domain.

    Parameters
    ----------
    goals : list[dict]
        Pool of user goals (from data extraction).  Each goal has keys
        ``info``, ``reqt``, ``book``, ``fail_info``, ``dial_id``.
    db : pd.DataFrame
        Restaurant database (used for noise injection — sampling wrong
        slot values from real DB values).
    noise_level : float
        Probability ∈ [0, 1] that a slot value is corrupted when provided.
        Half the corruptions replace with a random wrong value; the other
        half omit the value entirely.
    max_turns : int
        Maximum number of *system* turns before the dialogue is truncated.
    rng : np.random.Generator | None
        Random generator for reproducibility.
    """

    def __init__(
        self,
        goals: List[dict],
        db: pd.DataFrame,
        noise_level: float = 0.0,
        max_turns: int = 20,
        rng: np.random.Generator | None = None,
    ):
        self.all_goals = goals
        self.db = db
        self.noise_level = noise_level
        self.max_turns = max_turns
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # Pre-compute unique DB values for noise injection
        self._slot_values: Dict[str, list] = {}
        for slot in INFORMABLE_SLOTS:
            if slot in db.columns:
                vals = db[slot].dropna().unique().tolist()
                self._slot_values[slot] = [v for v in vals if v]

        # ── per-episode state (set in reset) ────────────────────────────
        self.goal: dict = {}
        self.agenda: List[str] = []          # stack (pop from end)
        self.constraints: Dict[str, str] = {}
        self.booking_info: Dict[str, str] = {}
        self.pending_requests: Set[str] = set()
        self.received_info: Dict[str, str] = {}
        self.offered_venue: Optional[dict] = None
        self.offer_matches: bool = False
        self.booked: bool = False
        self.booking_ref: str = ""
        self.turn_count: int = 0
        self.done: bool = False

        # Track what the system has been told (for detecting redundancy)
        self.informed_to_system: Dict[str, str] = {}

    # ─── reset ──────────────────────────────────────────────────────────

    def reset(self, goal: Optional[dict] = None) -> dict:
        """
        Start a new dialogue episode.

        Parameters
        ----------
        goal : dict or None
            If None, sample a random goal from the pool.

        Returns
        -------
        dict with keys:
            user_action : str   — primary user act name
            slots       : dict  — slot→value pairs being informed
            requests    : list  — slot names being requested
            done        : bool  — always False at reset
        """
        # ── select goal ─────────────────────────────────────────────────
        if goal is None:
            idx = int(self.rng.integers(len(self.all_goals)))
            self.goal = copy.deepcopy(self.all_goals[idx])
        else:
            self.goal = copy.deepcopy(goal)

        # ── initialise tracking ─────────────────────────────────────────
        self.constraints = dict(self.goal.get("info", {}))
        self.booking_info = dict(self.goal.get("book", {}))
        self.pending_requests = set(self.goal.get("reqt", []))
        self.received_info = {}
        self.offered_venue = None
        self.offer_matches = False
        self.booked = False
        self.booking_ref = ""
        self.turn_count = 0
        self.done = False
        self.informed_to_system = {}

        # ── build agenda (bottom → top) ─────────────────────────────────
        #   Bottom of stack = bye (last thing to say)
        #   Top of stack    = first constraint to inform
        self.agenda = []
        self.agenda.append("bye")

        # Booking acts (if needed)
        if self.booking_info:
            for slot in BOOKING_SLOTS:
                if slot in self.booking_info:
                    self.agenda.append(f"inform_{slot}")

        # Request acts
        for slot in sorted(self.pending_requests):
            self.agenda.append(f"request_{slot}")

        # Informable constraints (top of stack = first to inform)
        for slot in reversed(INFORMABLE_SLOTS):
            if slot in self.constraints:
                self.agenda.append(f"inform_{slot}")

        # ── generate opening user utterance ─────────────────────────────
        return self._generate_opening()

    # ─── opening utterance ──────────────────────────────────────────────

    def _generate_opening(self) -> dict:
        """
        Generate the user's first utterance.

        Typically informs 1–3 constraints and optionally mentions requests.
        """
        slots_to_inform = {}
        primary_act = "other"

        # Inform all available constraints in the opening
        for slot in INFORMABLE_SLOTS:
            if slot in self.constraints:
                act_name = f"inform_{slot}"
                if act_name in self.agenda:
                    self.agenda.remove(act_name)
                value = self._maybe_corrupt(slot, self.constraints[slot])
                slots_to_inform[slot] = value
                self.informed_to_system[slot] = value
                primary_act = act_name  # last one becomes "primary"

        # Mention pending requests (but don't remove from agenda yet —
        # the system still needs to actually provide the info)
        mentioned_requests = list(self.pending_requests)

        return {
            "user_action": primary_act,
            "slots": slots_to_inform,
            "requests": mentioned_requests,
            "done": False,
        }

    # ─── main step ──────────────────────────────────────────────────────

    def step(
        self,
        system_action_idx: int,
        action_context: Optional[dict] = None,
    ) -> dict:
        """
        Process one system action and return the user's response.

        Parameters
        ----------
        system_action_idx : int
            Index into SYSTEM_ACTIONS.
        action_context : dict or None
            Extra info the environment attaches to the action:
              - offer_venue  → {"name": …, "food": …, "area": …, …}
              - confirm_*    → {"value": …}
              - inform_*     → {"<slot>": …}
              - book         → {"reference": …, "name": …}

        Returns
        -------
        dict with keys: user_action, slots, requests, done
        """
        self.turn_count += 1
        ctx = action_context or {}
        action_name = IDX_TO_ACTION[system_action_idx]

        # Default response
        response = {
            "user_action": "other",
            "slots": {},
            "requests": [],
            "done": False,
        }

        # ── dispatch by system action ───────────────────────────────────
        if action_name.startswith("request_") and action_name.replace("request_", "") in INFORMABLE_SLOTS:
            response = self._handle_info_request(action_name, ctx)

        elif action_name.startswith("confirm_"):
            response = self._handle_confirm(action_name, ctx)

        elif action_name == "offer_venue":
            response = self._handle_offer(ctx)

        elif action_name.startswith("inform_"):
            response = self._handle_inform(action_name, ctx)

        elif action_name.startswith("request_") and action_name.replace("request_", "") in BOOKING_SLOTS:
            response = self._handle_booking_request(action_name, ctx)

        elif action_name == "book":
            response = self._handle_book(ctx)

        elif action_name == "bye":
            response = self._handle_bye()

        # ── check termination ───────────────────────────────────────────
        if self.turn_count >= self.max_turns:
            response["done"] = True
            self.done = True
        if response["done"]:
            self.done = True

        return response

    # ─── handlers for each system action type ───────────────────────────

    def _handle_info_request(self, action_name: str, ctx: dict) -> dict:
        """System asks for an informable slot (food / area / pricerange)."""
        slot = action_name.replace("request_", "")
        slots = {}
        act = "other"

        if slot in self.constraints:
            value = self._maybe_corrupt(slot, self.constraints[slot])
            slots[slot] = value
            self.informed_to_system[slot] = value
            act = f"inform_{slot}"
            # Remove from agenda if present
            agenda_act = f"inform_{slot}"
            if agenda_act in self.agenda:
                self.agenda.remove(agenda_act)
        else:
            # User doesn't care about this slot
            slots[slot] = "dontcare"
            self.informed_to_system[slot] = "dontcare"
            act = f"inform_{slot}"

        return {"user_action": act, "slots": slots, "requests": [], "done": False}

    def _handle_confirm(self, action_name: str, ctx: dict) -> dict:
        """System confirms a slot value (e.g. 'You want Chinese food?')."""
        slot = action_name.replace("confirm_", "")
        confirmed_value = ctx.get("value", "").lower().strip()

        if slot in self.constraints:
            true_value = self.constraints[slot].lower().strip()
            if confirmed_value == true_value or true_value == "dontcare":
                # Correct — affirm
                return {
                    "user_action": "accept",
                    "slots": {},
                    "requests": [],
                    "done": False,
                }
            else:
                # Wrong — deny and provide correct value
                value = self._maybe_corrupt(slot, self.constraints[slot])
                self.informed_to_system[slot] = value
                return {
                    "user_action": "deny",
                    "slots": {slot: value},
                    "requests": [],
                    "done": False,
                }
        else:
            # Slot not in constraints — accept whatever
            return {
                "user_action": "accept",
                "slots": {},
                "requests": [],
                "done": False,
            }

    def _handle_offer(self, ctx: dict) -> dict:
        """System offers / recommends a venue."""
        self.offered_venue = ctx if ctx else None

        # Check if the offered venue matches goal constraints
        if self.offered_venue:
            self.offer_matches = self._venue_matches_goal(self.offered_venue)

            # Auto-receive "display" properties from the offer.
            # In real dialogues the system announces the venue name and its
            # key properties (food type, area, price range).  Detailed info
            # (address, phone, postcode) still requires explicit inform acts.
            for slot in ("food", "area", "pricerange", "name"):
                if slot in self.offered_venue and slot in self.pending_requests:
                    self.received_info[slot] = self.offered_venue[slot]
                    self.pending_requests.discard(slot)
                    req_act = f"request_{slot}"
                    if req_act in self.agenda:
                        self.agenda.remove(req_act)
        else:
            # No venue info — treat as a failed offer
            self.offer_matches = False

        # User's next move depends on what's left on the agenda
        act = "accept"
        slots = {}
        requests = []

        # If there are pending requests, pop one
        for slot in list(self.pending_requests):
            req_act = f"request_{slot}"
            if req_act in self.agenda:
                self.agenda.remove(req_act)
                requests.append(slot)
                act = f"request_{slot}"
                break  # one request per turn

        # If booking is needed and we haven't started providing details,
        # the user might proactively offer booking info
        if not requests and self.booking_info:
            for bslot in BOOKING_SLOTS:
                inform_act = f"inform_{bslot}"
                if inform_act in self.agenda:
                    # Don't proactively provide yet — wait for system to ask
                    break

        return {"user_action": act, "slots": slots, "requests": requests, "done": False}

    def _handle_inform(self, action_name: str, ctx: dict) -> dict:
        """System informs a requestable slot (address / phone / postcode)."""
        slot = action_name.replace("inform_", "")

        # Record the received information
        value = ctx.get(slot, "")
        if value:
            self.received_info[slot] = value

        # Remove from pending requests
        self.pending_requests.discard(slot)
        req_act = f"request_{slot}"
        if req_act in self.agenda:
            self.agenda.remove(req_act)

        # If more requests pending, pop the next one
        act = "accept"
        requests = []
        for rslot in list(self.pending_requests):
            rq = f"request_{rslot}"
            if rq in self.agenda:
                self.agenda.remove(rq)
                requests.append(rslot)
                act = f"request_{rslot}"
                break

        return {"user_action": act, "slots": {}, "requests": requests, "done": False}

    def _handle_booking_request(self, action_name: str, ctx: dict) -> dict:
        """System asks for booking details (people / day / time)."""
        slot = action_name.replace("request_", "")
        slots = {}
        act = "other"

        if slot in self.booking_info:
            value = self.booking_info[slot]
            slots[slot] = value
            act = f"inform_{slot}"
            inform_act = f"inform_{slot}"
            if inform_act in self.agenda:
                self.agenda.remove(inform_act)
        else:
            # System asks for booking detail we don't have
            # Provide a default value
            defaults = {"people": "2", "day": "friday", "time": "12:00"}
            value = defaults.get(slot, "")
            slots[slot] = value
            self.booking_info[slot] = value
            act = f"inform_{slot}"

        return {"user_action": act, "slots": slots, "requests": [], "done": False}

    def _handle_book(self, ctx: dict) -> dict:
        """System confirms a booking."""
        self.booked = True
        self.booking_ref = ctx.get("reference", "REF000")

        # Pop remaining booking acts from agenda
        for slot in BOOKING_SLOTS:
            act = f"inform_{slot}"
            if act in self.agenda:
                self.agenda.remove(act)

        # Check if anything is left on the agenda besides bye
        remaining = [a for a in self.agenda if a != "bye"]
        if not remaining:
            # All done — say bye
            if "bye" in self.agenda:
                self.agenda.remove("bye")
            return {"user_action": "bye", "slots": {}, "requests": [], "done": True}

        return {"user_action": "accept", "slots": {}, "requests": [], "done": False}

    def _handle_bye(self) -> dict:
        """System says goodbye."""
        self.agenda.clear()
        return {"user_action": "bye", "slots": {}, "requests": [], "done": True}

    # ─── success evaluation ─────────────────────────────────────────────

    def is_success(self) -> bool:
        """
        Check whether the dialogue achieved the user's goal.

        All three criteria must be met:
          1. A venue was offered that matches the user's constraints.
          2. All requested slots were provided by the system.
          3. If booking was required, booking was confirmed.
        """
        # 1. Venue offered and matches constraints
        if not self.offer_matches:
            return False

        # 2. All requested info received
        for slot in self.goal.get("reqt", []):
            if slot not in self.received_info:
                return False

        # 3. Booking completed (if required)
        if self.goal.get("book", {}):
            if not self.booked:
                return False

        return True

    def get_goal_summary(self) -> str:
        """Return a human-readable summary of the current goal."""
        parts = []
        if self.constraints:
            c = ", ".join(f"{k}={v}" for k, v in self.constraints.items())
            parts.append(f"Find: {c}")
        if self.pending_requests or self.goal.get("reqt"):
            r = ", ".join(self.goal.get("reqt", []))
            parts.append(f"Request: {r}")
        if self.booking_info:
            b = ", ".join(f"{k}={v}" for k, v in self.booking_info.items())
            parts.append(f"Book: {b}")
        return " | ".join(parts) if parts else "(empty goal)"

    # ─── noise injection ────────────────────────────────────────────────

    def _maybe_corrupt(self, slot: str, true_value: str) -> str:
        """
        With probability ``noise_level``, return a corrupted slot value.

        Corruption is 50/50 between:
          • wrong value (sampled from DB values for that slot)
          • missing value (empty string)
        """
        if self.rng.random() < self.noise_level:
            if self.rng.random() < 0.5 and slot in self._slot_values:
                # Wrong value
                candidates = [v for v in self._slot_values[slot]
                              if v != true_value.lower().strip()]
                if candidates:
                    return str(self.rng.choice(candidates))
            # Missing value
            return ""
        return true_value

    def _venue_matches_goal(self, venue: dict) -> bool:
        """Check whether an offered venue satisfies the user's constraints."""
        for slot, value in self.goal.get("info", {}).items():
            if value.lower().strip() == "dontcare":
                continue
            venue_val = venue.get(slot, "").lower().strip()
            if venue_val != value.lower().strip():
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════════
# Helper: run a full dialogue (for testing without the Gym environment)
# ═══════════════════════════════════════════════════════════════════════════

def run_dialogue(
    simulator: AgendaBasedSimulator,
    policy_fn,
    db: pd.DataFrame,
    goal: Optional[dict] = None,
    max_turns: int = 20,
    verbose: bool = False,
) -> dict:
    """
    Run a single dialogue between a policy function and the simulator.

    Parameters
    ----------
    simulator : AgendaBasedSimulator
    policy_fn : callable(state_dict) → int
        Given a simple state dict, returns a system action index.
    db : pd.DataFrame
        Restaurant DB for resolving offers / informs.
    goal : dict or None
        If None, simulator samples one.
    max_turns : int
    verbose : bool
        If True, print each turn.

    Returns
    -------
    dict with keys: success, turns, reward, trace
    """
    # Reset simulator
    user_response = simulator.reset(goal)
    trace = []
    total_reward = 0.0

    # Build a simple state dict that the policy can use
    # (This is a lightweight stand-in for the full Gym environment state.)
    state = _init_state_dict(user_response, db, simulator)

    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Goal: {simulator.get_goal_summary()}")
        print(f"{'═'*60}")
        _print_user_turn(0, user_response)

    for turn in range(max_turns):
        # ── system acts ─────────────────────────────────────────────
        sys_action_idx = policy_fn(state)
        sys_action_name = IDX_TO_ACTION[sys_action_idx]

        # Build action context (simulate what the environment would do)
        context = _build_action_context(sys_action_idx, state, db)

        if verbose:
            _print_system_turn(turn + 1, sys_action_name, context)

        trace.append({
            "turn": turn + 1,
            "system_action": sys_action_name,
            "context": context,
        })

        # ── user responds ───────────────────────────────────────────
        user_response = simulator.step(sys_action_idx, context)

        if verbose:
            _print_user_turn(turn + 1, user_response)

        trace[-1]["user_response"] = user_response

        # ── update state dict ───────────────────────────────────────
        _update_state_dict(state, sys_action_idx, context, user_response, db)

        # ── reward (simplified: -1 per turn) ────────────────────────
        total_reward -= 1.0

        if user_response["done"]:
            break

    # Final reward
    success = simulator.is_success()
    if success:
        total_reward += 2.0 * max_turns
    else:
        total_reward -= max_turns

    if verbose:
        status = "✅ SUCCESS" if success else "❌ FAILURE"
        print(f"\n  {status}  |  Turns: {turn + 1}  |  Reward: {total_reward:.1f}")
        print(f"{'═'*60}\n")

    return {
        "success": success,
        "turns": turn + 1,
        "reward": total_reward,
        "trace": trace,
    }


# ─── lightweight state dict (stand-in for the Gym env) ──────────────────

def _init_state_dict(
    user_response: dict,
    db: pd.DataFrame,
    simulator: AgendaBasedSimulator,
) -> dict:
    """Create an initial state dict from the user's opening utterance."""
    filled = {}
    for slot, val in user_response.get("slots", {}).items():
        if val and slot in INFORMABLE_SLOTS:
            filled[slot] = val

    constraints = {k: v for k, v in filled.items() if v and v != "dontcare"}
    matches = query_db(db, constraints)

    return {
        "filled_slots": filled,
        "db_matches": matches,
        "db_match_count": len(matches),
        "user_action": user_response["user_action"],
        "user_requests": set(user_response.get("requests", [])),
        "pending_requests": set(user_response.get("requests", [])),
        "offer_made": False,
        "offered_venue": None,
        "booked": False,
        "turn": 0,
    }


def _update_state_dict(
    state: dict,
    sys_action_idx: int,
    context: dict,
    user_response: dict,
    db: pd.DataFrame,
) -> None:
    """Update state dict after a system-action / user-response pair."""
    action_name = IDX_TO_ACTION[sys_action_idx]

    # Update filled slots from user's response
    for slot, val in user_response.get("slots", {}).items():
        if val and slot in INFORMABLE_SLOTS:
            state["filled_slots"][slot] = val

    # Track user requests
    for slot in user_response.get("requests", []):
        state["pending_requests"].add(slot)

    # System action effects
    if action_name == "offer_venue":
        state["offer_made"] = True
        state["offered_venue"] = context
    elif action_name.startswith("inform_"):
        slot = action_name.replace("inform_", "")
        state["pending_requests"].discard(slot)
    elif action_name == "book":
        state["booked"] = True

    # Re-query DB with current constraints
    constraints = {k: v for k, v in state["filled_slots"].items()
                   if v and v != "dontcare"}
    state["db_matches"] = query_db(db, constraints)
    state["db_match_count"] = len(state["db_matches"])

    # Update user action
    state["user_action"] = user_response["user_action"]
    state["turn"] += 1


def _build_action_context(
    sys_action_idx: int,
    state: dict,
    db: pd.DataFrame,
) -> dict:
    """
    Build the context dict that accompanies a system action.

    This simulates what the Gym environment will do.
    """
    action_name = IDX_TO_ACTION[sys_action_idx]
    ctx: dict = {}

    if action_name == "offer_venue":
        # Pick the first matching venue from DB
        matches = state["db_matches"]
        if len(matches) > 0:
            venue = matches.iloc[0].to_dict()
            ctx = venue
        else:
            ctx = {}  # no match

    elif action_name.startswith("confirm_"):
        slot = action_name.replace("confirm_", "")
        ctx["value"] = state["filled_slots"].get(slot, "")

    elif action_name.startswith("inform_"):
        slot = action_name.replace("inform_", "")
        # Provide info from the offered venue
        venue = state.get("offered_venue") or {}
        if venue and slot in venue:
            ctx[slot] = venue[slot]

    elif action_name == "book":
        venue = state.get("offered_venue") or {}
        ctx["reference"] = f"REF{np.random.randint(10000):05d}"
        ctx["name"] = venue.get("name", "unknown")

    return ctx


# ─── pretty-printing ───────────────────────────────────────────────────

def _print_user_turn(turn: int, response: dict) -> None:
    """Print a user turn in a readable format."""
    act = response["user_action"]
    slots = response.get("slots", {})
    reqs = response.get("requests", [])
    parts = [f"  USR [{turn}]: {act}"]
    if slots:
        parts.append(f"  slots={slots}")
    if reqs:
        parts.append(f"  requests={reqs}")
    print("  ".join(parts))


def _print_system_turn(turn: int, action: str, context: dict) -> None:
    """Print a system turn in a readable format."""
    parts = [f"  SYS [{turn}]: {action}"]
    if context:
        # Abbreviate venue context
        show = {k: v for k, v in context.items()
                if k in ("name", "food", "area", "pricerange",
                         "address", "phone", "postcode", "reference", "value")}
        if show:
            parts.append(f"  ctx={show}")
    print("  ".join(parts))
