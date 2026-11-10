"""
environment.py — Gymnasium Environment for Restaurant Booking Dialogue

The agent is the *dialogue system*, choosing from 15 discrete actions.
An agenda-based user simulator plays the user role.

Observation  : 28-d float vector (see utils.py for layout)
Action       : Discrete(15)
Reward       : -1/turn, +2*max_turns on success, -max_turns on failure,
               extra penalties for redundant or invalid actions

Design decisions:
  • The environment maintains a *belief state* — what the system thinks it
    knows — which may differ from the user's true goal when noise > 0.
  • Actions that are out of context (e.g. booking before offering) are
    penalised but not blocked, so the RL agent learns from the signal.
  • The success function lives in the user simulator: the env just asks
    "did the user's goal get met?" at the end of each episode.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Set, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from src.utils import (
    INFORMABLE_SLOTS,
    REQUESTABLE_SLOTS,
    BOOKING_SLOTS,
    ALL_REQUESTABLE,
    SYSTEM_ACTIONS,
    IDX_TO_ACTION,
    ACTION_TO_IDX,
    NUM_SYSTEM_ACTIONS,
    USER_ACTION_TO_IDX,
    STATE_DIM,
)
from src.data_loader import build_state_vector, query_db
from src.user_simulator import AgendaBasedSimulator


class RestaurantBookingEnv(gym.Env):
    """
    Gymnasium environment for RL-based dialogue policy learning.

    Parameters
    ----------
    goals : list[dict]
        Pool of user goals for the simulator.
    db : pd.DataFrame
        Restaurant database.
    noise_level : float
        User simulator noise (probability of corrupted slot values).
    max_turns : int
        Maximum system turns per episode.
    reward_config : dict | None
        Override default reward values.  Keys:
        step_penalty, success_multiplier, failure_multiplier,
        redundant_penalty, invalid_penalty.
    render_mode : str | None
        "human" prints each turn to stdout.
    rng : np.random.Generator | None
        For reproducibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        goals: list,
        db: pd.DataFrame,
        noise_level: float = 0.0,
        max_turns: int = 20,
        reward_config: Optional[dict] = None,
        render_mode: Optional[str] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        super().__init__()

        # ── spaces ──────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(NUM_SYSTEM_ACTIONS)

        # ── components ──────────────────────────────────────────────────
        self.db = db
        self.max_turns = max_turns
        self.render_mode = render_mode
        self.rng = rng if rng is not None else np.random.default_rng(42)

        # User simulator
        self.simulator = AgendaBasedSimulator(
            goals=goals,
            db=db,
            noise_level=noise_level,
            max_turns=max_turns,
            rng=self.rng,
        )

        # ── reward configuration ────────────────────────────────────────
        rc = reward_config or {}
        self.step_penalty = float(rc.get("step_penalty", -1.0))
        self.success_reward = float(rc.get("success_multiplier", 2)) * max_turns
        self.failure_penalty = float(rc.get("failure_multiplier", -1)) * max_turns
        self.redundant_penalty = float(rc.get("redundant_penalty", -2.0))
        self.invalid_penalty = float(rc.get("invalid_penalty", -3.0))

        # ── per-episode state ───────────────────────────────────────────
        self._reset_tracking()

    # ═══════════════════════════════════════════════════════════════════
    # Core Gymnasium API
    # ═══════════════════════════════════════════════════════════════════

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Start a new dialogue episode.

        Options
        -------
        goal : dict
            Pass a specific user goal (for deterministic evaluation).
        """
        super().reset(seed=seed)
        self._reset_tracking()

        # Optionally accept a specific goal
        goal = options.get("goal") if options else None

        # Reset simulator → user's opening utterance
        user_response = self.simulator.reset(goal)

        # Process the user's opening (fills belief state, etc.)
        self._process_user_response(user_response)

        # Build initial observation
        obs = self._build_observation()

        info = {
            "goal": self.simulator.get_goal_summary(),
            "user_response": user_response,
        }

        if self.render_mode == "human":
            print(f"\n{'═' * 60}")
            print(f"  Goal: {self.simulator.get_goal_summary()}")
            print(f"{'═' * 60}")
            self._render_user(0, user_response)

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one system action.

        Returns (observation, reward, terminated, truncated, info).
        """
        self.turn_count += 1
        action = int(action)
        action_name = IDX_TO_ACTION[action]

        # ── reward (step penalty + any extra penalties) ─────────────────
        reward = self._compute_reward(action)

        # ── build action context for the simulator ──────────────────────
        context = self._build_context(action)

        # ── update environment state from system action ─────────────────
        self._apply_system_action(action, context)

        if self.render_mode == "human":
            self._render_system(self.turn_count, action_name, context)

        # ── get user response ───────────────────────────────────────────
        user_response = self.simulator.step(action, context)

        # ── update belief state from user response ──────────────────────
        self._process_user_response(user_response)

        if self.render_mode == "human":
            self._render_user(self.turn_count, user_response)

        # ── build next observation ──────────────────────────────────────
        obs = self._build_observation()

        # ── termination ─────────────────────────────────────────────────
        terminated = user_response["done"]
        truncated = False  # max-turns is handled inside the simulator

        # ── terminal reward ─────────────────────────────────────────────
        success = False
        if terminated:
            success = self.simulator.is_success()
            if success:
                reward += self.success_reward   # +40
            else:
                reward += self.failure_penalty  # -20

            if self.render_mode == "human":
                tag = "✅ SUCCESS" if success else "❌ FAILURE"
                print(f"\n  {tag}  |  Turns: {self.turn_count}"
                      f"  |  Total reward: {sum(t['reward'] for t in self.dialogue_trace) + reward:.1f}")
                print(f"{'═' * 60}\n")

        # ── log ─────────────────────────────────────────────────────────
        self.dialogue_trace.append({
            "turn": self.turn_count,
            "system_action": action_name,
            "user_action": user_response["user_action"],
            "reward": reward,
        })

        info = {
            "success": success if terminated else None,
            "user_response": user_response,
            "action_name": action_name,
        }

        return obs, reward, terminated, truncated, info

    # ═══════════════════════════════════════════════════════════════════
    # State for rule-based / interpretable policies
    # ═══════════════════════════════════════════════════════════════════

    def get_state_info(self) -> dict:
        """
        Return a human-readable dict of the current dialogue state.

        Useful for rule-based policies that need structured
        access rather than a flat vector.
        """
        return {
            "filled_slots": dict(self.belief_state),
            "db_match_count": self.db_match_count,
            "db_matches": self.db_results,
            "user_action": self.last_user_action,
            "pending_requests": set(self.pending_user_requests),
            "offer_made": self.has_offered,
            "offered_venue": self.offered_venue,
            "booked": self.is_booked,
            "booking_state": dict(self.booking_state),
            "turn": self.turn_count,
        }

    # ═══════════════════════════════════════════════════════════════════
    # Internal: belief state management
    # ═══════════════════════════════════════════════════════════════════

    def _reset_tracking(self) -> None:
        """Reset all per-episode tracking variables."""
        self.belief_state: Dict[str, str] = {}       # informable slots
        self.booking_state: Dict[str, str] = {}      # booking slots
        self.offered_venue: Optional[dict] = None
        self.has_offered: bool = False
        self.is_booked: bool = False
        self.turn_count: int = 0
        self.pending_user_requests: Set[str] = set()
        self.last_user_action: str = "other"
        self.db_results: pd.DataFrame = pd.DataFrame()
        self.db_match_count: int = 0
        self.dialogue_trace: list = []

    def _process_user_response(self, response: dict) -> None:
        """
        Update the environment's belief state from the user's response.

        This is what the system "hears" — if the user gives a noisy value,
        the system records the noisy value (it can't detect noise).
        """
        # Informable slot values
        for slot, value in response.get("slots", {}).items():
            if not value:
                continue
            if slot in INFORMABLE_SLOTS:
                self.belief_state[slot] = value
            elif slot in BOOKING_SLOTS:
                self.booking_state[slot] = value

        # User requests (system must address these)
        for slot in response.get("requests", []):
            self.pending_user_requests.add(slot)

        # Track user's action (for state vector)
        self.last_user_action = response.get("user_action", "other")

        # Re-query DB with updated belief
        constraints = {
            k: v for k, v in self.belief_state.items()
            if v and v != "dontcare" and k in INFORMABLE_SLOTS
        }
        self.db_results = query_db(self.db, constraints)
        self.db_match_count = len(self.db_results)

    def _build_observation(self) -> np.ndarray:
        """Construct the 28-d observation vector from current state."""
        filled_flags = {
            slot: bool(self.belief_state.get(slot, ""))
            for slot in INFORMABLE_SLOTS
        }

        return build_state_vector(
            filled_slots=filled_flags,
            db_match_count=self.db_match_count,
            user_act=self.last_user_action,
            pending_requests=self.pending_user_requests,
            offer_made=self.has_offered,
            booked=self.is_booked,
            turn_num=self.turn_count,
            max_turns=self.max_turns,
        )

    # ═══════════════════════════════════════════════════════════════════
    # Internal: action context & effects
    # ═══════════════════════════════════════════════════════════════════

    def _build_context(self, action: int) -> dict:
        """
        Build the context dict that accompanies a system action.

        The simulator needs this to know *what* the system is offering,
        confirming, or informing.
        """
        action_name = IDX_TO_ACTION[action]
        ctx: dict = {}

        if action_name == "offer_venue":
            if len(self.db_results) > 0:
                ctx = self.db_results.iloc[0].to_dict()
            # else: empty context → simulator treats as failed offer

        elif action_name.startswith("confirm_"):
            slot = action_name.replace("confirm_", "")
            ctx["value"] = self.belief_state.get(slot, "")

        elif action_name.startswith("inform_"):
            slot = action_name.replace("inform_", "")
            if self.offered_venue and slot in self.offered_venue:
                ctx[slot] = self.offered_venue[slot]
            # else: nothing to inform (penalty already applied)

        elif action_name == "book":
            ref = f"REF{int(self.rng.integers(100000)):05d}"
            name = (self.offered_venue.get("name", "unknown")
                    if self.offered_venue else "unknown")
            ctx = {"reference": ref, "name": name}

        return ctx

    def _apply_system_action(self, action: int, context: dict) -> None:
        """Update environment tracking based on the system action taken."""
        action_name = IDX_TO_ACTION[action]

        if action_name == "offer_venue" and context:
            self.offered_venue = context
            self.has_offered = True

        elif action_name.startswith("inform_"):
            slot = action_name.replace("inform_", "")
            self.pending_user_requests.discard(slot)

        elif action_name == "book":
            self.is_booked = True
            self.pending_user_requests.discard("reference")

    # ═══════════════════════════════════════════════════════════════════
    # Internal: reward shaping
    # ═══════════════════════════════════════════════════════════════════

    def _compute_reward(self, action: int) -> float:
        """
        Compute immediate reward for taking an action.

        Reward = step_penalty
               + (at most one extra penalty, whichever is worse)

        Invalid (−3) takes priority over redundant (−2).
        """
        reward = self.step_penalty          # −1 every turn

        # ── check invalidity first (worse penalty) ──────────────────────
        if self._is_invalid(action):
            return reward + self.invalid_penalty   # −1 + (−3) = −4

        # ── then redundancy ─────────────────────────────────────────────
        if self._is_redundant(action):
            return reward + self.redundant_penalty  # −1 + (−2) = −3

        return reward                               # −1

    def _is_invalid(self, action: int) -> bool:
        """An action is invalid if it's nonsensical in the current state."""
        name = IDX_TO_ACTION[action]

        # Can't inform requestable slots without having offered a venue
        if name.startswith("inform_") and not self.has_offered:
            return True

        # Can't book without having offered a venue
        if name == "book" and not self.has_offered:
            return True

        # Can't request booking details without having offered
        if not self.has_offered:
            slot = name.replace("request_", "")
            if slot in BOOKING_SLOTS:
                return True

        return False

    def _is_redundant(self, action: int) -> bool:
        """An action is redundant if it repeats something already done."""
        name = IDX_TO_ACTION[action]

        # Re-requesting an already-filled informable slot
        if name.startswith("request_"):
            slot = name.replace("request_", "")
            if slot in INFORMABLE_SLOTS and self.belief_state.get(slot, ""):
                return True

        # Re-offering when we already offered
        if name == "offer_venue" and self.has_offered:
            return True

        # Re-booking
        if name == "book" and self.is_booked:
            return True

        return False

    # ═══════════════════════════════════════════════════════════════════
    # Rendering
    # ═══════════════════════════════════════════════════════════════════

    def _render_system(self, turn: int, action: str, context: dict) -> None:
        show_keys = {"name", "food", "area", "pricerange", "address",
                     "phone", "postcode", "reference", "value"}
        ctx_short = {k: v for k, v in context.items() if k in show_keys}
        ctx_str = f"  ctx={ctx_short}" if ctx_short else ""
        print(f"  SYS [{turn}]: {action}{ctx_str}")

    def _render_user(self, turn: int, response: dict) -> None:
        act = response["user_action"]
        parts = [f"  USR [{turn}]: {act}"]
        if response.get("slots"):
            parts.append(f"slots={response['slots']}")
        if response.get("requests"):
            parts.append(f"requests={response['requests']}")
        print("    ".join(parts))

