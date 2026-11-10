"""
dqn.py — DQN Agent for Restaurant Booking Dialogue

Implements Deep Q-Network (Mnih et al. 2015) adapted for dialogue:
  • Experience replay buffer (circular, numpy-backed for speed)
  • Target network with periodic hard updates
  • Linear epsilon-greedy decay
  • Gradient clipping for training stability
  • Warm-start support: initialise Q-network from a supervised policy

Uses the same DialoguePolicyMLP architecture (no dropout for Q-values)
and the same `select_action(obs, env)` interface as the other policies.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import STATE_DIM, NUM_SYSTEM_ACTIONS
from src.policies.supervised import DialoguePolicyMLP


# ═══════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Fixed-size circular replay buffer backed by pre-allocated numpy arrays.

    Much faster than a Python list of tuples for large buffers because
    sampling uses numpy integer indexing (no Python-level loops).
    """

    def __init__(self, capacity: int, state_dim: int = STATE_DIM):
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.size = 0
        self.ptr = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> Dict[str, torch.Tensor]:
        """Sample a random batch of transitions."""
        idxs = rng.integers(0, self.size, size=batch_size)
        return {
            "states": torch.FloatTensor(self.states[idxs]),
            "actions": torch.LongTensor(self.actions[idxs]),
            "rewards": torch.FloatTensor(self.rewards[idxs]),
            "next_states": torch.FloatTensor(self.next_states[idxs]),
            "dones": torch.FloatTensor(self.dones[idxs]),
        }

    def __len__(self) -> int:
        return self.size


# ═══════════════════════════════════════════════════════════════════════════
# DQN Agent
# ═══════════════════════════════════════════════════════════════════════════

class DQNAgent:
    """
    Deep Q-Network agent for dialogue policy optimisation.

    Parameters
    ----------
    hidden_dims : list[int]
        Hidden layer sizes for the Q-network MLP.
    lr, gamma, buffer_size, batch_size, target_update_freq :
        Standard DQN hyper-parameters (see config.yaml).
    epsilon_start, epsilon_end, epsilon_decay_steps :
        Linear epsilon schedule for exploration.
    device : str
        'cpu' or 'cuda'.
    seed : int
        For the agent's internal RNG.
    """

    def __init__(
        self,
        hidden_dims: List[int] = None,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 10_000,
        device: str = "cpu",
        seed: int = 42,
    ):
        hidden_dims = hidden_dims or [128, 128]
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.rng = np.random.default_rng(seed)

        # ── networks ────────────────────────────────────────────────────
        # Q-network uses dropout=0.0 (standard for value estimation)
        self.q_net = DialoguePolicyMLP(
            STATE_DIM, NUM_SYSTEM_ACTIONS, hidden_dims, dropout=0.0
        ).to(device)

        self.target_net = DialoguePolicyMLP(
            STATE_DIM, NUM_SYSTEM_ACTIONS, hidden_dims, dropout=0.0
        ).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        # ── optimiser ───────────────────────────────────────────────────
        self.optimiser = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # ── replay buffer ───────────────────────────────────────────────
        self.buffer = ReplayBuffer(buffer_size, STATE_DIM)

        # ── step counter (drives epsilon decay & target updates) ────────
        self.step_count = 0

    # ─── epsilon schedule ───────────────────────────────────────────────

    @property
    def epsilon(self) -> float:
        """Current exploration rate (linear decay)."""
        frac = min(1.0, self.step_count / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ─── action selection ───────────────────────────────────────────────

    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        env=None,
        greedy: bool = False,
    ) -> int:
        """
        Epsilon-greedy action selection.

        Set greedy=True for evaluation (no exploration).
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(NUM_SYSTEM_ACTIONS))

        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_values = self.q_net(x)
            return q_values.argmax(dim=-1).item()

    # ─── learning step ──────────────────────────────────────────────────

    def update(self) -> Optional[float]:
        """
        Sample a batch, compute TD error, update Q-network.

        Returns the loss value, or None if the buffer is too small.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size, self.rng)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        # Q(s, a) for the actions that were taken
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target: r + γ · max_a' Q_target(s', a') · (1 − done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimiser.step()

        # ── bookkeeping ─────────────────────────────────────────────────
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    # ─── warm start ────────────────────────────────────────────────────

    def warm_start(self, source_state_dict: dict) -> None:
        """
        Initialise Q-network from a pre-trained model (e.g. supervised).

        Both Q-net and target-net get the same initial weights.
        """
        self.q_net.load_state_dict(source_state_dict)
        self.target_net.load_state_dict(source_state_dict)
        self.q_net.to(self.device)
        self.target_net.to(self.device)

    # ─── save / load ───────────────────────────────────────────────────

    def save(self, path: str) -> None:
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "step_count": self.step_count,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.step_count = ckpt.get("step_count", 0)


# ═══════════════════════════════════════════════════════════════════════════
# Training loop
# ═══════════════════════════════════════════════════════════════════════════

def train_dqn(
    agent: DQNAgent,
    env,
    num_episodes: int = 5000,
    max_turns: int = 20,
    eval_env=None,
    eval_goals: list = None,
    eval_interval: int = 500,
    log_interval: int = 100,
) -> dict:
    """
    Train a DQN agent in the given environment.

    Parameters
    ----------
    agent : DQNAgent
    env : RestaurantBookingEnv
        Training environment (resets with random goals).
    num_episodes : int
    max_turns : int
    eval_env : RestaurantBookingEnv | None
        Separate env for periodic evaluation.
    eval_goals : list | None
        Fixed goals for evaluation.
    eval_interval : int
        Evaluate every N episodes.
    log_interval : int
        Print progress every N episodes.

    Returns
    -------
    dict with training history (rewards, success, losses, eval metrics).
    """
    history = {
        "episode_rewards": [],
        "episode_turns": [],
        "episode_success": [],
        "losses": [],
        "eval_episodes": [],
        "eval_success": [],
        "eval_reward": [],
    }

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_losses = []

        for t in range(max_turns):
            action = agent.select_action(obs=obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            agent.buffer.push(obs, action, reward, next_obs, terminated)
            loss = agent.update()
            if loss is not None:
                ep_losses.append(loss)

            ep_reward += reward
            obs = next_obs
            if terminated:
                break

        success = bool(info.get("success", False))
        history["episode_rewards"].append(ep_reward)
        history["episode_turns"].append(t + 1)
        history["episode_success"].append(float(success))
        if ep_losses:
            history["losses"].append(np.mean(ep_losses))

        # ── periodic logging ────────────────────────────────────────────
        if ep % log_interval == 0:
            window = min(log_interval, ep)
            recent_r = np.mean(history["episode_rewards"][-window:])
            recent_s = np.mean(history["episode_success"][-window:])
            recent_t = np.mean(history["episode_turns"][-window:])
            print(f"  Ep {ep:5d}  ε={agent.epsilon:.3f}  "
                  f"reward={recent_r:+.1f}  success={recent_s:.1%}  "
                  f"turns={recent_t:.1f}  buf={len(agent.buffer)}")

        # ── periodic evaluation ─────────────────────────────────────────
        if eval_goals and eval_env and ep % eval_interval == 0:
            eval_stats = _quick_eval(agent, eval_env, eval_goals[:200],
                                     max_turns)
            history["eval_episodes"].append(ep)
            history["eval_success"].append(eval_stats["success_rate"])
            history["eval_reward"].append(eval_stats["avg_reward"])
            print(f"  ── EVAL ep {ep}: success={eval_stats['success_rate']:.1%}  "
                  f"reward={eval_stats['avg_reward']:.1f}")

    return history


def _quick_eval(
    agent: DQNAgent,
    env,
    goals: list,
    max_turns: int,
) -> dict:
    """Quick greedy evaluation on a subset of goals."""
    successes, rewards = [], []

    for goal in goals:
        obs, _ = env.reset(options={"goal": goal})
        ep_reward = 0.0
        for t in range(max_turns):
            action = agent.select_action(obs=obs, greedy=True)
            obs, reward, terminated, _, info = env.step(action)
            ep_reward += reward
            if terminated:
                break
        successes.append(bool(info.get("success", False)))
        rewards.append(ep_reward)

    return {
        "success_rate": np.mean(successes),
        "avg_reward": np.mean(rewards),
    }

