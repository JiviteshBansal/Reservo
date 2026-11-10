"""
supervised.py — Behaviour-Cloning MLP Policy

Trains a multi-layer perceptron to imitate the system actions observed
in the MultiWOZ 2.1 restaurant dialogues.

Design decisions:
  • Inverse-frequency class weights compensate for the heavy imbalance
    (offer_venue 35%, bye 29% vs. request_time 1%).
  • Dropout (0.2) between hidden layers for regularisation.
  • The same `select_action(obs, env)` interface as the baselines.
  • Supports both greedy (argmax) and stochastic (sample) decoding.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.utils import STATE_DIM, NUM_SYSTEM_ACTIONS


# ═══════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════

class DialoguePolicyMLP(nn.Module):
    """
    Feed-forward MLP for dialogue-act classification.

    Architecture:
      input (28) → [Linear → ReLU → Dropout] × N → Linear → logits (15)
    """

    def __init__(
        self,
        input_dim: int = STATE_DIM,
        output_dim: int = NUM_SYSTEM_ACTIONS,
        hidden_dims: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]

        layers: list = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (batch, num_actions)."""
        return self.network(x)


# ═══════════════════════════════════════════════════════════════════════════
# Policy wrapper
# ═══════════════════════════════════════════════════════════════════════════

class SupervisedPolicy:
    """
    Wraps a trained DialoguePolicyMLP for use in the Gym environment.

    Shares the `select_action(obs, env)` interface with RuleBasedPolicy
    and RandomPolicy.
    """

    def __init__(
        self,
        model: DialoguePolicyMLP,
        device: str = "cpu",
        greedy: bool = True,
    ):
        self.model = model
        self.device = device
        self.greedy = greedy
        self.model.eval()
        self.model.to(device)

    def select_action(
        self,
        obs: Optional[np.ndarray] = None,
        env=None,
    ) -> int:
        """Select action from the MLP's output distribution."""
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            logits = self.model(x)

            if self.greedy:
                return logits.argmax(dim=-1).item()
            else:
                probs = F.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).item()


# ═══════════════════════════════════════════════════════════════════════════
# Training utilities
# ═══════════════════════════════════════════════════════════════════════════

def load_sl_data(
    processed_dir: Path,
) -> Tuple[TensorDataset, TensorDataset, TensorDataset]:
    """
    Load supervised-learning pairs from the .npz files produced by prepare_data.

    Returns (train_dataset, val_dataset, test_dataset).
    """
    datasets = {}
    for split in ["train", "val", "test"]:
        path = processed_dir / f"sl_pairs_{split}.npz"
        data = np.load(path)
        states = torch.FloatTensor(data["states"])
        actions = torch.LongTensor(data["actions"])
        datasets[split] = TensorDataset(states, actions)
        print(f"  {split}: {len(states)} pairs, "
              f"state shape {states.shape}, "
              f"action range [{actions.min()}, {actions.max()}]")

    return datasets["train"], datasets["val"], datasets["test"]


def compute_class_weights(
    dataset: TensorDataset,
    num_classes: int = NUM_SYSTEM_ACTIONS,
    smoothing: float = 0.1,
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights.

    Smoothing prevents extreme weights for very rare classes.
    Classes with zero examples get weight = 0 (so they don't dominate).
    """
    actions = dataset.tensors[1]
    counts = torch.bincount(actions, minlength=num_classes).float()

    # Smooth: add a small count to non-zero classes
    nonzero = counts > 0
    counts[nonzero] += smoothing

    # Inverse frequency
    weights = torch.zeros(num_classes)
    total = counts[nonzero].sum()
    weights[nonzero] = total / (counts[nonzero] * nonzero.sum().float())

    # Normalise so weights sum to num_classes (for stable loss magnitude)
    weights = weights / weights[nonzero].mean()

    return weights


def train_epoch(
    model: DialoguePolicyMLP,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for states, actions in loader:
        states = states.to(device)
        actions = actions.to(device)

        logits = model(states)
        loss = criterion(logits, actions)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * len(states)
        correct += (logits.argmax(dim=-1) == actions).sum().item()
        total += len(states)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(
    model: DialoguePolicyMLP,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Tuple[float, float]:
    """Evaluate on a dataset. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for states, actions in loader:
        states = states.to(device)
        actions = actions.to(device)

        logits = model(states)
        loss = criterion(logits, actions)

        total_loss += loss.item() * len(states)
        correct += (logits.argmax(dim=-1) == actions).sum().item()
        total += len(states)

    return total_loss / total, correct / total


def train_model(
    model: DialoguePolicyMLP,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    lr: float = 1e-3,
    batch_size: int = 128,
    epochs: int = 50,
    device: str = "cpu",
    patience: int = 10,
    class_weights: Optional[torch.Tensor] = None,
) -> dict:
    """
    Full training loop with early stopping.

    Returns a dict with training history.
    """
    model = model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=5,
    )

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        t_loss, t_acc = train_epoch(model, train_loader, optimiser,
                                     criterion, device)
        v_loss, v_acc = eval_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        scheduler.step(v_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  "
                  f"train_loss={t_loss:.4f}  train_acc={t_acc:.3f}  "
                  f"val_loss={v_loss:.4f}  val_acc={v_acc:.3f}")

        # Early stopping
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best val_loss={best_val_loss:.4f})")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history

