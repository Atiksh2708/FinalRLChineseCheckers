"""
model.py — ResNet-MLP ActorCritic for Chinese Checkers.

═══════════════════════════════════════════════════════════════════════════
DESIGN
═══════════════════════════════════════════════════════════════════════════

Network topology
────────────────
Input (176)
    → Linear(176 → 512) + LayerNorm + ReLU
    → ResBlock(512) × 4
    → Trunk output (512 dims)
    → Policy head: Linear(512 → 256) + ReLU + Linear(256 → 1210)
    → Value head:  Linear(512 → 256) + ReLU + Linear(256 → 1)

Total params: ~2.5M (vs ~200K for the previous 256-hidden 2-layer MLP).
Trains comfortably on a 4060 GPU; runnable on CPU but slower.

Why residual blocks
───────────────────
Plain deep MLPs are hard to train — gradients vanish. Residual blocks
(x + F(x)) let gradient flow directly through skip connections, making
deep networks trainable. Standard in modern policy networks since AlphaZero.

LayerNorm vs BatchNorm
──────────────────────
LayerNorm is preferred here because:
  • Works with batch size 1 (during inference)
  • Doesn't track running statistics (cleaner for RL)
  • Stable across train/eval mode transitions

Action masking
──────────────
Done inside act() and evaluate() — invalid actions get logits set to -1e9
before the Categorical distribution. The agent can never pick an illegal move.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def layer_init(layer, std: float = 0.01, bias: float = 0.0):
    """Orthogonal weight initialization — standard for stable RL training."""
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, bias)
    return layer


# ── Residual MLP block ───────────────────────────────────────────────────────
class ResBlock(nn.Module):
    """
    Pre-activation residual MLP block:
        h = LayerNorm(x); h = Linear(h); h = ReLU(h)
        h = Linear(h); h = ReLU(h)
        return x + h
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm  = nn.LayerNorm(dim)
        self.fc1   = layer_init(nn.Linear(dim, dim), std=0.5)
        self.fc2   = layer_init(nn.Linear(dim, dim), std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return x + h     # residual skip connection


# ── ActorCritic network ──────────────────────────────────────────────────────
class ActorCritic(nn.Module):
    """
    Shared-trunk ResNet-MLP with separate policy and value heads.

    Args:
        obs_dim    : length of the flat state vector (176 for our env)
        action_dim : flat action space size (1210 for our 121-cell board)
        hidden_dim : trunk width (default 512)
        n_blocks   : number of residual blocks (default 4)
    """

    def __init__(self, obs_dim: int, action_dim: int,
                 hidden_dim: int = 512, n_blocks: int = 4):
        super().__init__()

        # ── Stem: project input to hidden dimension ─────────────────────────
        self.stem = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim), std=1.0),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── Trunk: stack of residual blocks ─────────────────────────────────
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(n_blocks)
        ])

        # ── Policy head ─────────────────────────────────────────────────────
        # Two-layer head: project trunk → 256, then to action_dim.
        # Final layer uses small std so initial logits are near-uniform
        # → uniform exploration at training start.
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 256), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

        # ── Value head ──────────────────────────────────────────────────────
        # Two-layer head: project trunk → 256, then to scalar value.
        # std=1.0 on the final layer — value can be any magnitude.
        self.value_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 256), std=1.0),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, obs: torch.Tensor) -> tuple:
        """
        Args:
            obs: (batch, obs_dim) float tensor
        Returns:
            logits: (batch, action_dim)  — raw, UNMASKED
            value : (batch,)
        """
        h = self.stem(obs)
        for block in self.blocks:
            h = block(h)

        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

    # ── act (used during rollout) ────────────────────────────────────────────
    def act(self, obs: torch.Tensor, action_mask: torch.Tensor) -> tuple:
        """
        Sample an action given a single observation and binary mask.

        Args:
            obs        : (obs_dim,) or (1, obs_dim) float tensor
            action_mask: (action_dim,) binary tensor (1=valid, 0=invalid)

        Returns:
            action_id : int
            log_prob  : scalar tensor
            entropy   : scalar tensor
            value     : scalar tensor
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        logits, value = self.forward(obs)
        logits = logits.squeeze(0)       # (action_dim,)
        value  = value.squeeze(0)        # scalar

        # Mask out illegal actions before sampling
        masked_logits = logits.clone()
        masked_logits[action_mask == 0] = -1e9

        dist     = Categorical(logits=masked_logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        return action.item(), log_prob, entropy, value

    # ── act deterministically (used at deployment) ───────────────────────────
    def act_greedy(self, obs: torch.Tensor, action_mask: torch.Tensor) -> int:
        """
        Pick the highest-probability legal action (no sampling).
        Used at evaluation / deployment time for deterministic behavior.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.forward(obs)
            logits = logits.squeeze(0).clone()
            logits[action_mask == 0] = -1e9
            return int(torch.argmax(logits).item())

    # ── evaluate (used during PPO update) ────────────────────────────────────
    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor,
                 action_masks: torch.Tensor) -> tuple:
        """
        Re-evaluate stored (obs, action) pairs during PPO update.

        Args:
            obs          : (batch, obs_dim)
            actions      : (batch,)  long tensor
            action_masks : (batch, action_dim)  binary tensor

        Returns:
            log_probs : (batch,)
            entropies : (batch,)
            values    : (batch,)
        """
        logits, values = self.forward(obs)

        masked_logits = logits.clone()
        masked_logits[action_masks == 0] = -1e9

        dist      = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()

        return log_probs, entropies, values

    # ── parameter count helper ───────────────────────────────────────────────
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
