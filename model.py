
# model.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# INIT
# ============================================================

def layer_init(layer,
               std=0.01,
               bias=0.0):

    nn.init.orthogonal_(
        layer.weight,
        std,
    )

    nn.init.constant_(
        layer.bias,
        bias,
    )

    return layer


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResBlock(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.fc1 = layer_init(
            nn.Linear(dim, dim),
            std=0.5,
        )

        self.fc2 = layer_init(
            nn.Linear(dim, dim),
            std=0.5,
        )

    def forward(self, x):

        h = self.norm(x)

        h = F.relu(
            self.fc1(h)
        )

        h = F.relu(
            self.fc2(h)
        )

        return x + h


# ============================================================
# ACTOR CRITIC
# ============================================================

class ActorCritic(nn.Module):

    def __init__(self,
                 obs_dim,
                 action_dim,
                 hidden_dim=512,
                 n_blocks=4):

        super().__init__()

        self.obs_dim = obs_dim

        self.action_dim = action_dim

        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        self.stem = nn.Sequential(

            layer_init(
                nn.Linear(
                    obs_dim,
                    hidden_dim,
                ),
                std=1.0,
            ),

            nn.LayerNorm(
                hidden_dim
            ),

            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Residual trunk
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([

            ResBlock(hidden_dim)

            for _ in range(n_blocks)

        ])

        # ----------------------------------------------------
        # Policy head
        # ----------------------------------------------------

        self.policy_head = nn.Sequential(

            layer_init(
                nn.Linear(
                    hidden_dim,
                    256,
                ),
                std=1.0,
            ),

            nn.ReLU(),

            layer_init(
                nn.Linear(
                    256,
                    action_dim,
                ),
                std=0.01,
            ),
        )

        # ----------------------------------------------------
        # Value head
        # ----------------------------------------------------

        self.value_head = nn.Sequential(

            layer_init(
                nn.Linear(
                    hidden_dim,
                    256,
                ),
                std=1.0,
            ),

            nn.ReLU(),

            layer_init(
                nn.Linear(
                    256,
                    1,
                ),
                std=1.0,
            ),
        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, obs):

        """
        obs:
            (batch, obs_dim)

        returns:
            logits:
                (batch, action_dim)

            values:
                (batch,)
        """

        h = self.stem(obs)

        for block in self.blocks:
            h = block(h)

        logits = self.policy_head(h)

        values = self.value_head(
            h
        ).squeeze(-1)

        return logits, values

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(self,
                 obs,
                 actions,
                 action_masks):

        """
        PPO evaluation step.
        """

        logits, values = self.forward(obs)

        masked_logits = logits.clone()

        masked_logits[
            action_masks == 0
        ] = -1e9

        dist = torch.distributions.Categorical(
            logits=masked_logits
        )

        log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy()

        return (
            log_probs,
            entropy,
            values,
        )

    # ========================================================
    # SAMPLE ACTIONS
    # ========================================================

    def act(self,
            obs,
            action_masks):

        """
        Vectorized action sampling.

        obs:
            (N, obs_dim)

        action_masks:
            (N, action_dim)
        """

        logits, values = self.forward(obs)

        masked_logits = logits.clone()

        masked_logits[
            action_masks == 0
        ] = -1e9

        dist = torch.distributions.Categorical(
            logits=masked_logits
        )

        actions = dist.sample()

        log_probs = dist.log_prob(
            actions
        )

        entropy = dist.entropy()

        return (
            actions,
            log_probs,
            entropy,
            values,
        )

    # ========================================================
    # GREEDY ACTIONS
    # ========================================================

    def act_greedy(self,
                   obs,
                   action_masks):

        logits, _ = self.forward(obs)

        masked_logits = logits.clone()

        masked_logits[
            action_masks == 0
        ] = -1e9

        return torch.argmax(
            masked_logits,
            dim=-1,
        )

    # ========================================================
    # PARAM COUNT
    # ========================================================

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )
