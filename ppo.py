"""
ppo.py — PPO update with policy and value clipping.

Standard clipped PPO objective:
    L_policy = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]
    L_value  = max( MSE(V, R), MSE(clip(V), R) )
    L_total  = L_policy + c1·L_value - c2·H[π]

where r = π_new / π_old (the probability ratio).
"""

# ppo.py

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_


class PPOUpdater:

    def __init__(self,
                 model,
                 lr=3e-4,
                 clip_eps=0.2,
                 c_value=0.5,
                 c_entropy=0.01,
                 max_grad_norm=0.5,
                 n_epochs=4,
                 batch_size=2048,
                 device=torch.device("cpu")):

        self.model = model

        self.clip_eps = clip_eps

        self.c_value = c_value

        self.c_entropy = c_entropy

        self.max_grad_norm = max_grad_norm

        self.n_epochs = n_epochs

        self.batch_size = batch_size

        self.device = device

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            eps=1e-5,
        )

    def update(self, buffer):

        stats = {
            "policy_loss": [],
            "value_loss": [],
            "entropy": [],
            "total_loss": [],
            "approx_kl": [],
        }

        # ====================================================
        # PPO EPOCHS
        # ====================================================

        for _ in range(self.n_epochs):

            for (
                obs,
                actions,
                old_log_probs,
                advantages,
                returns,
                action_masks,
                old_values,
            ) in buffer.get_batches(
                self.batch_size
            ):

                # --------------------------------------------
                # Re-evaluate actions
                # --------------------------------------------

                (
                    new_log_probs,
                    entropy,
                    values,
                ) = self.model.evaluate(
                    obs,
                    actions,
                    action_masks,
                )

                # --------------------------------------------
                # PPO ratio
                # --------------------------------------------

                log_ratio = (
                    new_log_probs
                    - old_log_probs
                )

                ratio = log_ratio.exp()

                # --------------------------------------------
                # Approx KL
                # --------------------------------------------

                with torch.no_grad():

                    approx_kl = (
                        (ratio - 1)
                        - log_ratio
                    ).mean()

                # --------------------------------------------
                # Policy loss
                # --------------------------------------------

                surr1 = (
                    ratio
                    * advantages
                )

                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_eps,
                        1.0 + self.clip_eps,
                    )
                    * advantages
                )

                policy_loss = -torch.min(
                    surr1,
                    surr2,
                ).mean()

                # --------------------------------------------
                # Value loss
                # --------------------------------------------

                values_clipped = (
                    old_values
                    + torch.clamp(
                        values - old_values,
                        -self.clip_eps,
                        self.clip_eps,
                    )
                )

                value_loss_unclipped = (
                    (values - returns) ** 2
                )

                value_loss_clipped = (
                    (values_clipped - returns) ** 2
                )

                value_loss = 0.5 * torch.max(
                    value_loss_unclipped,
                    value_loss_clipped,
                ).mean()

                # --------------------------------------------
                # Entropy bonus
                # --------------------------------------------

                entropy_loss = entropy.mean()

                # --------------------------------------------
                # Total loss
                # --------------------------------------------

                total_loss = (
                    policy_loss
                    + self.c_value * value_loss
                    - self.c_entropy * entropy_loss
                )

                # --------------------------------------------
                # Backprop
                # --------------------------------------------

                self.optimizer.zero_grad()

                total_loss.backward()

                clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                # --------------------------------------------
                # Stats
                # --------------------------------------------

                stats["policy_loss"].append(
                    policy_loss.item()
                )

                stats["value_loss"].append(
                    value_loss.item()
                )

                stats["entropy"].append(
                    entropy_loss.item()
                )

                stats["total_loss"].append(
                    total_loss.item()
                )

                stats["approx_kl"].append(
                    approx_kl.item()
                )

        return {
            k: sum(v) / len(v)
            for k, v in stats.items()
        }

