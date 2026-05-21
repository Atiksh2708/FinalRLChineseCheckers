"""
buffer.py — Rollout buffer for on-policy PPO.

Fixed-size buffer that collects T transitions, computes GAE advantages,
and yields shuffled mini-batches for PPO updates.

Stores `old_values` (the critic's prediction at rollout time) so that
ppo.py can apply value-loss clipping (a stability trick that prevents
the critic from making big jumps in any single mini-batch).
"""

# buffer.py

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """
    PPO rollout buffer for vectorized environments.

    Stores:
        T timesteps
        N parallel environments

    Shape convention:
        (T, N, ...)
    """

    def __init__(self,
                 num_envs: int,
                 rollout_len: int,
                 obs_dim: int,
                 action_dim: int,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 device: torch.device = torch.device("cpu")):

        self.num_envs = num_envs
        self.rollout_len = rollout_len
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.device = device

        self.reset()

    def reset(self):

        T = self.rollout_len
        N = self.num_envs

        self.obs = np.zeros(
            (T, N, self.obs_dim),
            dtype=np.float32,
        )

        self.actions = np.zeros(
            (T, N),
            dtype=np.int64,
        )

        self.log_probs = np.zeros(
            (T, N),
            dtype=np.float32,
        )

        self.rewards = np.zeros(
            (T, N),
            dtype=np.float32,
        )

        self.values = np.zeros(
            (T, N),
            dtype=np.float32,
        )

        self.dones = np.zeros(
            (T, N),
            dtype=np.float32,
        )

        self.action_masks = np.zeros(
            (T, N, self.action_dim),
            dtype=np.float32,
        )

        self.advantages = np.zeros(
            (T, N),
            dtype=np.float32,
        )

        self.returns = np.zeros(
            (T, N),
            dtype=np.float32,
        )

        self.ptr = 0

    def add(self,
            obs,
            action,
            log_prob,
            reward,
            value,
            done,
            action_mask):

        assert self.ptr < self.rollout_len, (
            "RolloutBuffer overflow."
        )

        t = self.ptr

        self.obs[t] = obs
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.rewards[t] = reward
        self.values[t] = value
        self.dones[t] = done
        self.action_masks[t] = action_mask

        self.ptr += 1

    def compute_gae(self, last_values):

        """
        last_values:
            shape (N,)
        """

        T = self.rollout_len
        N = self.num_envs

        gae = np.zeros(N, dtype=np.float32)

        for t in reversed(range(T)):

            if t == T - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            nonterminal = 1.0 - self.dones[t]

            delta = (
                self.rewards[t]
                + self.gamma
                * next_values
                * nonterminal
                - self.values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * nonterminal
                * gae
            )

            self.advantages[t] = gae

        self.returns = (
            self.advantages
            + self.values
        )

        # Normalize advantages globally
        flat_adv = self.advantages.reshape(-1)

        adv_mean = flat_adv.mean()
        adv_std = flat_adv.std()

        self.advantages = (
            self.advantages - adv_mean
        ) / (adv_std + 1e-8)

    def get_batches(self, batch_size):

        """
        Flatten:
            (T, N, ...) → (T*N, ...)
        """

        total = (
            self.rollout_len
            * self.num_envs
        )

        obs = self.obs.reshape(
            total,
            self.obs_dim,
        )

        actions = self.actions.reshape(total)

        log_probs = self.log_probs.reshape(total)

        advantages = self.advantages.reshape(total)

        returns = self.returns.reshape(total)

        masks = self.action_masks.reshape(
            total,
            self.action_dim,
        )

        values = self.values.reshape(total)

        indices = np.random.permutation(total)

        obs_t = torch.tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        )

        actions_t = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device,
        )

        log_probs_t = torch.tensor(
            log_probs,
            dtype=torch.float32,
            device=self.device,
        )

        advantages_t = torch.tensor(
            advantages,
            dtype=torch.float32,
            device=self.device,
        )

        returns_t = torch.tensor(
            returns,
            dtype=torch.float32,
            device=self.device,
        )

        masks_t = torch.tensor(
            masks,
            dtype=torch.float32,
            device=self.device,
        )

        values_t = torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

        for start in range(0, total, batch_size):

            idx = indices[
                start:start + batch_size
            ]

            yield (
                obs_t[idx],
                actions_t[idx],
                log_probs_t[idx],
                advantages_t[idx],
                returns_t[idx],
                masks_t[idx],
                values_t[idx],
            )

    @property
    def is_full(self):

        return (
            self.ptr
            >= self.rollout_len
        )

