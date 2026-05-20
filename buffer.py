"""
buffer.py — Rollout buffer for on-policy PPO.

Fixed-size buffer that collects T transitions, computes GAE advantages,
and yields shuffled mini-batches for PPO updates.

Stores `old_values` (the critic's prediction at rollout time) so that
ppo.py can apply value-loss clipping (a stability trick that prevents
the critic from making big jumps in any single mini-batch).
"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """
    Args:
        rollout_len : number of timesteps per rollout (T)
        obs_dim     : flat state vector length
        action_dim  : flat action space size
        gamma       : discount factor
        gae_lambda  : GAE smoothing parameter (λ)
        device      : torch device for batch tensors
    """

    def __init__(self, rollout_len: int, obs_dim: int, action_dim: int,
                 gamma: float = 0.99, gae_lambda: float = 0.95,
                 device: torch.device = torch.device("cpu")):

        self.rollout_len = rollout_len
        self.obs_dim     = obs_dim
        self.action_dim  = action_dim
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.device      = device

        self._reset_storage()

    def _reset_storage(self):
        T, D, A = self.rollout_len, self.obs_dim, self.action_dim

        self.obs          = np.zeros((T, D), dtype=np.float32)
        self.actions      = np.zeros(T,      dtype=np.int64)
        self.log_probs    = np.zeros(T,      dtype=np.float32)
        self.rewards      = np.zeros(T,      dtype=np.float32)
        self.values       = np.zeros(T,      dtype=np.float32)
        self.dones        = np.zeros(T,      dtype=np.float32)
        self.action_masks = np.zeros((T, A), dtype=np.float32)

        self.advantages   = np.zeros(T, dtype=np.float32)
        self.returns      = np.zeros(T, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done, action_mask):
        """Store one transition."""
        assert self.ptr < self.rollout_len, \
            "Buffer is full — call compute_gae() then reset()"

        def to_np(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        i = self.ptr
        self.obs[i]          = to_np(obs)
        self.actions[i]      = int(action)
        self.log_probs[i]    = float(to_np(log_prob))
        self.rewards[i]      = float(reward)
        self.values[i]       = float(to_np(value))
        self.dones[i]        = float(done)
        self.action_masks[i] = to_np(action_mask)

        self.ptr += 1

    def compute_gae(self, last_value: float):
        """
        Compute GAE advantages and discounted returns.
        Must be called after the rollout is complete (ptr == rollout_len).
        """
        assert self.ptr == self.rollout_len, \
            f"Buffer not full ({self.ptr}/{self.rollout_len})"

        gae = 0.0
        for t in reversed(range(self.rollout_len)):
            next_non_terminal = 1.0 - self.dones[t]
            next_value        = (last_value if t == self.rollout_len - 1
                                 else self.values[t + 1])

            delta = (self.rewards[t]
                     + self.gamma * next_value * next_non_terminal
                     - self.values[t])

            gae = delta + self.gamma * self.gae_lambda \
                        * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

        # Normalize advantages — zero mean, unit variance
        adv = self.advantages
        if adv.std() > 1e-8:
            self.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        else:
            self.advantages = adv - adv.mean()

    def get_batches(self, batch_size: int):
        """
        Yield random mini-batches as torch tensors on self.device.

        Yields tuples of (obs, actions, old_log_probs, advantages, returns,
                          action_masks, old_values).
        """
        T       = self.rollout_len
        indices = np.random.permutation(T)

        # Convert entire buffer to tensors once — cheaper than per-batch
        obs_t  = torch.tensor(self.obs,         device=self.device)
        act_t  = torch.tensor(self.actions,      device=self.device)
        lp_t   = torch.tensor(self.log_probs,    device=self.device)
        adv_t  = torch.tensor(self.advantages,   device=self.device)
        ret_t  = torch.tensor(self.returns,      device=self.device)
        mask_t = torch.tensor(self.action_masks, device=self.device)
        val_t  = torch.tensor(self.values,       device=self.device)

        for start in range(0, T, batch_size):
            idx = indices[start: start + batch_size]
            yield (obs_t[idx], act_t[idx], lp_t[idx],
                   adv_t[idx], ret_t[idx], mask_t[idx], val_t[idx])

    def reset(self):
        """Clear buffer for next rollout."""
        self._reset_storage()

    @property
    def is_full(self) -> bool:
        return self.ptr == self.rollout_len
