"""
opponents.py — Heterogeneous opponents for population-based PPO training.

All opponents implement the same interface:

    select_action(env, state, mask, device) -> int

so the training rollout can swap opponent types without branching on class.
Models how typical tournament agents are likely to play, giving the
learner exposure to a wider distribution of board states than self-play
alone would produce.
"""

from __future__ import annotations

import numpy as np
import torch

from env import encode_action


# ─────────────────────────────────────────────────────────────────────────────
# Random — uniform over legal moves
# ─────────────────────────────────────────────────────────────────────────────
class RandomOpponent:
    """
    Pure-random opponent. Models the floor of the field: teams that didn't
    get an agent working in time and submit the template's random-move logic.

    The agent needs to know how to win against this kind of opponent because
    in a 2-player game against a random mover the win is largely about
    not stalling — committing forward when the moves are there.
    """

    name = "random"

    def select_action(self, env, state, mask, device):
        valid = np.where(mask == 1.0)[0]
        if len(valid) == 0:
            return 0  # caller should have skipped, defensive
        return int(np.random.choice(valid))


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic — greedy forward progress
# ─────────────────────────────────────────────────────────────────────────────
class HeuristicOpponent:
    """
    Greedy forward-progress opponent. Models the middle of the field:
    teams that wrote a hand-crafted agent (no RL training).

    Move scoring:
        score = (prev_dist - new_dist)          # distance progress to nearest
                                                # EMPTY target cell
              + chain_bonus * max(0, jump - 1)  # prefer chain hops

    `prev_dist` and `new_dist` use BFS true distance; `jump` is the hex
    distance from pin to destination (1 = single step, >1 = a chain hop).
    Ties are broken randomly.

    This opponent typically races straight at the goal zone without
    coordinating its pins, which means it leaves laggards behind and is
    beatable in the long game — but in the short game it produces strong
    pressure that the learner must respond to.
    """

    name = "heuristic"

    def __init__(self, chain_bonus: float = 0.3):
        self.chain_bonus = chain_bonus

    def select_action(self, env, state, mask, device):
        color  = env.current_color
        target = env._target_cache[color]

        # Pin-to-nearest-EMPTY-target — matches the env's reward computation.
        # Pins already in goal "occupy" target cells from the perspective of
        # other same-color pins still trying to enter.
        my_pin_cells = {p.axialindex for p in env.pins if p.color == color}
        empty_targets = [t for t in target if t not in my_pin_cells]
        targets_for_dist = empty_targets if empty_targets else list(target)

        best_score = -float("inf")
        best_aids: list = []

        for pin, dest in env.get_valid_moves(color):
            aid = encode_action(pin.id, dest, env.num_cells)
            if mask[aid] == 0:
                continue   # masked (e.g. goal-pin lock)

            prev_d = min(env._bfs_dist(pin.axialindex, t)
                         for t in targets_for_dist)
            new_d  = min(env._bfs_dist(dest, t)
                         for t in targets_for_dist)
            progress = prev_d - new_d

            jump = env._hex_dist(pin.axialindex, dest)
            chain = self.chain_bonus * max(0, jump - 1)

            score = progress + chain

            if score > best_score:
                best_score = score
                best_aids  = [aid]
            elif score == best_score:
                best_aids.append(aid)

        if not best_aids:
            # No score-positive move under the mask — fall back to a
            # uniform pick from whatever the mask allows.
            valid = np.where(mask == 1.0)[0]
            if len(valid) == 0:
                return 0
            return int(np.random.choice(valid))

        return int(np.random.choice(best_aids))


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper — wraps live or frozen ActorCritic as an opponent
# ─────────────────────────────────────────────────────────────────────────────
class ModelOpponent:
    """
    Wraps an ActorCritic (live or frozen snapshot) as an opponent.

    Uses .act() (stochastic) rather than .act_greedy() during training —
    a small amount of opponent stochasticity widens the state distribution
    the learner sees, which is the whole point of having a population.

    `label` is used for per-opponent-type winrate logging in train.py.
    Anything not equal to "current" is bucketed under "frozen" for stats.
    """

    def __init__(self, model, label: str = "frozen"):
        self.model = model
        self.name  = "current" if label == "current" else "frozen"

    def select_action(self, env, state, mask, device):
        obs_t  = torch.tensor(state, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask,  dtype=torch.float32, device=device)
        with torch.no_grad():
            action_id, _, _, _ = self.model.act(obs_t, mask_t)
        return action_id
