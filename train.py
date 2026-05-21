# train.py

from __future__ import annotations

import os
import copy
import time
import random
from collections import deque

import numpy as np
import torch

from env import ChineseCheckersEnv
from vec_env import VecEnv

from model import ActorCritic
from buffer import RolloutBuffer
from ppo import PPOUpdater

from opponents import (
    RandomOpponent,
    HeuristicOpponent,
    ModelOpponent,
)


# ============================================================
# CONFIG
# ============================================================

CFG = dict(

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    n_players=2,

    total_timesteps=15_000_000,

    num_envs=128,

    # --------------------------------------------------------
    # PPO
    # --------------------------------------------------------

    rollout_len=128,

    batch_size=8192,

    n_epochs=4,

    gamma=0.99,

    gae_lambda=0.95,

    lr=3e-4,

    clip_eps=0.2,

    c_value=0.5,

    c_entropy_start=0.03,

    c_entropy_end=0.01,

    max_grad_norm=0.5,

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------

    hidden_dim=512,

    n_blocks=4,

    # --------------------------------------------------------
    # Self-play
    # --------------------------------------------------------

    snapshot_every=50,

    snapshot_pool_size=5,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_interval=5,

    save_interval=50,

    # color_pair=["red", "blue"],
    # checkpoint_dir="checkpoints_axis1",
    # checkpoint_dir="final_models/axis1",

    # color_pair=["lawn green", "gray0"],
    # checkpoint_dir="checkpoints_axis2",
    # checkpoint_dir="final_models/axis2",

    color_pair=["yellow", "purple"],
    checkpoint_dir="checkpoints_axis3",
    # checkpoint_dir="final_models/axis3",


    device="cuda",
)


# ============================================================
# UTILS
# ============================================================

def make_snapshot(model):

    snap = copy.deepcopy(model)

    snap.eval()

    for p in snap.parameters():
        p.requires_grad_(False)

    return snap


def sample_opponent(model,
                    frozen_pool,
                    random_op,
                    heuristic_op):

    choices = [
        "current",
        "frozen",
        "heuristic",
        "random",
    ]

    probs = [
        0.20,
        0.25,
        0.35,
        0.20,
    ]

    if len(frozen_pool) == 0:

        probs = [
            0.40,
            0.00,
            0.40,
            0.20,
        ]

    choice = random.choices(
        choices,
        probs,
    )[0]

    if choice == "current":
        return ModelOpponent(
            model,
            label="current",
        )

    if choice == "frozen":
        return random.choice(frozen_pool)

    if choice == "heuristic":
        return heuristic_op

    return random_op


# ============================================================
# TRAIN
# ============================================================

def train():

    device = torch.device(CFG["device"])

    print(f"[TRAIN] device={device}")

    if device.type == "cuda":
        print(
            f"[TRAIN] GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Dummy env for dimensions
    # --------------------------------------------------------

    dummy_env = ChineseCheckersEnv(
        n_players=CFG["n_players"],
    )

    obs = dummy_env.reset()

    obs_dim = len(obs)

    action_dim = dummy_env.action_dim

    print(
        f"[TRAIN] obs_dim={obs_dim}  "
        f"action_dim={action_dim}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ActorCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=CFG["hidden_dim"],
        n_blocks=CFG["n_blocks"],
    ).to(device)

    print(
        f"[TRAIN] params="
        f"{model.num_parameters():,}"
    )

    # --------------------------------------------------------
    # PPO updater
    # --------------------------------------------------------

    updater = PPOUpdater(
        model=model,
        lr=CFG["lr"],
        clip_eps=CFG["clip_eps"],
        c_value=CFG["c_value"],
        c_entropy=CFG["c_entropy_start"],
        max_grad_norm=CFG["max_grad_norm"],
        n_epochs=CFG["n_epochs"],
        batch_size=CFG["batch_size"],
        device=device,
    )

    # --------------------------------------------------------
    # Rollout buffer
    # --------------------------------------------------------

    buffer = RolloutBuffer(
        num_envs=CFG["num_envs"],
        rollout_len=CFG["rollout_len"],
        obs_dim=obs_dim,
        action_dim=action_dim,
        gamma=CFG["gamma"],
        gae_lambda=CFG["gae_lambda"],
        device=device,
    )

    # --------------------------------------------------------
    # Opponent pool
    # --------------------------------------------------------

    random_op = RandomOpponent()

    heuristic_op = HeuristicOpponent()

    frozen_pool = []

    def opponent_sampler(env, color):

        return sample_opponent(
            model,
            frozen_pool,
            random_op,
            heuristic_op,
        )

    # --------------------------------------------------------
    # Vectorized envs
    # --------------------------------------------------------
    print(f"[INFO] Training for {CFG['color_pair']}")
    vec_env = VecEnv(
        num_envs=CFG["num_envs"],
        env_kwargs=dict(
            n_players=CFG["n_players"],
            color_pair=CFG["color_pair"],
        ),
        opponent_sampler=opponent_sampler,
    )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_steps = 0

    iteration = 0

    recent_rewards = deque(maxlen=100)

    start_time = time.time()

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while total_steps < CFG["total_timesteps"]:

        buffer.reset()

        # ----------------------------------------------------
        # Entropy annealing
        # ----------------------------------------------------

        progress = min(
            total_steps / CFG["total_timesteps"],
            1.0,
        )

        cur_entropy = (
            CFG["c_entropy_start"] * (1.0 - progress)
            + CFG["c_entropy_end"] * progress
        )

        updater.c_entropy = cur_entropy

        # ====================================================
        # COLLECT ROLLOUT
        # ====================================================

        for _ in range(CFG["rollout_len"]):

            obs_batch = vec_env.get_states()

            mask_batch = vec_env.get_masks()

            obs_t = torch.tensor(
                obs_batch,
                dtype=torch.float32,
                device=device,
            )

            mask_t = torch.tensor(
                mask_batch,
                dtype=torch.float32,
                device=device,
            )

            # ------------------------------------------------
            # Policy inference
            # ------------------------------------------------

            with torch.no_grad():

                logits, values = model(obs_t)

                masked_logits = logits.clone()

                masked_logits[
                    mask_t == 0
                ] = -1e9

                dist = torch.distributions.Categorical(
                    logits=masked_logits
                )

                actions = dist.sample()

                log_probs = dist.log_prob(actions)

            # ------------------------------------------------
            # Step envs
            # ------------------------------------------------

            next_states, rewards, dones, infos = (
                vec_env.step(
                    actions.cpu().numpy()
                )
            )

            # ------------------------------------------------
            # Store transitions
            # ------------------------------------------------

            buffer.add(
                obs=obs_batch,
                action=actions.cpu().numpy(),
                log_prob=log_probs.cpu().numpy(),
                reward=rewards,
                value=values.cpu().numpy(),
                done=dones,
                action_mask=mask_batch,
            )

            total_steps += CFG["num_envs"]

            recent_rewards.extend(
                rewards.tolist()
            )

        # ====================================================
        # BOOTSTRAP VALUES
        # ====================================================

        with torch.no_grad():

            last_obs = torch.tensor(
                vec_env.get_states(),
                dtype=torch.float32,
                device=device,
            )

            _, last_values = model(last_obs)

        # ====================================================
        # COMPUTE GAE
        # ====================================================

        buffer.compute_gae(
            last_values.cpu().numpy()
        )

        # ====================================================
        # PPO UPDATE
        # ====================================================

        stats = updater.update(buffer)

        iteration += 1

        # ====================================================
        # SNAPSHOT POOL
        # ====================================================

        if iteration % CFG["snapshot_every"] == 0:

            snap = make_snapshot(model)

            frozen_pool.append(
                ModelOpponent(
                    snap,
                    label="frozen",
                )
            )

            if (
                len(frozen_pool)
                > CFG["snapshot_pool_size"]
            ):
                frozen_pool.pop(0)

            print(
                f"[SNAPSHOT] "
                f"pool={len(frozen_pool)}"
            )

        # ====================================================
        # LOGGING
        # ====================================================

        if iteration % CFG["log_interval"] == 0:

            elapsed = (
                time.time()
                - start_time
            )

            mean_reward = (
                np.mean(recent_rewards)
                if recent_rewards
                else 0.0
            )

            print(
                f"iter={iteration:5d}  "
                f"steps={total_steps:>10,}  "
                f"reward={mean_reward:.2f}  "
                f"ploss={stats['policy_loss']:.4f}  "
                f"vloss={stats['value_loss']:.4f}  "
                f"ent={stats['entropy']:.4f}  "
                f"kl={stats['approx_kl']:.5f}  "
                f"pool={len(frozen_pool)}  "
                f"t={elapsed:.0f}s"
            )

        # ====================================================
        # SAVE
        # ====================================================

        if iteration % CFG["save_interval"] == 0:

            os.makedirs(
                CFG["checkpoint_dir"],
                exist_ok=True,
            )

            path = os.path.join(
                CFG["checkpoint_dir"],
                f"model_iter{iteration}.pt",
            )

            torch.save(
                {
                    "model_state":
                        model.state_dict(),

                    "optimizer_state":
                        updater.optimizer.state_dict(),

                    "iteration":
                        iteration,

                    "total_steps":
                        total_steps,
                },
                path,
            )

            print(f"[SAVE] {path}")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    train()

