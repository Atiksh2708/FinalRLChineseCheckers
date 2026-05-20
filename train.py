"""
train.py — PPO with POPULATION-BASED OPPONENT MIXING.

═══════════════════════════════════════════════════════════════════════════
WHY THIS DESIGN
═══════════════════════════════════════════════════════════════════════════

Previous version: vanilla self-play (same model plays both sides every
move). Two problems:

  • Non-stationarity. The opponent improves with the learner every update,
    so V(s) is a moving target. Training plateaus.

  • Distribution shift at deployment. The learner only ever sees opponents
    that play like itself. Against a random or differently-tuned tournament
    agent the board states look unfamiliar and the agent stalls.

Fix: per-episode opponent sampling from a heterogeneous population:

  • current   — live learner (no_grad). Classic self-play signal.
  • frozen    — past snapshot of the learner. Stationary; learnable.
  • heuristic — greedy forward-progress (see opponents.py). Models a
                typical hand-crafted tournament agent.
  • random    — uniform-random over legal moves. Models the field's floor.

Mix is configurable via cfg["opponent_mix"]. The frozen pool starts empty
and grows up to `snapshot_pool_size` as snapshots are taken.

═══════════════════════════════════════════════════════════════════════════
KEY CORRECTNESS CHANGES vs PREVIOUS train.py
═══════════════════════════════════════════════════════════════════════════

1. ONLY learner transitions enter the buffer. Previously, every env.step
   was buffered regardless of whose turn it was, which was OK in vanilla
   self-play (same model both sides) but is incorrect with heterogeneous
   opponents — those transitions are from the opponent's perspective, not
   the learner's.

2. The rule-based rearrangement (env.find_rearrangement_move) now fires
   ONLY on the learner's turn. Previously it fired for every color, which
   meant opponents got an inductive bias real tournament opponents won't
   have. We want opponents to model what we'll actually face.

3. active_color is sampled randomly each episode (not always assigned[0]).
   The state is player-centric so this is mostly cosmetic, but it does
   broaden the distribution of opponent-color configurations the learner
   sees during training.

═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import copy
import time
import random as pyrandom
from collections import deque
from typing import Dict, List

import numpy as np
import torch

from env import ChineseCheckersEnv, NUM_PINS_PER_PLAYER, encode_action
from model import ActorCritic
from buffer import RolloutBuffer
from ppo import PPOUpdater
from opponents import RandomOpponent, HeuristicOpponent, ModelOpponent


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
CFG = dict(
    # ── Game ────────────────────────────────────────────────────────────────
    n_players       = 2,
    total_timesteps = 5_000_000,

    # ── Color axis (NEW) ─────────────────────────────────────────────────────
    # Pick ONE of these per training run:
    #   None                       → default ('red', 'blue', ...)  [axis A]
    #   ['lawn green', 'gray0']    → axis B
    #   ['yellow', 'purple']       → axis C
    color_pair      = ['lawn green', 'gray0']   ,

    # ── PPO ────────────────────────────────────────────────────────────────
    rollout_len     = 2048,
    n_epochs        = 4,
    batch_size      = 256,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    c_value         = 0.5,
    max_grad_norm   = 0.5,
    c_entropy_start = 0.03,
    c_entropy_end   = 0.01,

    # ── Network ─────────────────────────────────────────────────────────────
    hidden_dim = 512,
    n_blocks   = 4,

    # ── Opponent mixing (probabilities must sum to 1.0) ─────────────────────
    # If the frozen pool is empty (early in training), its mass is
    # re-distributed proportionally over the other types automatically.
    opponent_mix = {
        "current":   0.20,
        "frozen":    0.25,
        "heuristic": 0.35,
        "random":    0.20,
    },

    # ── Frozen snapshot pool ────────────────────────────────────────────────
    snapshot_every        = 50,    # take a new snapshot every N iterations
    snapshot_pool_size    = 5,     # cap the pool to keep memory bounded
    min_iter_for_snapshot = 50,    # don't snapshot until the learner has
                                   # trained a bit — early snapshots are noise

    # ── Logging / checkpointing ─────────────────────────────────────────────
    log_interval    = 5,
    save_interval   = 50,
    checkpoint_dir  = "checkpoints_axis2",
    device          = "auto",
    resume_from     = "checkpoints_axis2/model_best.pt",        # e.g. "checkpoints/model_iter1000.pt"
)


# ─────────────────────────────────────────────────────────────────────────────
def pick_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def count_pins_home_for_color(env: ChineseCheckersEnv, color: str) -> int:
    target = env._target_cache[color]
    return sum(1 for p in env.pins
               if p.color == color and p.axialindex in target)


def make_frozen_snapshot(model: ActorCritic) -> ActorCritic:
    """Deep-copy the model into a frozen, eval-mode, grad-disabled snapshot."""
    snap = copy.deepcopy(model)
    snap.eval()
    for p in snap.parameters():
        p.requires_grad_(False)
    return snap


def sample_opponent(opponent_mix: Dict[str, float],
                    model: ActorCritic,
                    frozen_pool: List[ModelOpponent],
                    random_op: RandomOpponent,
                    heuristic_op: HeuristicOpponent):
    """
    Sample one opponent instance for one non-learner color.

    If the frozen pool is empty, "frozen" mass is redistributed
    proportionally over the other opponent types.
    """
    types   = list(opponent_mix.keys())
    weights = list(opponent_mix.values())

    if not frozen_pool and "frozen" in types:
        i = types.index("frozen")
        frozen_mass = weights[i]
        weights[i]  = 0.0
        rest_total  = sum(weights)
        if rest_total > 0:
            weights = [w * (1.0 + frozen_mass / rest_total) for w in weights]
        else:
            # All mass was on "frozen" — degenerate config; use current.
            return ModelOpponent(model, label="current")

    choice = pyrandom.choices(types, weights=weights, k=1)[0]

    if choice == "current":
        return ModelOpponent(model, label="current")
    if choice == "frozen":
        return pyrandom.choice(frozen_pool)
    if choice == "heuristic":
        return heuristic_op
    if choice == "random":
        return random_op
    return random_op   # unreachable


# ─────────────────────────────────────────────────────────────────────────────
# Training entry point
# ─────────────────────────────────────────────────────────────────────────────
def train(cfg: dict = CFG):
    device = pick_device(cfg["device"])
    print(f"[TRAIN] device={device}")
    if device.type == "cuda":
        print(f"[TRAIN] GPU: {torch.cuda.get_device_name(0)}")

    # ── env + dims ──────────────────────────────────────────────────────────
    env   = ChineseCheckersEnv(n_players=cfg["n_players"],
                                color_pair=cfg.get("color_pair"))
    state = env.reset(pins_advanced=0)
    obs_dim    = len(state)
    action_dim = env.action_dim

    print(f"[TRAIN] n_players={cfg['n_players']}  obs_dim={obs_dim}  "
          f"action_dim={action_dim}")
    print(f"[TRAIN] opponent_mix={cfg['opponent_mix']}")

    # ── model ───────────────────────────────────────────────────────────────
    model = ActorCritic(
        obs_dim    = obs_dim,
        action_dim = action_dim,
        hidden_dim = cfg["hidden_dim"],
        n_blocks   = cfg["n_blocks"],
    ).to(device)
    print(f"[TRAIN] params={model.num_parameters():,}")

    # ── buffer & updater ────────────────────────────────────────────────────
    buffer = RolloutBuffer(
        rollout_len = cfg["rollout_len"],
        obs_dim     = obs_dim,
        action_dim  = action_dim,
        gamma       = cfg["gamma"],
        gae_lambda  = cfg["gae_lambda"],
        device      = device,
    )
    updater = PPOUpdater(
        model         = model,
        lr            = cfg["lr"],
        clip_eps      = cfg["clip_eps"],
        c_value       = cfg["c_value"],
        c_entropy     = cfg["c_entropy_start"],
        max_grad_norm = cfg["max_grad_norm"],
        n_epochs      = cfg["n_epochs"],
        batch_size    = cfg["batch_size"],
        device        = device,
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ── Resume from checkpoint ──────────────────────────────────────────────
    start_iter, start_steps = 0, 0
    if cfg.get("resume_from") and os.path.exists(cfg["resume_from"]):
        ckpt = torch.load(cfg["resume_from"], map_location=device,
                          weights_only=False)
        if ckpt.get("obs_dim", obs_dim) != obs_dim:
            print(f"[TRAIN] WARNING: checkpoint obs_dim mismatch "
                  f"({ckpt.get('obs_dim')} vs {obs_dim}); starting fresh.")
        else:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                try:
                    updater.optimizer.load_state_dict(ckpt["optimizer_state"])
                except Exception as e:
                    print(f"[TRAIN] optimizer state failed to load ({e}); "
                          f"using fresh optimizer.")
            start_iter  = ckpt.get("iteration", 0)
            start_steps = ckpt.get("total_steps", 0)
            print(f"[TRAIN] Resumed from {cfg['resume_from']} "
                  f"(iter={start_iter} steps={start_steps:,})")
    else:
        print("[TRAIN] Fresh start (no resume).")

    # ── opponent population ─────────────────────────────────────────────────
    random_op    = RandomOpponent()
    heuristic_op = HeuristicOpponent()
    frozen_pool: List[ModelOpponent] = []

    # If resuming, seed the pool with a snapshot of the loaded model so
    # there's a frozen opponent available from iteration 1 onward.
    if start_iter > 0:
        seed_snap = make_frozen_snapshot(model)
        frozen_pool.append(ModelOpponent(seed_snap, label="frozen"))
        print(f"[TRAIN] Seeded frozen pool with resumed model.")

    # ── counters & telemetry ────────────────────────────────────────────────
    total_steps = start_steps
    iteration   = start_iter

    episode_outcomes  = deque(maxlen=100)
    episode_pins_home = deque(maxlen=20)
    episode_lengths   = deque(maxlen=20)
    per_opp_wins      = {"current":   [0, 0],
                         "frozen":    [0, 0],
                         "heuristic": [0, 0],
                         "random":    [0, 0]}    # [wins, total]
    total_wins     = 0
    total_episodes = 0

    best_pins_home = -1.0
    t_start = time.time()

    # ── episode setup ───────────────────────────────────────────────────────
    def new_episode():
        env_new   = ChineseCheckersEnv(n_players=cfg["n_players"],
                                        color_pair=cfg.get("color_pair"))
        state_new = env_new.reset(pins_advanced=0)
        # Rotate active color randomly so the learner sees all perspectives.
        active_color_new = pyrandom.choice(env_new.assigned)
        # One opponent per non-learner color.
        opps = {}
        for c in env_new.assigned:
            if c == active_color_new:
                continue
            opps[c] = sample_opponent(cfg["opponent_mix"], model, frozen_pool,
                                      random_op, heuristic_op)
        return env_new, state_new, active_color_new, opps

    def end_episode(env_e, active_color_e, opponents_e):
        """Record episode stats and start a new one. Returns (env, state, active, opps)."""
        nonlocal total_wins, total_episodes
        won_active = env_e._check_win(active_color_e)
        total_wins     += int(won_active)
        total_episodes += 1
        episode_outcomes.append(1 if won_active else 0)
        episode_pins_home.append(count_pins_home_for_color(env_e, active_color_e))
        episode_lengths.append(env_e._step_count)
        for _, opp in opponents_e.items():
            per_opp_wins[opp.name][1] += 1
            if won_active:
                per_opp_wins[opp.name][0] += 1
        return new_episode()

    env, state, active_color, opponents_by_color = new_episode()

    # ── main training loop ──────────────────────────────────────────────────
    while total_steps < cfg["total_timesteps"]:
        buffer.reset()

        # Linear entropy anneal across the whole training run.
        progress  = min(total_steps / cfg["total_timesteps"], 1.0)
        cur_c_ent = (cfg["c_entropy_start"] * (1.0 - progress)
                     + cfg["c_entropy_end"] * progress)
        updater.c_entropy = cur_c_ent

        # ── Rollout: keep playing until the buffer holds rollout_len ────────
        # learner transitions. Opponent moves change the board but do NOT
        # advance the buffer pointer.
        consecutive_no_moves = 0
        while not buffer.is_full:
            color_now       = env.current_color
            is_learner_turn = (color_now == active_color)

            # ── 1. Rule-based rearrangement — ONLY on the learner's turn ────
            if is_learner_turn:
                rule_move = env.find_rearrangement_move(active_color)
                if rule_move is not None:
                    pin_id, dest_idx = rule_move
                    rule_aid = encode_action(pin_id, dest_idx, env.num_cells)
                    next_state, _, done = env.step(rule_aid)
                    # Don't add to buffer — structural rule, not learner choice.
                    state = next_state
                    consecutive_no_moves = 0
                    if done:
                        env, state, active_color, opponents_by_color = \
                            end_episode(env, active_color, opponents_by_color)
                    continue

            # ── 2. Legal-move check ─────────────────────────────────────────
            mask = env.build_action_mask(color_now)
            if mask.sum() == 0:
                env._advance_turn()
                consecutive_no_moves += 1
                # Whole game stuck → end as a draw.
                if consecutive_no_moves >= len(env.assigned):
                    env.done = True
                    env, state, active_color, opponents_by_color = \
                        end_episode(env, active_color, opponents_by_color)
                    consecutive_no_moves = 0
                continue
            consecutive_no_moves = 0

            # ── 3. Act ──────────────────────────────────────────────────────
            if is_learner_turn:
                # Learner — buffer this transition.
                obs_t  = torch.tensor(state, dtype=torch.float32, device=device)
                mask_t = torch.tensor(mask,  dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_id, log_prob, entropy, value = model.act(obs_t, mask_t)

                next_state, reward, done = env.step(action_id)

                buffer.add(
                    obs         = state,
                    action      = action_id,
                    log_prob    = log_prob,
                    reward      = reward,
                    value       = value,
                    done        = done,
                    action_mask = mask,
                )
                state        = next_state
                total_steps += 1
            else:
                # Opponent — DO NOT buffer.
                opp = opponents_by_color[color_now]
                action_id = opp.select_action(env, state, mask, device)
                next_state, _, done = env.step(action_id)
                state = next_state

            if done:
                env, state, active_color, opponents_by_color = \
                    end_episode(env, active_color, opponents_by_color)

        # ── Bootstrap V(s_T) for GAE ────────────────────────────────────────
        if buffer.ptr == 0:
            iteration += 1
            continue

        if buffer.dones[buffer.ptr - 1]:
            last_value = 0.0
        else:
            with torch.no_grad():
                last_obs_t = torch.tensor(state, dtype=torch.float32,
                                          device=device).unsqueeze(0)
                _, lv = model(last_obs_t)
                last_value = lv.item()

        # Compute GAE and run PPO update. If the buffer ended up partially
        # filled (e.g., because of weird stall combinations), do a partial
        # GAE pass — this matches the partial-fill handling from the
        # previous train.py and the old environment.py.
        actual_len = buffer.ptr
        if actual_len < buffer.rollout_len:
            gae = 0.0
            for t in reversed(range(actual_len)):
                nonterm = 1.0 - buffer.dones[t]
                nv = (last_value if t == actual_len - 1
                      else buffer.values[t + 1])
                delta = (buffer.rewards[t]
                         + cfg["gamma"] * nv * nonterm
                         - buffer.values[t])
                gae = delta + cfg["gamma"] * cfg["gae_lambda"] * nonterm * gae
                buffer.advantages[t] = gae
            for t in range(actual_len):
                buffer.returns[t] = buffer.advantages[t] + buffer.values[t]
            adv = buffer.advantages[:actual_len]
            if adv.std() > 1e-8:
                buffer.advantages[:actual_len] = (adv - adv.mean()) / (adv.std() + 1e-8)
            else:
                buffer.advantages[:actual_len] = adv - adv.mean()
            saved_len = buffer.rollout_len
            buffer.rollout_len = actual_len
            stats = updater.update(buffer)
            buffer.rollout_len = saved_len
        else:
            buffer.compute_gae(last_value)
            stats = updater.update(buffer)

        iteration += 1

        # ── Snapshot rotation: grow / cycle the frozen pool ─────────────────
        if (iteration % cfg["snapshot_every"] == 0
                and iteration >= cfg["min_iter_for_snapshot"]):
            snap = make_frozen_snapshot(model)
            frozen_pool.append(ModelOpponent(snap, label="frozen"))
            if len(frozen_pool) > cfg["snapshot_pool_size"]:
                frozen_pool.pop(0)
            print(f"  ❄ snapshot (pool={len(frozen_pool)})")

        # ── Logging ─────────────────────────────────────────────────────────
        if iteration % cfg["log_interval"] == 0:
            elapsed   = time.time() - t_start
            mean_pins  = float(np.mean(episode_pins_home)) if episode_pins_home else 0.0
            mean_eplen = float(np.mean(episode_lengths))   if episode_lengths   else 0.0
            recent_wr  = (sum(episode_outcomes) / len(episode_outcomes)
                          if episode_outcomes else 0.0)

            opp_parts = []
            for name in ("current", "frozen", "heuristic", "random"):
                w, t = per_opp_wins[name]
                wr   = (w / t) if t > 0 else 0.0
                opp_parts.append(f"{name[:4]}={wr:.2f}({t})")
            opp_str = " ".join(opp_parts)

            print(
                f"iter={iteration:5d}  steps={total_steps:>9,}  "
                f"wr={recent_wr:.2f}  "
                f"pins={mean_pins:.1f}/10  "
                f"eplen={mean_eplen:.0f}  "
                f"vloss={stats['value_loss']:.2f}  "
                f"ent={stats['entropy']:.3f}  "
                f"cent={cur_c_ent:.4f}  "
                f"kl={stats['approx_kl']:.4f}  "
                f"pool={len(frozen_pool)}  "
                f"{opp_str}  "
                f"t={elapsed:.0f}s"
            )

        # ── Checkpoint: scheduled ───────────────────────────────────────────
        if iteration % cfg["save_interval"] == 0:
            path = os.path.join(cfg["checkpoint_dir"],
                                f"model_iter{iteration}.pt")
            torch.save({
                "iteration"      : iteration,
                "total_steps"    : total_steps,
                "model_state"    : model.state_dict(),
                "optimizer_state": updater.optimizer.state_dict(),
                "obs_dim"        : obs_dim,
                "action_dim"     : action_dim,
                "n_players"      : cfg["n_players"],
                "hidden_dim"     : cfg["hidden_dim"],
                "n_blocks"       : cfg["n_blocks"],
                "color_pair"     : cfg.get("color_pair"),
            }, path)
            print(f"  [save → {path}]")

        # ── Checkpoint: best pins_home ──────────────────────────────────────
        if episode_pins_home and float(np.mean(episode_pins_home)) > best_pins_home:
            best_pins_home = float(np.mean(episode_pins_home))
            best_path = os.path.join(cfg["checkpoint_dir"], "model_best.pt")
            torch.save({
                "iteration"      : iteration,
                "total_steps"    : total_steps,
                "model_state"    : model.state_dict(),
                "optimizer_state": updater.optimizer.state_dict(),
                "obs_dim"        : obs_dim,
                "action_dim"     : action_dim,
                "n_players"      : cfg["n_players"],
                "hidden_dim"     : cfg["hidden_dim"],
                "n_blocks"       : cfg["n_blocks"],
                "color_pair"     : cfg.get("color_pair"),
                "best_pins_home" : best_pins_home,
            }, best_path)

    # ── Final ───────────────────────────────────────────────────────────────
    final = os.path.join(cfg["checkpoint_dir"], "model_final.pt")
    torch.save({
        "iteration"   : iteration,
        "total_steps" : total_steps,
        "model_state" : model.state_dict(),
        "obs_dim"     : obs_dim,
        "action_dim"  : action_dim,
        "n_players"   : cfg["n_players"],
        "hidden_dim"  : cfg["hidden_dim"],
        "n_blocks"    : cfg["n_blocks"],
    }, final)
    print(f"\n[TRAIN] complete → {final}")
    print(f"[TRAIN] best pins_home: {best_pins_home:.2f}")
    print(f"[TRAIN] wins: {total_wins}/{total_episodes}")


if __name__ == "__main__":
    train()