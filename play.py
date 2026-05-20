"""
play.py — Local evaluation modes for the trained agent.

Modes:
    python play.py self                      # agent plays all colors (n_players=2)
    python play.py vs_random [n_games]       # agent vs random opponents
    python play.py vs_heuristic [n_games]    # agent vs heuristic opponents (THE KEY ONE)
    python play.py interactive               # human plays one color
    python play.py six_player [n_games]      # 6-player tournament-style eval

Args:
    --checkpoint PATH    override CHECKPOINT_PATH
    --cpu                force CPU (useful when training is using the GPU)
    --players N          override N_PLAYERS (only respected by self / vs_random / vs_heuristic / interactive)

Examples:
    # Evaluate iter750 against 5 heuristic opponents, on CPU so it doesn't fight training for the GPU:
    python play.py six_player 5 --checkpoint checkpoints/model_iter750_PROTECTED.pt --cpu

    # Quick sanity check vs random at 2-player:
    python play.py vs_random 10 --cpu
"""

from __future__ import annotations

import sys
import time
import argparse

import numpy as np
import torch
from env import (
    ChineseCheckersEnv, decode_action, encode_action,
    NUM_PINS_PER_PLAYER,
)
from model import ActorCritic
from opponents import RandomOpponent, HeuristicOpponent


# ── Defaults (overridable via CLI) ───────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/model_best.pt"
N_PLAYERS       = 2


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrapper — loads a checkpoint and exposes a single choose_action method
# ─────────────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = ActorCritic(
            obs_dim    = ckpt["obs_dim"],
            action_dim = ckpt["action_dim"],
            hidden_dim = ckpt.get("hidden_dim", 512),
            n_blocks   = ckpt.get("n_blocks", 4),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.iteration = ckpt.get("iteration", "?")
        print(f"[AGENT] loaded {checkpoint_path}  iter={self.iteration}  device={device}")

    def choose_action(self, env: ChineseCheckersEnv, deterministic: bool = True) -> int:
        """Pick an action for env.current_color. Falls back to random if mask empty."""
        # Honor the rule first — matches training and tournament deployment.
        rule = env.find_rearrangement_move()
        if rule is not None:
            pin_id, dest_idx = rule
            return encode_action(pin_id, dest_idx, env.num_cells)

        mask = env.build_action_mask()
        if mask.sum() == 0:
            return -1   # caller will skip turn

        state  = env._get_state()
        obs_t  = torch.tensor(state, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask,  dtype=torch.float32, device=self.device)
        if deterministic:
            return self.model.act_greedy(obs_t, mask_t)
        with torch.no_grad():
            action_id, _, _, _ = self.model.act(obs_t, mask_t)
            return action_id


def pin_count(env: ChineseCheckersEnv, color: str) -> int:
    return sum(1 for p in env.pins
               if p.color == color and p.axialindex in env._target_cache[color])


# ─────────────────────────────────────────────────────────────────────────────
# Mode: self-play (agent plays every color)
# ─────────────────────────────────────────────────────────────────────────────
def self_play(agent: Agent, n_players: int, verbose: bool = True):
    env = ChineseCheckersEnv(n_players=n_players)
    env.reset(pins_advanced=0)

    move_count_per_color = {c: 0 for c in env.assigned}
    t_start = time.time()

    while not env.done:
        color = env.current_color
        mask  = env.build_action_mask()
        if mask.sum() == 0 and env.find_rearrangement_move() is None:
            if verbose:
                print(f"  {color}: no legal moves, skip")
            env._advance_turn()
            continue

        action_id = agent.choose_action(env, deterministic=True)
        if action_id < 0:
            env._advance_turn()
            continue

        pin_id, dest = decode_action(action_id, env.num_cells)
        env.step(action_id)
        move_count_per_color[color] += 1

        if verbose and env._step_count % 40 == 0:
            print(f"  step {env._step_count}: {color} pin {pin_id} → {dest}")

    elapsed = time.time() - t_start
    print(f"\nGame over after {env._step_count} steps ({elapsed:.1f}s)")
    print("\n=== STANDINGS ===")
    for c in env.assigned:
        won = env._check_win(c)
        marker = "  WINNER!" if won else ""
        print(f"  {c:12s}  pins_home={pin_count(env, c):2d}/10   "
              f"moves={move_count_per_color[c]:3d}{marker}")


# ─────────────────────────────────────────────────────────────────────────────
# Generic opponent-eval driver (used by vs_random, vs_heuristic, six_player)
# ─────────────────────────────────────────────────────────────────────────────
def _run_eval(agent: Agent, n_players: int, n_games: int,
              opponent_factory, opponent_name: str,
              force_color: str = None):
    """
    Run n_games games. Agent plays env.assigned[0] (or force_color if set);
    all other colors are handled by opponent_factory() (one instance per
    opponent color, freshly created each game).

    When force_color is set and n_players==2, the env is constructed with
    the matching opposite-color pair so _target_cache contains the right
    colors. This is what lets us evaluate axis B/C specialists on their
    native axes without crashing.
    """
    agent_wins   = 0
    pin_totals   = []
    step_counts  = []
    finish_count = 0   # number of games that ended with a winner

    inference_times = []

    # Opposite-color map for 2-player axis-specific evaluation
    opposites = {'red':'blue', 'blue':'red',
                 'lawn green':'gray0', 'gray0':'lawn green',
                 'yellow':'purple', 'purple':'yellow'}

    for game_i in range(n_games):
        # If a specific color is forced and we're at 2-player, construct
        # the env with the matching opposite-color pair so _target_cache
        # has the right colors and home positions exist on the board.
        if force_color is not None and n_players == 2:
            color_pair = [force_color, opposites[force_color]]
            env = ChineseCheckersEnv(n_players=2, color_pair=color_pair)
        else:
            env = ChineseCheckersEnv(n_players=n_players)
        env.reset(pins_advanced=0)

        if force_color is not None:
            agent_color = force_color
        else:
            agent_color = env.assigned[0]
        opp_colors = [c for c in env.assigned if c != agent_color]
        opponents  = {c: opponent_factory() for c in opp_colors}

        consecutive_no_moves = 0

        while not env.done:
            color = env.current_color
            mask  = env.build_action_mask()

            # No legal moves: skip turn. If every player is stuck → draw.
            if mask.sum() == 0:
                # Allow rule firing for own color (it doesn't need mask)
                if color == agent_color and env.find_rearrangement_move() is not None:
                    pass  # falls through to agent.choose_action which calls rule
                else:
                    env._advance_turn()
                    consecutive_no_moves += 1
                    if consecutive_no_moves >= len(env.assigned):
                        env.done = True
                        break
                    continue
            consecutive_no_moves = 0

            if color == agent_color:
                t0 = time.perf_counter()
                action_id = agent.choose_action(env, deterministic=True)
                inference_times.append((time.perf_counter() - t0) * 1000)
                if action_id < 0:
                    env._advance_turn()
                    continue
            else:
                # Opponent picks; rule doesn't fire for opponents in eval
                # (we want the opponent to be itself, not a rule-assisted version).
                state = env._get_state()
                action_id = opponents[color].select_action(
                    env, state, mask, agent.device
                )

            env.step(action_id)

        # Tally
        my_pins = pin_count(env, agent_color)
        pin_totals.append(my_pins)
        step_counts.append(env._step_count)
        if env._check_win(agent_color):
            agent_wins += 1
        if any(env._check_win(c) for c in env.assigned):
            finish_count += 1

        # Per-game line: keep it short
        winner_str = "agent" if env._check_win(agent_color) else (
            "OTHER" if any(env._check_win(c) for c in env.assigned)
            else "timeout"
        )
        opp_pins = " ".join(
            f"{c[:3]}={pin_count(env, c):d}" for c in opp_colors
        )
        print(f"Game {game_i+1:2d}/{n_games}: "
              f"agent_pins={my_pins:2d}/10  steps={env._step_count:3d}  "
              f"opp_pins=[{opp_pins}]  → {winner_str}")

    # Summary
    print(f"\n=== SUMMARY: agent vs {opponent_name}  ({n_players}-player, "
          f"{n_games} games) ===")
    print(f"  Agent wins:        {agent_wins}/{n_games}  ({100*agent_wins/n_games:.0f}%)")
    print(f"  Games finishing:   {finish_count}/{n_games}")
    print(f"  Avg agent pins:    {np.mean(pin_totals):.1f}/10  "
          f"(min={min(pin_totals)}, max={max(pin_totals)})")
    print(f"  Avg game steps:    {np.mean(step_counts):.0f}")
    if inference_times:
        print(f"  Inference time:    "
              f"mean={np.mean(inference_times):.1f}ms  "
              f"max={max(inference_times):.1f}ms  "
              f"(n={len(inference_times)})")


# ─────────────────────────────────────────────────────────────────────────────
# Public modes
# ─────────────────────────────────────────────────────────────────────────────
def vs_random(agent: Agent, n_players: int, n_games: int,
              force_color: str = None):
    _run_eval(agent, n_players, n_games, RandomOpponent, "random",
              force_color=force_color)


def vs_heuristic(agent: Agent, n_players: int, n_games: int,
                 force_color: str = None):
    _run_eval(agent, n_players, n_games, HeuristicOpponent, "heuristic",
              force_color=force_color)


def six_player(agent: Agent, n_games: int):
    """6-player game, agent vs 5 heuristic opponents. Tournament-style stress test."""
    print("=" * 70)
    print("6-PLAYER EVALUATION — agent vs 5 heuristic opponents")
    print("This is the closest local approximation to your tournament setup.")
    print("=" * 70)
    _run_eval(agent, n_players=6, n_games=n_games,
              opponent_factory=HeuristicOpponent,
              opponent_name="heuristic (6p)")


def interactive(agent: Agent, n_players: int):
    env = ChineseCheckersEnv(n_players=n_players)
    env.reset(pins_advanced=0)
    human_color = env.assigned[0]
    print(f"\nYou are: {human_color}\nAgent plays: {env.assigned[1:]}\n")

    while not env.done:
        color = env.current_color
        mask  = env.build_action_mask()
        if mask.sum() == 0:
            print(f"{color}: no legal moves, skip"); env._advance_turn(); continue

        if color == human_color:
            env.print_board()
            print(f"\nYour turn ({color}).")
            my_pins = [p for p in env.pins if p.color == color]
            print("Your pins (id, position):",
                  [(p.id, p.axialindex) for p in my_pins])
            while True:
                inp = input("Enter (pin_id,dest_idx): ").strip()
                try:
                    inp = inp.replace("(", "").replace(")", "")
                    pid, dst = inp.split(",")
                    pid, dst = int(pid), int(dst)
                    aid = encode_action(pid, dst, env.num_cells)
                    if mask[aid] == 1.0:
                        action_id = aid; break
                    print("Illegal move. Try again.")
                except Exception:
                    print("Format: (pin_id, dest_idx) e.g. (3,45)")
        else:
            action_id = agent.choose_action(env, deterministic=True)
            if action_id < 0:
                env._advance_turn(); continue
            pid, dst = decode_action(action_id, env.num_cells)
            print(f"  {color} (agent) plays pin {pid} → {dst}")

        env.step(action_id)

    print("\n=== GAME OVER ===")
    env.print_board()
    for c in env.assigned:
        if env._check_win(c):
            print(f"\n{c} wins!"); break


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["self", "vs_random", "vs_heuristic",
                                     "six_player", "interactive"])
    ap.add_argument("n_games", nargs="?", type=int, default=5,
                    help="(for vs_* and six_player) number of games to play")
    ap.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    ap.add_argument("--players", type=int, default=N_PLAYERS,
                    help="n_players for non-six_player modes (six_player is always 6)")
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (useful if training is using the GPU)")
    ap.add_argument("--color", default=None,
                    help="force agent to play as this color "
                         "(e.g. 'red', 'lawn green', 'gray0'). At 2-player "
                         "this also forces the env to use the matching "
                         "opposite-color pair.")
    args = ap.parse_args()

    device = torch.device("cpu") if args.cpu else (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    agent = Agent(args.checkpoint, device=device)

    if args.mode == "self":
        self_play(agent, args.players, verbose=True)
    elif args.mode == "vs_random":
        vs_random(agent, args.players, args.n_games, force_color=args.color)
    elif args.mode == "vs_heuristic":
        vs_heuristic(agent, args.players, args.n_games, force_color=args.color)
    elif args.mode == "six_player":
        six_player(agent, args.n_games)
    elif args.mode == "interactive":
        interactive(agent, args.players)