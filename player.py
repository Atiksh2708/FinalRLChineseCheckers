# =============================================================
# player.py — TOURNAMENT CLIENT WITH TRAINED PPO AGENT
# Based on the professor's template. Only the PLAYING LOGIC
# block is changed; all networking/RPC behavior is preserved.
# =============================================================

import os
import json
import random
import socket
import time
import argparse
from typing import Dict, Any

import numpy as np
import torch

from env import ChineseCheckersEnv, decode_action
from model import ActorCritic
from checkers_pins import Pin

HOST = "127.0.0.1"
PORT = 50555
DEBUG_NET = os.getenv("DEBUG_NET", "0") not in ("0", "", "false", "False")

# ── Multi-axis dispatch ──────────────────────────────────────────────────────
# Each opposite-color pair on the hex board is one of three rotational axes.
# We train a separate specialist model per axis (see train.py CFG.color_pair).
# At deployment, the assigned color determines which model loads.
#
# Edit these paths to point at your final checkpoints.
# COLOR_TO_CHECKPOINT = {
#     'red'       : "checkpoints/model_iter1000.pt",
#     'blue'      : "checkpoints/model_iter1000.pt",
#     'lawn green': "checkpoints_axis2/model_iter1000.pt",
#     'gray0'     : "checkpoints_axis2/model_iter1000.pt",
#     'yellow'    : "checkpoints_axis3/model_best.pt",
#     'purple'    : "checkpoints_axis3/model_best.pt",
# }
COLOR_TO_CHECKPOINT = {
    'red'       : "final_models/axis1/model_iter1000.pt",
    'blue'      : "final_models/axis1/model_iter1000.pt",
    'lawn green': "final_models/axis2/model_iter1000.pt",
    'gray0'     : "final_models/axis2/model_iter1000.pt",
    'yellow'    : "final_models/axis3/model_best.pt",
    'purple'    : "final_models/axis3/model_best.pt",
}

# Fallback if a specific axis's checkpoint is missing (e.g. axis C never
# finished training). At least the red↔blue model partly recognizes the
# board; better than a crash.
# FALLBACK_CHECKPOINT = "checkpoints/model_iter1000.pt"
FALLBACK_CHECKPOINT = "final_models/axis1/model_iter1000.pt"

USE_RULE_REARRANGEMENT = True   # match training-time behavior


def debug(*args):
    if DEBUG_NET:
        print("[NET]", *args)


def rpc(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Send JSON to server and receive JSON reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(10.0)
    try:
        s.connect((HOST, PORT))
    except Exception as e:
        return {"ok": False, "error": f"connect-failed: {e}"}

    s.sendall(json.dumps(payload).encode("utf-8"))
    data = s.recv(1_000_000)
    s.close()

    if not data:
        return {"ok": False, "error": "no-response"}

    try:
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": f"bad-json: {e}"}


# =============================================================
# Simple renderer for the server's JSON board (optional)
# =============================================================
def render_json_board(state):
    pins = state.get("pins", {})
    print("=== BOARD STATE ===")
    for colour, indices in pins.items():
        print(f"{colour}: {indices}")
    print("===================")


# =============================================================
# AGENT GLUE — everything new lives in this section
# =============================================================
class Agent:
    """
    Holds the trained model + a local ChineseCheckersEnv used to compute the
    state vector and action mask from each turn's server state.

    On each turn:
      1. sync_to_server(server_pins, my_color) rebuilds the local env to
         match the server's authoritative pin positions
      2. choose_move() runs rule-based rearrangement first, then the agent
      3. The caller submits the returned (pin_id, to_index) via RPC
    """

    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AGENT] device={self.device}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        obs_dim    = ckpt["obs_dim"]
        action_dim = ckpt["action_dim"]
        hidden_dim = ckpt.get("hidden_dim", 512)
        n_blocks   = ckpt.get("n_blocks", 4)

        self.model = ActorCritic(
            obs_dim=obs_dim, action_dim=action_dim,
            hidden_dim=hidden_dim, n_blocks=n_blocks,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        print(f"[AGENT] loaded {checkpoint_path}  "
              f"obs_dim={obs_dim} action_dim={action_dim} "
              f"hidden={hidden_dim} blocks={n_blocks} "
              f"iter={ckpt.get('iteration', '?')}")

        # n_players=6 so the HexBoard / BFS cache is built once; we
        # overwrite env.assigned and env.pins each turn from server state.
        self.env = ChineseCheckersEnv(n_players=6)
        self.env.reset(pins_advanced=0)

    def sync_to_server(self, server_pins_by_color: Dict[str, list],
                       my_color: str) -> None:
        """Rebuild local env's pin positions + caches from the server state."""
        env = self.env

        # Clear occupancy
        for cell in env.board.cells:
            cell.occupied = False

        # Determine the colors actually in this game
        new_assigned = sorted(server_pins_by_color.keys())

        # Re-create Pin objects at the server-reported positions.
        # Pin.__init__ sets board.cells[axialindex].occupied = True.
        new_pins = []
        for color in new_assigned:
            positions = server_pins_by_color[color]
            for pin_id, axial_idx in enumerate(positions):
                new_pins.append(Pin(env.board, int(axial_idx),
                                    id=pin_id, color=color))
        env.pins        = new_pins
        env.assigned    = new_assigned
        env.num_cells   = len(env.board.cells)
        env.done        = False
        env._step_count = 0

        if my_color not in new_assigned:
            raise RuntimeError(f"my_color={my_color} not in {new_assigned}")
        env.current_idx = new_assigned.index(my_color)

        # Target cache: each color's destination is the opposite-color home zone
        env._target_cache = {}
        for color in new_assigned:
            opp = env.board.colour_opposites[color]
            env._target_cache[color] = frozenset(env.board.axial_of_colour(opp))

        # Goal-depth map (used by the rearrangement rule)
        env._goal_depths = {}
        for color in new_assigned:
            target      = env._target_cache[color]
            source_home = list(env.board.axial_of_colour(color))
            raw = {t: min(env._bfs_dist(t, h) for h in source_home)
                   for t in target}
            min_d = min(raw.values()) if raw else 0
            env._goal_depths[color] = {t: d - min_d for t, d in raw.items()}

        # pin_start_dists: matches training (pins_advanced=0 → from home cells)
        env._pin_start_dists = {}
        for color in new_assigned:
            target       = env._target_cache[color]
            home_indices = env.board.axial_of_colour(color)
            env._pin_start_dists[color] = [
                min(env._bfs_dist(h, t) for t in target)
                for h in home_indices
            ]

    def choose_move(self, my_color: str, use_rule: bool = True):
        """
        Returns (pin_id, dest_idx, source_label) for the move to play.
        Returns (None, None, None) if the agent's mask is empty (caller
        should fall back to a server-legal move).
        """
        env = self.env

        # 1. Rule-based rearrangement first (matches training)
        if use_rule:
            rule = env.find_rearrangement_move(my_color)
            if rule is not None:
                return int(rule[0]), int(rule[1]), "RULE"

        # 2. Agent (greedy)
        mask = env.build_action_mask(my_color)
        if mask.sum() == 0:
            return None, None, None

        state  = env._get_state()
        obs_t  = torch.tensor(state, dtype=torch.float32, device=self.device)
        mask_t = torch.tensor(mask,  dtype=torch.float32, device=self.device)
        action_id = self.model.act_greedy(obs_t, mask_t)
        pin_id, dest_idx = decode_action(action_id, env.num_cells)
        return int(pin_id), int(dest_idx), "AGENT"


# =============================================================
# Main client loop  (structure preserved from the professor's template)
# =============================================================
def main():
    global HOST, PORT
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="override the per-color checkpoint dispatch; "
                         "load this checkpoint regardless of assigned color")
    ap.add_argument("--name", default=None)
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--auto-start", action="store_true",
                    help="skip the ENTER prompt before sending START")
    ap.add_argument("--no-rule", action="store_true",
                    help="disable the rule-based rearrangement")
    args = ap.parse_args()
    HOST, PORT = args.host, args.port
    use_rule = USE_RULE_REARRANGEMENT and not args.no_rule

    # NOTE: We do NOT load the model yet. We need the assigned color first
    # so the multi-axis dispatcher can pick the right specialist. Model load
    # happens after JOIN below, once we know which axis we're on.

    timeoutnotice_move = -1
    print("==== Player ====")
    if args.name:
        name = args.name
        print(f"Name: {name}")
    else:
        name = input("Enter name: ").strip()
        if not name:
            return

    # JOIN GAME
    r = rpc({"op": "join", "player_name": name})
    if not r.get("ok"):
        print("JOIN ERROR:", r.get("error"))
        return

    game_id = r["game_id"]
    player_id = r["player_id"]
    colour = r["colour"]

    print(f"Joined game {game_id} as {colour}")

    # ── Load the right specialist model for this color ──────────────────────
    if args.checkpoint is not None:
        # Manual override (useful for testing one specific checkpoint)
        ckpt_path = args.checkpoint
        print(f"[DISPATCH] override checkpoint: {ckpt_path}")
    else:
        ckpt_path = COLOR_TO_CHECKPOINT.get(colour)
        if ckpt_path is None or not os.path.exists(ckpt_path):
            print(f"[DISPATCH] no checkpoint for color {colour!r} "
                  f"(tried {ckpt_path}); using fallback {FALLBACK_CHECKPOINT}")
            ckpt_path = FALLBACK_CHECKPOINT
        else:
            print(f"[DISPATCH] color={colour} → {ckpt_path}")
    try:
        agent = Agent(ckpt_path)
    except Exception as e:
        print(f"[DISPATCH] FATAL: could not load {ckpt_path} ({e}); "
              f"falling back to {FALLBACK_CHECKPOINT}")
        agent = Agent(FALLBACK_CHECKPOINT)

    # Wait until game ready
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") in ("READY_TO_START", "PLAYING"):
            break
        print("Waiting for players...")
        time.sleep(0.5)

    if args.auto_start:
        print("Sending START (auto)...")
    else:
        input("Press ENTER to send START...")
    rpc({"op": "start", "game_id": game_id, "player_id": player_id})
    print("Sent START")

    # Wait until PLAYING
    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if st.get("state", {}).get("status") == "PLAYING":
            break
        time.sleep(0.5)

    print("=== GAME STARTED ===\n")

    last_move_seen = 0

    while True:
        st = rpc({"op": "get_state", "game_id": game_id})
        if not st.get("ok"):
            print("Error:", st.get("error"))
            return

        state = st["state"]

        # Timeout messages
        if state.get("turn_timeout_notice") and timeoutnotice_move < state.get("move_count"):
            print("⚠ TIMEOUT:", state["turn_timeout_notice"])
            timeoutnotice_move = state.get("move_count")

        # Finished?
        if state["status"] == "FINISHED":
            print("\n=== GAME FINISHED ===")
            print("FINAL SCORES:")
            for pl in state["players"]:
                sc = pl.get("score")
                if sc:
                    print(
                        f"{pl['name']} ({pl['colour']}): "
                        f"{sc['final_score']:.1f} "
                        f"[time={sc['time_score']:.1f}, "
                        f"moves({sc['moves']})={sc['move_score']:.1f}, "
                        f"pins={sc['pin_goal_score']:.1f}, "
                        f"dist={sc['distance_score']:.1f}]"
                    )
            print("======================")
            break

        # Show last move
        if state["move_count"] > last_move_seen:
            mv = state.get("last_move")
            if mv:
                print(
                    f"MOVE: {mv['by']} ({mv['colour']}) "
                    f"{mv['from']}→{mv['to']}  [{mv['move_ms']:.1f}ms]"
                )
            last_move_seen = state["move_count"]

        # If it's our turn, run the agent
        if state.get("current_turn_colour") == colour and state["status"] == "PLAYING":
            print("\nMy turn")
            '''------------PLAYING LOGIC-----------'''
            t0 = time.perf_counter()

            # 1. Sync local env to the server's authoritative state
            try:
                agent.sync_to_server(state.get("pins", {}), colour)
                # 2. Pick a move: rule → agent
                pid, to_index, src = agent.choose_move(colour, use_rule=use_rule)
            except Exception as e:
                print(f"  ⚠ agent failure ({e}); falling back to server-legal move")
                pid, to_index, src = None, None, "FALLBACK"

            # 3. Fallback: if the agent couldn't produce a move (empty mask
            #    or a sync failure), pull legal moves from the server and
            #    pick one randomly. Should rarely happen but defensive.
            if pid is None:
                legal_req = rpc({
                    "op": "get_legal_moves",
                    "game_id": game_id,
                    "player_id": player_id,
                })
                if not legal_req.get("ok"):
                    print("Error requesting legal moves:", legal_req.get("error"))
                    time.sleep(0.1)
                    continue
                legal_moves = legal_req.get("legal_moves", {})
                movable = [(int(p), moves) for p, moves in legal_moves.items() if moves]
                if not movable:
                    print("No legal moves available.")
                    time.sleep(0.1)
                    continue
                pid, mvs = random.choice(movable)
                to_index = int(random.choice(mvs))
                src = "FALLBACK"

            inf_ms = (time.perf_counter() - t0) * 1000
            print(f"  [{src}] pin={pid} → {to_index}  (infer={inf_ms:.0f}ms)")
            '''-----------------PLAYING LOGIC----------------'''

            mv = rpc({
                "op": "move",
                "game_id": game_id,
                "player_id": player_id,
                "pin_id": pid,
                "to_index": to_index,
            })
            if not mv.get("ok"):
                print("Move rejected:", mv.get("error"))
                # Last-resort retry with a server-legal random move
                legal_req = rpc({
                    "op": "get_legal_moves",
                    "game_id": game_id,
                    "player_id": player_id,
                })
                if legal_req.get("ok"):
                    movable = [(int(p), m) for p, m in
                               legal_req.get("legal_moves", {}).items() if m]
                    if movable:
                        pid2, mvs2 = random.choice(movable)
                        ti2 = int(random.choice(mvs2))
                        rpc({"op": "move", "game_id": game_id,
                             "player_id": player_id,
                             "pin_id": pid2, "to_index": ti2})
                        print(f"  fallback played pin {pid2} → {ti2}")
            else:
                if mv.get("status") == "WIN":
                    print("YOU WIN!")
                    print(mv.get("msg"))
                elif mv.get("status") == "DRAW":
                    print("DRAW")
                    print(mv.get("msg"))

            # We just acted. Don't sleep — immediately poll for the next
            # state. The wall-clock matters: a 0.5s sleep here was eating
            # ~half a second per turn off the 60-second game budget.
            continue

        # Not our turn — poll less aggressively but stay responsive enough
        # to act when our turn comes around. 0.1s balances server load
        # and responsiveness.
        time.sleep(0.1)


if __name__ == "__main__":
    main()