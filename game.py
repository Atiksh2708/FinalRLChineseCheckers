# ==========================================================
# game.py — SERVER
# ==========================================================

import os
import json
import uuid
import time
import socket
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import math
import random
# Game logic imports
from checkers_board import HexBoard
from checkers_pins import Pin



# ==========================================================
# Utilities
# ==========================================================
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_path(game_id: str) -> str:
    os.makedirs("games", exist_ok=True)
    return os.path.join("games", f"game_{game_id}.log")


def write_log(game_id: str, msg: str):
    with open(log_path(game_id), "a", encoding="utf-8") as f:
        f.write(f"[{ts()}] {msg}\n")


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"json-encode-failed: {e}"})


# ==========================================================
# Game Constants
# ==========================================================
COLOUR_ORDER = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']
PRIMARY_COLOURS = ['red', 'lawn green', 'yellow']
random.shuffle(PRIMARY_COLOURS)
COMPLEMENT = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}
MAX_PLAYERS = 6
TURN_TIMEOUT_SEC = 10
GAME_TIME_LIMIT_SEC = 300


# ==========================================================
# Player Class
# ==========================================================
class Player:
    def __init__(self, pid: str, name: str, colour: str):
        self.player_id = pid
        self.name = name
        self.colour = colour
        self.ready = False
        self.status = "PLAYING"

        # Scoring stats
        self.move_count = 0
        self.time_taken_sec = 0.0


# ==========================================================
# Game Class
# ==========================================================
class Game:
    def __init__(self):
        self.game_id = str(uuid.uuid4())
        self.board = HexBoard()
        self.players: List[Player] = []
        self.pins_by_colour: Dict[str, List[Pin]] = {}
        self.status = "AVAILABLE"
        self.created_ts = ts()
        self.joined_primary_index = 0
        self.lock_joining = False

        # Timing
        self.total_start_ns = None
        self.turn_started_ns = None

        # Turn rotation
        self.turn_order: List[str] = []
        self.current_turn_index = 0

        # Moves
        self.move_count = 0
        self.move_times_ms: List[float] = []
        self.last_move = None
        self.turn_timeout_notice = None

        # Score dictionary
        self.scores: Dict[str, Dict[str, float]] = {}

    # ----------------------------------
    # Assign Colour
    # ----------------------------------
    def assign_colour(self) -> Optional[str]:
        n = len(self.players) + 1
        if n > MAX_PLAYERS:
            return None
        if n % 2 == 1:  # primary
            if self.joined_primary_index >= len(PRIMARY_COLOURS):
                return None
            return PRIMARY_COLOURS[self.joined_primary_index]
        else:
            primary = PRIMARY_COLOURS[self.joined_primary_index]
            self.joined_primary_index += 1
            return COMPLEMENT[primary]

    # ----------------------------------
    def init_pins(self, colour: str):
        if colour in self.pins_by_colour:
            return
        idxs = self.board.axial_of_colour(colour)[:10]
        self.pins_by_colour[colour] = [
            Pin(self.board, idxs[i], id=i, color=colour)
            for i in range(len(idxs))
        ]

    # ----------------------------------
    def compute_turn_order(self):
        present = [p.colour for p in self.players]
        first = present[0]
        if first in COLOUR_ORDER:
            idx = COLOUR_ORDER.index(first)
            rotated = COLOUR_ORDER[idx:] + COLOUR_ORDER[:idx]
        else:
            rotated = COLOUR_ORDER[:]
        self.turn_order = [c for c in rotated if c in present]
        self.current_turn_index = 0

    # ----------------------------------
    def current_turn_colour(self):
        if self.status != "PLAYING":
            return None
        if not self.turn_order:
            return None
        return self.turn_order[self.current_turn_index]

    # ----------------------------------
    def advance_turn(self):
        if self.turn_order:
            self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)
            self.turn_started_ns = time.perf_counter_ns()

    # ----------------------------------
    def ensure_time_limits(self):
        # Game time limit
        if self.total_start_ns:
            elapsed = (time.perf_counter_ns() - self.total_start_ns) / 1e9
            if elapsed > GAME_TIME_LIMIT_SEC:
                self.status = "FINISHED"
                self.turn_timeout_notice = "GAME TIME LIMIT REACHED."
                self.compute_scores()
                write_log(self.game_id, self.turn_timeout_notice)
                return

        # Per-turn timeout
        if self.status == "PLAYING" and self.turn_started_ns:
            turn_elapsed = (time.perf_counter_ns() - self.turn_started_ns) / 1e9
            if turn_elapsed > TURN_TIMEOUT_SEC:
                colour = self.current_turn_colour()
                self.turn_timeout_notice = (
                    f"Player with colour {colour} exceeded {TURN_TIMEOUT_SEC}s at move {self.move_count}. Turn skipped."
                )
                self.compute_scores()
                write_log(self.game_id, f"TURN TIMEOUT: {self.turn_timeout_notice}")
                self.advance_turn()

    # ----------------------------------
    def check_player_status(self, colour: str) -> str:
        opposite = self.board.colour_opposites[colour]
        pins = self.pins_by_colour[colour]

        # WIN: all pins reached opposite target zones
        if all(self.board.cells[p.axialindex].postype == opposite for p in pins):
            return "WIN"

        # DRAW: no possible moves
        if all(len(p.getPossibleMoves()) == 0 for p in pins):
            return "DRAW"

        return "PLAYING"

    # ----------------------------------
    # Scoring Logic
    # ----------------------------------
    def compute_scores(self):
        def axial_dist(a, b):
            dq = abs(a.q - b.q)
            dr = abs(a.r - b.r)
            ds = abs((-a.q - a.r) - (-b.q - b.r))
            return max(dq, dr, ds)

        for pl in self.players:
            colour = pl.colour
            pins = self.pins_by_colour[colour]
            opposite = self.board.colour_opposites[colour]

            # Time score
            time_score = max(0.0, 100.0 - pl.time_taken_sec) if pl.time_taken_sec > 0 else 0

            # Move score — asymmetric Gaussian
            move_score_func = lambda x: math.exp(-((x - 45) ** 2) /
                                                 (2 * ((4 if x < 45 else 18) ** 2)))
            move_score = move_score_func(pl.move_count) if pl.move_count > 0 else 0

            # Pins in goal
            pins_in_goal = sum(
                1 for p in pins
                if self.board.cells[p.axialindex].postype == opposite
            )
            pin_goal_score = pins_in_goal * 100.0

            # Distance score
            target_idxs = self.board.axial_of_colour(opposite)
            target_cells = [self.board.cells[i] for i in target_idxs]
            total_dist = 0
            for p in pins:
                if self.board.cells[p.axialindex].postype != opposite:
                    best = min(axial_dist(self.board.cells[p.axialindex], tgt)
                               for tgt in target_cells)
                    total_dist += best
            distance_score = max(0.0, 200.0 - total_dist) if pl.move_count > 0 else 0

            final_score = time_score + move_score + pin_goal_score + distance_score

            self.scores[pl.player_id] = {
                "final_score": final_score,
                "time_score": time_score,
                "move_score": move_score,
                "pin_goal_score": pin_goal_score,
                "distance_score": distance_score,
                "moves": pl.move_count,
                "pins_in_goal": pins_in_goal,
                "total_distance": total_dist,
                "time_taken_sec": pl.time_taken_sec,
            }

            write_log(
                self.game_id,
                f"SCORE {pl.name} ({colour}): Final={final_score:.1f}, "
                f"Time={time_score:.1f}, Moves({pl.move_count})={move_score:.1f}, "
                f"Pins({pins_in_goal})={pin_goal_score:.1f}, Dist={distance_score:.1f}"
            )

    # ----------------------------------
    # Public State
    # ----------------------------------
    def to_public_state(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "status": self.status,
            "players": [
                {
                    "player_id": pl.player_id,
                    "name": pl.name,
                    "colour": pl.colour,
                    "ready": pl.ready,
                    "status": pl.status,
                    "score": self.scores.get(pl.player_id),
                }
                for pl in self.players
            ],
            "pins": {
                colour: [p.axialindex for p in pins]
                for colour, pins in self.pins_by_colour.items()
            },
            "move_count": self.move_count,
            "current_turn_colour": self.current_turn_colour(),
            "turn_order": self.turn_order,
            "last_move": self.last_move,
            "turn_timeout_notice": self.turn_timeout_notice,
        }


# ==========================================================
# Session
# ==========================================================
class Session:
    def __init__(self):
        self.games: Dict[str, Game] = {}
        self.session_games: List[str] = []
        self.lock = threading.RLock()

    # ------------------------------
    def create_game(self) -> str:
        with self.lock:
            g = Game()
            self.games[g.game_id] = g
            self.session_games.append(g.game_id)
            write_log(g.game_id, "GAME CREATED")

            return g.game_id

    # ------------------------------
    def pick_available_game(self) -> Optional[Game]:
        for gid in self.session_games:
            g = self.games[gid]
            if not g.lock_joining and len(g.players) < MAX_PLAYERS:
                if g.status in ("waiting for other player", "AVAILABLE", "READY_TO_START"):
                    return g
        return None

    # ------------------------------
    def join_request(self, player_name: str) -> Dict[str, Any]:
        with self.lock:
            g = self.pick_available_game()
            if not g:
                return {"ok": False, "error": "No available game. Ask admin to Create."}

            colour = g.assign_colour()
            if not colour:
                return {"ok": False, "error": "Game full"}

            pid = str(uuid.uuid4())
            pl = Player(pid, player_name, colour)
            g.players.append(pl)
            g.init_pins(colour)

            if len(g.players) == 1:
                g.status = "waiting for other player"
            else:
                g.status = "READY_TO_START"

            write_log(g.game_id, f"PLAYER JOINED: {player_name} as {colour}")


            return {
                "ok": True,
                "game_id": g.game_id,
                "player_id": pid,
                "colour": colour,
                "status": g.status,
            }

    # ------------------------------
    def mark_start_ready(self, game_id: str, player_id: str) -> Dict[str, Any]:
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            for pl in g.players:
                if pl.player_id == player_id:
                    pl.ready = True
                    write_log(g.game_id, f"PLAYER START: {pl.name} ({pl.colour})")
                    break

            if g.status == "READY_TO_START":
                g.lock_joining = True
                if len(g.players) >= 2 and all(pl.ready for pl in g.players):
                    g.status = "PLAYING"
                    g.total_start_ns = time.perf_counter_ns()
                    g.compute_turn_order()
                    g.turn_started_ns = time.perf_counter_ns()
                    write_log(g.game_id, f"GAME START — turn order {g.turn_order}")

            return {"ok": True, "status": g.status}

    def get_legal_moves(self, game_id: str, player_id: str):
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            pl = next((p for p in g.players if p.player_id == player_id), None)
            if not pl:
                return {"ok": False, "error": "Player not found"}

            pins = g.pins_by_colour[pl.colour]
            legal = {}

            for i, pin in enumerate(pins):
                legal[i] = pin.getPossibleMoves()

            return {"ok": True, "legal_moves": legal}

    # ------------------------------
    def validate_and_apply_move(self, game_id: str, player_id: str, pin_id: int, to_index: int):
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            g.ensure_time_limits()

            if g.status != "PLAYING":
                return {"ok": False, "error": f"Game not in PLAYING: {g.status}"}

            pl = next((p for p in g.players if p.player_id == player_id), None)
            if not pl:
                return {"ok": False, "error": "Player not in game"}

            if g.current_turn_colour() != pl.colour:
                return {"ok": False, "error": f"Not {pl.colour}'s turn. "}

            pins = g.pins_by_colour[pl.colour]
            if not (0 <= pin_id < len(pins)):
                return {"ok": False, "error": "Invalid pin ID"}

            pin = pins[pin_id]
            legal = pin.getPossibleMoves()
            if to_index not in legal:
                return {"ok": False, "error": "Illegal move"}

            # Time tracking
            if g.turn_started_ns:
                dt = (time.perf_counter_ns() - g.turn_started_ns) / 1e9
                pl.time_taken_sec += dt

            # Apply move
            start_ns = time.perf_counter_ns()
            from_idx = pin.axialindex
            moved_ok = pin.placePin(to_index)
            end_ns = time.perf_counter_ns()
            move_ms = (end_ns - start_ns) / 1e6

            if not moved_ok:
                return {"ok": False, "error": "Could not move"}

            pl.move_count += 1
            g.move_count += 1
            g.move_times_ms.append(move_ms)
            if g.turn_timeout_notice is not None:
                timeout_movenum = int(g.turn_timeout_notice.split('move')[1].split('.')[0].replace(' ',''))
                print(g.turn_timeout_notice.split('move')[1], g.turn_timeout_notice.split('move')[1].split('.'))
                if timeout_movenum<g.move_count:
                    g.turn_timeout_notice = None

            g.last_move = {
                "pin_id": pin_id,
                "from": from_idx,
                "to": to_index,
                "by": pl.name,
                "colour": pl.colour,
                "move_ms": move_ms,
            }
            write_log(
                g.game_id,
                f"MOVE {g.move_count}: {pl.name} ({pl.colour}) {from_idx}->{to_index} [{move_ms:.2f}ms]"
            )

            # Check WIN / DRAW
            pl.status = g.check_player_status(pl.colour)

            if pl.status == "WIN":
                g.status = "FINISHED"
                g.compute_scores()

                return {"ok": True, "status": "WIN", "state": g.to_public_state(),
                        "msg": f"{pl.name} Wins"}

            if pl.status == "DRAW":
                # Check if all others draw except one
                live = g.players
                draws = [p for p in live if g.check_player_status(p.colour) == "DRAW"]
                if len(draws) == len(live) - 1:
                    winner = next(p for p in live if p not in draws)
                    g.status = "FINISHED"
                    g.compute_scores()

                    return {"ok": True, "status": "WIN", "state": g.to_public_state(),
                            "msg": f"{winner.name} Wins, others Draw."}

            # Continue game
            g.advance_turn()
            g.compute_scores()

            return {"ok": True, "status": "CONTINUE",
                    "state": g.to_public_state()}

    # ------------------------------
    def game_status_list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {
                    "game_id": g.game_id,
                    "players": [{"name": p.name, "colour": p.colour} for p in g.players],
                    "moves": g.move_count,
                    "status": g.status,
                }
                for gid in self.session_games
                for g in [self.games[gid]]
            ]


SESSION = Session()

# ==========================================================
# RPC HANDLER
# ==========================================================
def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    op = req.get("op")

    if op == "get_legal_moves":
        return SESSION.get_legal_moves(
            req.get("game_id"),
            req.get("player_id")
        )

    if op == "join":
        return SESSION.join_request(req.get("player_name"))

    if op == "start":
        return SESSION.mark_start_ready(req.get("game_id"), req.get("player_id"))

    if op == "get_state":
        game_id = req.get("game_id")
        g = SESSION.games.get(game_id)
        if not g:
            return {"ok": False, "error": "Game not found"}
        g.ensure_time_limits()
        return {"ok": True, "state": g.to_public_state()}

    if op == "move":
        return SESSION.validate_and_apply_move(
            req.get("game_id"),
            req.get("player_id"),
            int(req.get("pin_id")),
            int(req.get("to_index")),
        )

    if op == "status":
        return {"ok": True, "games": SESSION.game_status_list()}

    return {"ok": False, "error": f"Unknown op {op}"}


# ==========================================================
# SERVER LOOP
# ==========================================================
def server_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 50555))
    s.listen(50)
    print("[Server] Listening on 0.0.0.0:50555")

    while True:
        conn, addr = s.accept()
        try:
            conn.settimeout(10.0)
            data = conn.recv(65535)
            if not data:
                conn.close()
                continue
            try:
                req = json.loads(data.decode("utf-8"))
            except:
                resp = {"ok": False, "error": "bad-json"}
                conn.sendall(json.dumps(resp).encode("utf-8"))
                conn.close()
                continue

            resp = handle_request(req)
            js = safe_json(resp)
            conn.sendall(js.encode("utf-8"))

        finally:
            conn.close()




# ==========================================================
# CLI LOOP
# ==========================================================
def cli_loop():
    print("Game Manager")
    print("Commands: Create, Status, Quit\n")

    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == "create":
            gid = SESSION.create_game()
            print("Game created:", gid)
        elif cmd == "status":
            for g in SESSION.game_status_list():
                print(g)
        elif cmd == "quit":
            os._exit(0)
        else:
            print("Invalid command")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    threading.Thread(target=server_loop, daemon=True).start()
    cli_loop()
