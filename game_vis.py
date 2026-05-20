# ==========================================================
# game_gui.py — SERVER + LIVE VISUALIZATION
#
# Runs the exact same server as game.py, but adds a Tkinter
# window that shows the board updating in real time as moves
# come in.
#
# game.py is left completely untouched — this imports from it,
# so the tournament server logic is identical. Use game.py for
# the real tournament; use this for watching games locally.
#
# USAGE
#   python game_gui.py
#   Then in the terminal: type "Create" to make a game.
#   Connect players (player.py) as usual.
#   The window auto-displays the most recent game in progress.
#
# THREADING MODEL
#   • main thread        → Tkinter mainloop (required by Tk)
#   • server thread      → socket server (daemon)
#   • CLI thread         → Create/Status/Quit prompt (daemon)
#   • GUI redraw         → root.after() polling, main thread only
#     (never touch Tk from another thread — we only read shared
#      game state in the poll callback, which runs on main thread)
# ==========================================================

import os
import time
import threading
import tkinter as tk

# Import the server machinery from game.py unchanged.
from game import SESSION, server_loop

from checkers_board import HexBoard


# Color the board cells / pins are drawn with (matches checkers_gui.py)
PIN_FILL = {
    'red':        'red',
    'blue':       'blue',
    'lawn green': 'green',
    'gray0':      'gray20',
    'yellow':     'gold',
    'purple':     'purple',
}
CELL_TINT = {
    'red':        'rosybrown1',
    'lawn green': 'palegreen1',
    'blue':       'lightblue1',
    'yellow':     'lightgoldenrod1',
    'purple':     'plum1',
    'gray0':      'gray60',
}


class LiveBoardGUI:
    """
    Tkinter visualization that polls SESSION for the active game and
    redraws the board each tick. Reads game state only; never mutates it.
    """

    POLL_MS = 200   # redraw interval

    def __init__(self):
        # We need a board geometry to lay out cells. Every Game builds its
        # own HexBoard, but they all share the same geometry, so we make a
        # local one purely for pixel coordinates / cell layout.
        self.layout_board = HexBoard()

        self.root = tk.Tk()
        self.root.title("Chinese Checkers — Live")

        xs = [x for x, y in self.layout_board.cartesian]
        ys = [y for x, y in self.layout_board.cartesian]
        pad = 60
        width  = int(max(xs) - min(xs) + 2 * pad)
        height = int(max(ys) - min(ys) + 2 * pad)

        self.offset_x = -min(xs) + pad
        self.offset_y = -min(ys) + pad

        # Status bar on top
        self.status_var = tk.StringVar(value="Waiting for a game...")
        self.status = tk.Label(self.root, textvariable=self.status_var,
                               font=("TkDefaultFont", 11), anchor="w")
        self.status.pack(side="top", fill="x", padx=8, pady=4)

        self.canvas = tk.Canvas(self.root, width=width, height=height + 40,
                                bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Which game we're displaying. We pick the most recent game that's
        # PLAYING (or the most recent one overall if none playing yet).
        self.current_game_id = None

        # Track the last move so we can highlight it
        self.last_move_count = -1

        # Kick off the polling redraw loop on the main thread.
        self.root.after(self.POLL_MS, self._tick)

    # ------------------------------------------------------------------
    def _to_canvas(self, x, y):
        return (x + self.offset_x, y + self.offset_y)

    # ------------------------------------------------------------------
    def _select_game(self):
        """
        Pick which game to display. Preference order:
          1. The game we're already showing (sticky), if it still exists.
          2. The most recently created game in PLAYING status.
          3. The most recently created game of any status.
        """
        with SESSION.lock:
            if not SESSION.session_games:
                return None

            # Sticky: keep showing the current game if it still exists
            if self.current_game_id in SESSION.games:
                return SESSION.games[self.current_game_id]

            # Prefer a PLAYING game, newest first
            for gid in reversed(SESSION.session_games):
                g = SESSION.games.get(gid)
                if g and g.status == "PLAYING":
                    return g

            # Fallback: newest game of any status
            gid = SESSION.session_games[-1]
            return SESSION.games.get(gid)

    # ------------------------------------------------------------------
    def _tick(self):
        """Polling redraw. Runs on the main (Tk) thread via root.after()."""
        try:
            self._redraw()
        except Exception as e:
            # Never let a redraw error kill the loop
            self.status_var.set(f"(redraw error: {e})")
        finally:
            self.root.after(self.POLL_MS, self._tick)

    # ------------------------------------------------------------------
    def _redraw(self):
        g = self._select_game()
        if g is None:
            self.status_var.set("Waiting for a game...  (type 'Create' in terminal)")
            return

        self.current_game_id = g.game_id

        # Snapshot the state we need under the lock, then release it before
        # drawing (drawing doesn't touch shared state).
        with SESSION.lock:
            status        = g.status
            move_count    = g.move_count
            turn_colour   = g.current_turn_colour()
            # pins_by_colour maps colour -> [Pin]; capture axial indices
            pins_snapshot = {
                colour: [p.axialindex for p in pins]
                for colour, pins in g.pins_by_colour.items()
            }
            last_move = dict(g.last_move) if g.last_move else None
            # scores for a compact readout
            score_snapshot = []
            for pl in g.players:
                sc = g.scores.get(pl.player_id)
                pins_home = sc["pins_in_goal"] if sc else 0
                score_snapshot.append((pl.colour, pins_home))

        # ---- status bar text ----
        if status == "PLAYING":
            turn_txt = f"turn: {turn_colour}"
        else:
            turn_txt = status
        score_txt = "  ".join(f"{c[:3]}={n}" for c, n in score_snapshot)
        self.status_var.set(
            f"game {g.game_id[:8]}…   moves={move_count}   {turn_txt}   [{score_txt}]"
        )

        # ---- draw board ----
        self.canvas.delete("all")   # IMPORTANT: clear before redraw, else ghosting

        r = self.layout_board.hole_radius
        for cell in self.layout_board.cells:
            cx, cy = self._to_canvas(cell.x, cell.y)
            if cell.postype == 'board':
                fill = "lightgray"
            else:
                fill = CELL_TINT.get(cell.postype, "lightgray")
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    fill=fill, outline="black")

        # ---- draw pins ----
        # Highlight the destination of the last move with a thicker outline.
        last_to = last_move["to"] if last_move else None
        pr = int(r * 0.7)
        for colour, axial_list in pins_snapshot.items():
            fill = PIN_FILL.get(colour, "black")
            for axial_idx in axial_list:
                if axial_idx < 0 or axial_idx >= len(self.layout_board.cells):
                    continue
                cell = self.layout_board.cells[axial_idx]
                cx, cy = self._to_canvas(cell.x, cell.y)
                outline = "white" if axial_idx == last_to else "black"
                w = 3 if axial_idx == last_to else 1
                self.canvas.create_oval(cx - pr, cy - pr, cx + pr, cy + pr,
                                        fill=fill, outline=outline, width=w)

    # ------------------------------------------------------------------
    def run(self):
        self.root.mainloop()


# ==========================================================
# CLI LOOP (same commands as game.py, runs on a daemon thread)
# ==========================================================
def cli_loop():
    print("Game Manager (GUI mode)")
    print("Commands: Create, Status, Quit\n")
    while True:
        try:
            cmd = input("Enter command: ").strip().lower()
        except EOFError:
            return
        if cmd == "create":
            gid = SESSION.create_game()
            print("Game created:", gid)
        elif cmd == "status":
            for gi in SESSION.game_status_list():
                print(gi)
        elif cmd == "quit":
            os._exit(0)
        else:
            print("Invalid command")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    # Server on a background daemon thread (same as game.py).
    threading.Thread(target=server_loop, daemon=True).start()
    # CLI on its own daemon thread, because the main thread must run Tk.
    threading.Thread(target=cli_loop, daemon=True).start()
    # GUI owns the main thread.
    gui = LiveBoardGUI()
    gui.run()