"""
env.py — Chinese Checkers RL environment.

═══════════════════════════════════════════════════════════════════════════
DESIGN
═══════════════════════════════════════════════════════════════════════════

Wraps the professor's HexBoard + Pin classes into a clean RL interface.

KEY DESIGN DECISIONS
────────────────────
1. Near-goal curriculum
   reset() takes `pins_advanced ∈ [0, NUM_PINS_PER_PLAYER]`. That many
   of the player's pins start in cells 1-3 hex-distance from the goal
   zone. Goal zone always empty at episode start.

2. Distance-weighted reward
   Each pin's per-step progress is weighted by its STARTING distance.
   Far-away pins (home zone, ~10 cells) give 5-8x more reward per cell
   of forward progress than near pins (1-2 cells). Asymmetric forward/
   backward weights (1.5x backward) prevent oscillation gaming.

3. State representation (186 dims)
   Original 176 dims plus 10 dims for `pin_start_dists`.

4. BFS true distances throughout

5. Multi-player support (2/3/4/6).

6. Hard pin lock + rule-based rearrangement
   Pins inside their destination zone are HARD-LOCKED from the agent's
   action mask — the agent never sees them as movable. A separate rule,
   find_rearrangement_move(), automatically moves a goal pin DEEPER when
   possible. The rule consumes the player's turn (one move per turn,
   same as a normal move). This handles the "fill destination from the
   back" strategic pattern deterministically, with no reward shaping
   that could be gamed. (Earlier attempts at depth bonuses and
   intra-destination rewards caused training collapse.)
"""

# env.py

from __future__ import annotations

import builtins
from collections import deque
from typing import List, Dict, Optional

import numpy as np

from checkers_board import HexBoard
from checkers_pins import Pin


# ── constants ────────────────────────────────────────────────────────────────
NUM_PINS_PER_PLAYER = 10
MAX_STEPS           = 600
MAX_HEX_DIST        = 16

ALL_COLOURS = ['red', 'blue', 'lawn green', 'gray0', 'yellow', 'purple']


# ── silence helper ───────────────────────────────────────────────────────────
class _SilencePrint:
    """Suppresses print() spam from HexBoard and placePin during construction."""
    def __enter__(self):
        self._orig = builtins.print
        builtins.print = lambda *a, **k: None
    def __exit__(self, *_):
        builtins.print = self._orig


# ── action encoding ──────────────────────────────────────────────────────────
def encode_action(pin_id: int, dest_idx: int, num_cells: int) -> int:
    """Flatten (pin_id, dest_cell_index) → single integer in [0, 10*num_cells)."""
    return pin_id * num_cells + dest_idx


def decode_action(action_id: int, num_cells: int) -> tuple:
    """Recover (pin_id, dest_cell_index) from a flat action integer."""
    return divmod(action_id, num_cells)


# ── BFS adjacency / distances (cached at class level) ────────────────────────
def _build_adjacency(board: HexBoard) -> Dict[int, List[int]]:
    """Build neighbor adjacency for the hex board using axial directions."""
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]
    adj = {i: [] for i in range(len(board.cells))}
    for i, cell in enumerate(board.cells):
        for dq, dr in directions:
            n_axial = (cell.q + dq, cell.r + dr)
            if n_axial in board.index_of:
                adj[i].append(board.index_of[n_axial])
    return adj


def _bfs_all_pairs(adj: Dict[int, List[int]], num_cells: int) -> Dict[int, Dict[int, int]]:
    """All-pairs shortest path via BFS."""
    all_dist = {}
    for start in range(num_cells):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for nbr in adj[cur]:
                if nbr not in dist:
                    dist[nbr] = dist[cur] + 1
                    queue.append(nbr)
        all_dist[start] = dist
    return all_dist


# ── environment ───────────────────────────────────────────────────────────────
class ChineseCheckersEnv:
    """
    Multi-player Chinese Checkers RL environment.

    Public API
    ──────────
    reset(pins_advanced=0)             → state ndarray (186,)
    step(action_id)                    → (next_state, reward, done)
    build_action_mask(color=None)      → ndarray (action_dim,) binary
    get_valid_moves(color=None)        → list[(Pin, dest_idx)]
    find_rearrangement_move(color=None) → (pin_id, dest_idx) or None  (NEW)
    current_color                      → property
    action_dim                         → property
    num_cells                          → int (set after first reset)
    """

    _BFS_CACHE: Optional[Dict[int, Dict[int, int]]] = None
    _ADJACENCY: Optional[Dict[int, List[int]]] = None

    def __init__(self, n_players: int = 2, color_pair: Optional[list] = None):
        """
        Args:
            n_players: 2, 3, 4, or 6.
            color_pair: optional explicit list of colors to assign. If given,
                len(color_pair) must equal n_players. This is what enables
                per-axis specialist training — e.g. color_pair=['lawn green',
                'gray0'] forces 2-player games on the lawn_green↔gray0 axis
                instead of the default red↔blue axis.
        """
        assert n_players in (2, 3, 4, 6), \
            f"n_players must be 2, 3, 4, or 6 (got {n_players})"
        if color_pair is not None:
            assert len(color_pair) == n_players, \
                f"color_pair len {len(color_pair)} != n_players {n_players}"
            for c in color_pair:
                assert c in ALL_COLOURS, f"unknown color {c!r}"
        self.n_players     = n_players
        self.color_pair    = color_pair      # None → default red↔blue↔...

        self.board         = None
        self.pins          = []
        self.assigned      = []
        self.current_idx   = 0
        self.done          = False
        self.num_cells     = None
        self._step_count   = 0
        self._target_cache = {}
        self._goal_depths = {}         # color → {cell_idx: depth} normalized 0..N
        self._pin_start_dists = {}
        self.valid_move_cache = {}

    # ── reset ────────────────────────────────────────────────────────────────
    def reset(self, pins_advanced: int = 0):
        """Set up a new game with optional curriculum head-start."""
        
        # print("AXIS:", self.assigned)

        assert 0 <= pins_advanced <= NUM_PINS_PER_PLAYER, \
            f"pins_advanced must be in [0, {NUM_PINS_PER_PLAYER}]"

        with _SilencePrint():
            self.board = HexBoard(R=4, hole_radius=16, spacing=34)

        # If color_pair was set in __init__, use it. Otherwise default to
        # ALL_COLOURS[:n_players] (preserves original behavior).
        if self.color_pair is not None:
            self.assigned = list(self.color_pair)
        else:
            self.assigned = list(ALL_COLOURS[:self.n_players])
        self.current_idx = 0
        self.done        = False
        self._step_count = 0
        self.pins        = []

        if ChineseCheckersEnv._BFS_CACHE is None:
            ChineseCheckersEnv._ADJACENCY = _build_adjacency(self.board)
            ChineseCheckersEnv._BFS_CACHE = _bfs_all_pairs(
                ChineseCheckersEnv._ADJACENCY, len(self.board.cells)
            )

        used_cells = set()
        for color in self.assigned:
            home_indices   = self.board.axial_of_colour(color)
            target_indices = self.board.axial_of_colour(
                self.board.colour_opposites[color]
            )
            target_set = set(target_indices)

            candidates = []
            for cell_idx in range(len(self.board.cells)):
                if cell_idx in used_cells:
                    continue
                if cell_idx in target_set:
                    continue
                if cell_idx in set(home_indices):
                    continue
                d = min(self._bfs_dist(cell_idx, t) for t in target_indices)
                if d <= 3:
                    candidates.append((d, cell_idx))

            candidates.sort(key=lambda x: x[0])
            advanced_cells = [c[1] for c in candidates[:pins_advanced]]

            placed_count = 0
            for axial_idx in advanced_cells:
                self.pins.append(Pin(self.board, axial_idx,
                                     id=placed_count, color=color))
                used_cells.add(axial_idx)
                placed_count += 1

            for h_idx in home_indices:
                if placed_count >= NUM_PINS_PER_PLAYER:
                    break
                if h_idx not in used_cells:
                    self.pins.append(Pin(self.board, h_idx,
                                         id=placed_count, color=color))
                    used_cells.add(h_idx)
                    placed_count += 1

            if placed_count < NUM_PINS_PER_PLAYER:
                fallback_candidates = []
                for cell_idx in range(len(self.board.cells)):
                    if cell_idx in used_cells:
                        continue
                    if cell_idx in target_set:
                        continue
                    d = min(self._bfs_dist(cell_idx, t) for t in target_indices)
                    fallback_candidates.append((-d, cell_idx))
                fallback_candidates.sort(key=lambda x: x[0])

                for _, cell_idx in fallback_candidates:
                    if placed_count >= NUM_PINS_PER_PLAYER:
                        break
                    self.pins.append(Pin(self.board, cell_idx,
                                         id=placed_count, color=color))
                    used_cells.add(cell_idx)
                    placed_count += 1

            assert placed_count == NUM_PINS_PER_PLAYER, (
                f"Failed to place {NUM_PINS_PER_PLAYER} pins for {color} "
                f"(only placed {placed_count})."
            )

        self.num_cells = len(self.board.cells)

        # Cache target zones
        self._target_cache = {}
        for color in self.assigned:
            opp = self.board.colour_opposites[color]
            self._target_cache[color] = frozenset(
                self.board.axial_of_colour(opp)
            )

        # Cache goal-cell depths (used by the rearrangement rule).
        # depth = BFS distance from goal cell back to source's home zone.
        # Normalized so shallowest cell = 0, deepest = N.
        self._goal_depths = {}
        for color in self.assigned:
            target = self._target_cache[color]
            source_home = list(self.board.axial_of_colour(color))
            raw_depths = {
                t: min(self._bfs_dist(t, h) for h in source_home)
                for t in target
            }
            min_d = min(raw_depths.values()) if raw_depths else 0
            self._goal_depths[color] = {
                t: d - min_d for t, d in raw_depths.items()
            }

        # Per-pin starting distances (for distance-weighted reward + state)
        self._pin_start_dists = {}
        for color in self.assigned:
            target = self._target_cache[color]
            color_pins = sorted([p for p in self.pins if p.color == color],
                                key=lambda p: p.id)
            self._pin_start_dists[color] = [
                min(self._bfs_dist(p.axialindex, t) for t in target)
                for p in color_pins
            ]

        return self._get_state()

    # ── step ─────────────────────────────────────────────────────────────────
    def step(self, action_id: int):
        """Apply an action. Ends episode if any player wins or step limit reached."""
        assert not self.done, "Episode is over. Call reset() first."

        color = self.current_color
        pin_id, dest_idx = decode_action(action_id, self.num_cells)

        pin = self._get_pin(color, pin_id)
        if pin is None:
            self._advance_turn()
            return self._get_state(), 0.0, self.done

        legal = pin.getPossibleMoves()
        
        # --------------------------------------------------------
        # No legal moves
        # --------------------------------------------------------

        if len(legal) == 0:

            self._advance_turn()

            return (
                self._get_state(),
                -1.0,
                self.done,
            )
        

        if dest_idx not in legal:
            self._advance_turn()
            return self._get_state(), 0.0, self.done

        prev_info = self._compute_state_info(color)

        with _SilencePrint():
            success = pin.placePin(dest_idx)
            self.valid_move_cache.clear()
        if not success:
            self._advance_turn()
            return self._get_state(), 0.0, self.done

        self._step_count += 1
        next_info = self._compute_state_info(color)

        # Check if current player won
        won = self._check_win(color)
        if won:
            self.done = True
            reward = self._compute_reward(color, prev_info, next_info, won=True)
            return self._get_state(), reward, self.done

        # Check if any opponent won (agent loses)
        for opp_color in self.assigned:
            if opp_color != color and self._check_win(opp_color):
                self.done = True
                # Assign strong negative reward for losing
                reward = -200.0
                return self._get_state(), reward, self.done

        # Normal reward
        reward = self._compute_reward(color, prev_info, next_info, won=False)

        if self._step_count >= MAX_STEPS:
            self.done = True

        self._advance_turn()
        return self._get_state(), reward, self.done

    # ── action mask ──────────────────────────────────────────────────────────
    _mask_warn_counts = {}

        
    def build_action_mask(self,
                        color: Optional[str] = None) -> np.ndarray:
        """
        Binary float32 mask of shape (action_dim,).

        1 = valid
        0 = invalid
        """

        if color is None:
            color = self.current_color

        target = self._target_cache[color]

        mask = np.zeros(
            self.action_dim,
            dtype=np.float32,
        )

        for pin, dest in self.get_valid_moves(color):

            # ----------------------------------------------------
            # Hard lock goal pins
            # ----------------------------------------------------

            if pin.axialindex in target:
                continue

            aid = encode_action(
                pin.id,
                dest,
                self.num_cells,
            )

            if 0 <= aid < self.action_dim:
                mask[aid] = 1.0

        return mask

    def get_valid_moves(self,
                        color: Optional[str] = None) -> List:

        if color is None:
            color = self.current_color

        cache_key = (
            color,
            tuple(
                sorted(
                    (p.color, p.axialindex)
                    for p in self.pins
                )
            )
        )

        if cache_key in self.valid_move_cache:
            return self.valid_move_cache[cache_key]

        moves = [
            (p, d)
            for p in self.pins
            if p.color == color
            for d in p.getPossibleMoves()
        ]

        self.valid_move_cache[cache_key] = moves

        return moves


    # ── rule-based rearrangement ────────────────────────────────────────────
    def find_rearrangement_move(self, color: Optional[str] = None):
        """
        Rule-based rearrangement move within destination.

        Looks at this color's pins currently inside their destination zone.
        If any can move to a STRICTLY DEEPER goal cell, returns the
        (pin_id, dest_idx) with the greatest depth gain.
        Returns None if no goal pin can go deeper.

        When this returns a move, the train/inference loop should execute
        it instead of asking the agent to choose. The move consumes the
        player's turn (same as a normal move) but it's deterministic, not
        learned. Handles the "fill destination from the back" strategic
        pattern automatically.

        Note: re-evaluated every turn. If a goal pin was previously locked
        because of no available deeper cell (e.g., blocked by an opponent
        pin), the rule will reconsider it next turn after the situation
        changes.

        Returns:
            (pin_id, dest_idx) tuple, or None if no rearrangement available.
        """
        if color is None:
            color = self.current_color
        target    = self._target_cache[color]
        depth_map = self._goal_depths.get(color, {})

        best_gain = 0
        best_move = None
        for pin in self.pins:
            if pin.color != color:
                continue
            if pin.axialindex not in target:
                continue   # only consider pins currently in destination
            cur_depth = depth_map.get(pin.axialindex, 0)
            for dest in pin.getPossibleMoves():
                if dest not in target:
                    continue   # destination must be inside goal too
                new_depth = depth_map.get(dest, 0)
                gain = new_depth - cur_depth
                if gain > best_gain:
                    best_gain = gain
                    best_move = (pin.id, dest)

        return best_move

    # ── properties ───────────────────────────────────────────────────────────
    @property
    def current_color(self) -> str:
        return self.assigned[self.current_idx]

    @property
    def action_dim(self) -> int:
        return NUM_PINS_PER_PLAYER * self.num_cells

    # ── distance helpers ────────────────────────────────────────────────────
    def _bfs_dist(self, idx_a: int, idx_b: int) -> int:
        """True graph shortest-path distance between two cells."""
        return ChineseCheckersEnv._BFS_CACHE[idx_a].get(idx_b, 999)

    def _hex_dist(self, idx_a: int, idx_b: int) -> int:
        """Geometric hex distance (faster, fallback only)."""
        a = self.board.cells[idx_a]
        b = self.board.cells[idx_b]
        return (abs(a.q - b.q) + abs(a.r - b.r)
                + abs((a.q + a.r) - (b.q + b.r))) // 2

    # ── state info for reward ────────────────────────────────────────────────
    def _compute_state_info(self, color: str) -> dict:
        """
        Per-pin distance summary for the reward function.
        Distances list is sorted by pin.id for consistent indexing.

        IMPORTANT: distance is to the nearest UNFILLED target cell.
        Pins approaching cells that are already filled by their own color
        wouldn't be making real progress (they can't enter those cells).
        Using empty-target distance makes the reward gradient point at
        cells the agent can actually use. Falls back to nearest target
        if all target cells are filled.
        """
        target  = self._target_cache[color]
        my_pins = sorted([p for p in self.pins if p.color == color],
                         key=lambda p: p.id)

        # Find the empty target cells. If a pin already in goal is sitting
        # on a target cell, that cell is "occupied" from the perspective of
        # the OTHER pins of the same color trying to enter goal.
        my_pin_cells = {p.axialindex for p in my_pins}
        empty_targets = [t for t in target if t not in my_pin_cells]

        # If all targets are filled (game basically over), fall back to
        # distance-to-nearest-target so reward calc doesn't blow up.
        if empty_targets:
            target_set_for_distance = empty_targets
        else:
            target_set_for_distance = list(target)

        # Distances: per-pin minimum to nearest EMPTY (or any if none) target.
        # If the pin is itself in goal, its distance is 0 (we don't want
        # this pin counted as needing to travel further).
        distances = []
        for p in my_pins:
            if p.axialindex in target:
                distances.append(0)
            else:
                distances.append(
                    min(self._bfs_dist(p.axialindex, t)
                        for t in target_set_for_distance)
                )

        # Identify the LAGGARD pin — furthest from goal among non-goal pins.
        laggard_id = None
        laggard_dist = -1
        for i, d in enumerate(distances):
            if d > 0 and d > laggard_dist:
                laggard_dist = d
                laggard_id = i

        return {
            "distances":       distances,
            "pieces_in_goal":  sum(1.0 for d in distances if d == 0),
            "pieces_within_1": sum(1.0 for d in distances if d <= 1),
            "pieces_within_2": sum(1.0 for d in distances if d <= 2),
            "pieces_within_3": sum(1.0 for d in distances if d <= 3),
            "total_distance":  float(sum(distances)),
            "laggard_id":      laggard_id,
            "laggard_dist":    laggard_dist,
        }

    # ── reward ──────────────────────────────────────────────────────────────
    def _compute_reward(self, color: str,
                        prev_info: dict, next_info: dict,
                        won: bool = False) -> float:
        """
        Distance-weighted reward — far pins are worth more per cell.

        Components
        ──────────
        1. Per-pin weighted progress: weight × delta_distance per pin.
           Asymmetric: backward weight 1.5x forward.

        2. Per-pin completion bonus: 2.0 × journey when a pin enters goal.
           (Depth bonuses removed — caused gaming via goal-pin shuffling.
           Goal-zone rearrangement is now handled by the rule, not reward.)

        3. Laggard pin bonus: +5.0 × (laggard_dist_before - laggard_dist_after)

        4. Band bonuses: within_1 +3, within_2 +1.5, within_3 +0.5.

        5. Time cost: -0.05 per step.

        6. Win bonus: +200.
        """
        if won:
            return 200.0

        start_dists = self._pin_start_dists[color]
        prev_d = prev_info["distances"]
        next_d = next_info["distances"]

        # 1. Distance-weighted progress
        progress_reward = 0.0
        for i in range(NUM_PINS_PER_PLAYER):
            delta = prev_d[i] - next_d[i]
            weight = max(start_dists[i], 1)
            if delta > 0:
                progress_reward += 0.05 * weight * delta
            else:
                progress_reward += 0.05 * weight * delta * 1.5

        # 2. Completion bonus
        completion_reward = 0.0
        leave_penalty = 0.0
        for i in range(NUM_PINS_PER_PLAYER):
            if prev_d[i] > 0 and next_d[i] == 0:
                journey = max(start_dists[i], 1)
                completion_reward += 2.0 * journey
            elif prev_d[i] == 0 and next_d[i] > 0:
                # Pin LEFT goal — blocked by mask, defense in depth
                leave_penalty -= 15.0

        # 3. Laggard bonus
        laggard_reward = 0.0
        laggard_id = prev_info.get("laggard_id")
        if laggard_id is not None:
            laggard_delta = prev_d[laggard_id] - next_d[laggard_id]
            laggard_reward = 1.0 * laggard_delta

        # 4. Band bonuses
        within_1_delta = next_info["pieces_within_1"]  - prev_info["pieces_within_1"]
        within_2_delta = next_info["pieces_within_2"]  - prev_info["pieces_within_2"]
        within_3_delta = next_info["pieces_within_3"]  - prev_info["pieces_within_3"]
        band_reward = (
            3.0 * within_1_delta
            + 1.5 * within_2_delta
            + 0.5 * within_3_delta
        )

        # 5. Time cost
        time_cost = -0.05

        return (progress_reward + completion_reward + leave_penalty
                + laggard_reward + band_reward + time_cost)

    # ── state vector (186-dim) ───────────────────────────────────────────────
    def _get_state(self) -> np.ndarray:
        """
        State from CURRENT player's perspective. Layout (186 dims total):
          [0   : 121]  occupancy        +1 mine, -1 enemy, 0 empty
          [121 : 131]  my_dists         pin-to-nearest-target / 16
          [131 : 141]  my_dists_empty   pin-to-nearest-EMPTY-target / 16
          [141 : 151]  am_i_home        1.0 if pin in goal zone
          [151 : 161]  my_path_blockers nearest enemy on my path / 16
          [161 : 171]  target_occupancy which goal slots I've filled
          [171 : 181]  pin_start_dists  each pin's STARTING distance / 16
          [181 : 186]  scalars          5 global summary statistics
        """
        try:
            color  = self.current_color
            target = self._target_cache[color]
            target_list = sorted(target)

            my_pins  = sorted([p for p in self.pins if p.color == color],
                              key=lambda p: p.id)
            opp_pins = [p for p in self.pins if p.color != color]

            # 1. Occupancy
            occupancy = np.zeros(self.num_cells, dtype=np.float32)
            for p in my_pins:
                occupancy[p.axialindex] = 1.0
            for p in opp_pins:
                occupancy[p.axialindex] = -1.0

            # 2. My pin distances to nearest target
            my_dists = np.array([
                min(self._bfs_dist(p.axialindex, t) for t in target) / MAX_HEX_DIST
                for p in my_pins
            ], dtype=np.float32)

            # 3. My pin distances to nearest EMPTY target
            empty_targets = [t for t in target if not self.board.cells[t].occupied]
            if empty_targets:
                my_dists_empty = np.array([
                    min(self._bfs_dist(p.axialindex, t) for t in empty_targets) / MAX_HEX_DIST
                    for p in my_pins
                ], dtype=np.float32)
            else:
                my_dists_empty = np.zeros(NUM_PINS_PER_PLAYER, dtype=np.float32)

            # 4. Am I home (per pin)
            am_i_home = np.array([
                1.0 if p.axialindex in target else 0.0
                for p in my_pins
            ], dtype=np.float32)

            # 5. Path blockers
            my_path_blockers = np.ones(NUM_PINS_PER_PLAYER, dtype=np.float32)
            for i, p in enumerate(my_pins):
                best = MAX_HEX_DIST
                d_pin_target_min = min(
                    self._bfs_dist(p.axialindex, t) for t in target
                )
                for e in opp_pins:
                    d_pin_enemy = self._bfs_dist(p.axialindex, e.axialindex)
                    d_enemy_target_min = min(
                        self._bfs_dist(e.axialindex, t) for t in target
                    )
                    if d_pin_enemy + d_enemy_target_min <= d_pin_target_min + 2:
                        best = min(best, d_pin_enemy)
                my_path_blockers[i] = best / MAX_HEX_DIST

            # 6. Target occupancy
            my_pin_indices = {p.axialindex for p in my_pins}
            target_occupancy = np.array([
                1.0 if t in my_pin_indices else 0.0
                for t in target_list
            ], dtype=np.float32)

            # 7. Per-pin starting distances
            start_dists_normalized = np.array([
                d / MAX_HEX_DIST
                for d in self._pin_start_dists[color]
            ], dtype=np.float32)

            # 8. Global scalars
            my_pins_home_frac = float(am_i_home.sum() / NUM_PINS_PER_PLAYER)

            opp_pins_home_fracs = []
            opp_mean_dists      = []
            active_opponents    = 0
            for opp_color in self.assigned:
                if opp_color == color:
                    continue
                opp_target = self._target_cache[opp_color]
                opp_color_pins = [p for p in self.pins if p.color == opp_color]
                if not opp_color_pins:
                    continue
                home = sum(1 for p in opp_color_pins if p.axialindex in opp_target)
                opp_pins_home_fracs.append(home / NUM_PINS_PER_PLAYER)
                mean_d = sum(
                    min(self._bfs_dist(p.axialindex, t) for t in opp_target)
                    for p in opp_color_pins
                ) / (len(opp_color_pins) * MAX_HEX_DIST)
                opp_mean_dists.append(mean_d)
                active_opponents += 1

            max_opp_home_frac = max(opp_pins_home_fracs) if opp_pins_home_fracs else 0.0
            min_opp_mean_dist = min(opp_mean_dists)      if opp_mean_dists      else 1.0
            active_opp_frac   = active_opponents / max(1, len(self.assigned) - 1)

            scalars = np.array([
                my_pins_home_frac,
                max_opp_home_frac,
                float(np.mean(my_dists)),
                min_opp_mean_dist,
                active_opp_frac,
            ], dtype=np.float32)

            state = np.concatenate([
                occupancy,
                my_dists,
                my_dists_empty,
                am_i_home,
                my_path_blockers,
                target_occupancy,
                start_dists_normalized,
                scalars,
            ])
            if np.any(np.isnan(state)) or np.any(np.isinf(state)):
                print("[WARN] _get_state: NaN or Inf detected in state vector, returning zeros.")
                return np.zeros(186, dtype=np.float32)
            return state
        except Exception as e:
            print(f"[ERROR] _get_state failed: {e}; returning zeros.")
            return np.zeros(186, dtype=np.float32)

    # ── win check ────────────────────────────────────────────────────────────
    def _check_win(self, color: str) -> bool:
        target = self._target_cache[color]
        return all(p.axialindex in target
                   for p in self.pins if p.color == color)

    # ── helpers ──────────────────────────────────────────────────────────────
    def _get_pin(self, color: str, pin_id: int):
        for p in self.pins:
            if p.color == color and p.id == pin_id:
                return p
        return None

    def _advance_turn(self):
        self.current_idx = (self.current_idx + 1) % len(self.assigned)

    def print_board(self):
        self.board.print_ascii(pins=self.pins, empty='·')

    def count_pins_home_total(self) -> int:
        total = 0
        for c in self.assigned:
            target = self._target_cache[c]
            total += sum(1 for p in self.pins
                         if p.color == c and p.axialindex in target)
        return total