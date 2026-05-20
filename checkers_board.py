import math


class BoardPosition:
    def __init__(self, q, r, spacing, postype='board'):
        self.q = q
        self.r = r
        self.x = spacing * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        self.y = spacing * (3/2 * r)
        self.postype = postype  # default type
        self.occupied = False


class HexBoard:
    """
    A true hexagonal board on a hex lattice (axial coordinates).
    Radius R=4 => 61 cells (1 + 3*R*(R+1)).
    """
    def __init__(self, R=4, hole_radius=18, spacing=34):
        self.R = R
        self.hole_radius = hole_radius   # circle radius in pixels for each hole
        self.spacing = spacing           # distance between neighboring hex centers
        self.colour_opposites ={'red':'blue', 'lawn green':'gray0', 'blue':'red', 'yellow':'purple', 'purple':'yellow', 'gray0':'lawn green'}


        # Axial coordinates (q, r), with s = -q - r
        self.cells = []                  # list of (q, r)
        self.index_of = {}               # map (q, r) -> index
        self.cartesian = []              # list of (x, y) pixel coords (for Tk)
        self._rows = []                  # rows grouped by r for ASCII

        self._generate_hexagon()
        self._project_to_pixels()
        self._build_rows_for_ascii()

    def _generate_hexagon(self):
        """Generate a regular hexagon of radius R in axial coordinates.""" #use a 12x12 square board instead
        R = self.R
        cells = []
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) <= R:
                    newcell = BoardPosition(q, r, self.spacing)
                    cells.append(newcell)
                    #cells.append((q, r, 'p'))
        base_blue =[(1,-5),(2,-5),(3,-5),(4,-5), (2,-6),(3,-6),(4,-6), (3,-7),(4,-7), (4,-8)]
        base_red =[(-1,5),(-2,5),(-3,5), (-4,5),(-2,6),(-3,6),(-4,6),(-3,7),(-4,7),(-4,8)]
        base_yellow =[(-1,-4),(-2,-3),(-3,-2),(-4,-1), (-2,-4),(-3,-3),(-4,-2), (-3,-4), (-4,-3), (-4,-4) ]
        base_green =[(5,-4),(5,-3),(5,-2),(5,-1), (6,-4),(6,-3),(6,-2), (7,-4),(7,-3), (8,-4)]
        base_purple =[(1,4),(2,3),(3,2),(4,1), (2,4),(3,3),(4,2), (3,4),(4,3), (4,4)]
        base_gray0 =[(-5,1),(-5,2),(-5,3),(-5,4), (-6,2),(-6,3),(-6,4), (-7,3),(-7,4), (-8,4)]
        for (q,r) in base_blue:
            newcell = BoardPosition(q, r, self.spacing, postype='blue')
            cells.append(newcell)
        for (q,r) in base_red:
            newcell = BoardPosition(q, r, self.spacing, postype='red')
            cells.append(newcell)
        for (q,r) in base_yellow:
            newcell = BoardPosition(q, r, self.spacing, postype='yellow')
            cells.append(newcell)
        for (q,r) in base_green:
            newcell = BoardPosition(q, r, self.spacing, postype='lawn green')
            cells.append(newcell)
        for (q,r) in base_purple:
            newcell = BoardPosition(q, r, self.spacing, postype='purple')
            cells.append(newcell)
        for (q,r) in base_gray0:
            newcell = BoardPosition(q, r, self.spacing, postype='gray0')
            cells.append(newcell)


        # Sort for stable indexing: by r, then q
        cells.sort(key=lambda t: (t.r, t.q))
        self.cells = cells
        self.index_of = {(ax.q,ax.r): i for i, ax in enumerate(cells)}
        #print('index',self.index_of)

    def _project_to_pixels(self):
        """Pointy-top axial -> pixel coordinates for Tk display."""
        cart = []
        for t in self.cells:
            x = t.x
            y = t.y
            cart.append((x, y))
            #print (f"Cell (q={t.q}, r={t.r}) -> (x={x}, y={y}), {self.spacing}* {t.q} + {t.r}, {t.postype}")
        self.cartesian = cart

    def _build_rows_for_ascii(self):
        """Group axial coords by r row for ASCII rendering."""
        rows = {}
        for t in self.cells:
            rows.setdefault(t.r, []).append((t.q, t.r, t.postype))
        # sort rows by r; within each row by q
        ordered = []
        for rr in sorted(rows.keys()):
            ordered.append(sorted(rows[rr], key=lambda x: x[0]))
        self._rows = ordered

    def print_ascii(self, pins=None, empty='·'):
        """
        Print a hexagon-shaped board as ASCII.
        - pins: iterable of Pin; they will be rendered using the first letter of their color (uppercased).
        - empty: glyph for empty holes.
        """
        pin_map = {}
        if pins:
            for p in pins:
                q= self.cells[p.axialindex].q
                r= self.cells[p.axialindex].r
                #q, r = self.cells[p.index]
                pin_map[(q, r)] = (p.color[:1].upper() if p.color else 'X')

        max_width = max(len(row) for row in self._rows)  # for indentation
        for row in self._rows:
            pad = " " * (max_width - len(row))  # left indentation to form a hex outline
            parts = []
            for (q, r, t) in row:
                parts.append(pin_map.get((q, r), empty if t == 'board' else t[:1].lower()))
            print(pad + " ".join(parts))

    # --- optional helpers ---
    def axial_index(self, q, r):
        """Return index of a cell at axial (q, r). Raises KeyError if not present."""
        return self.index_of[(q, r)]

    def axial_of_index(self, idx):
        """Return axial (q, r) for an index."""
        return self.cells[idx]

    def axial_of_colour(self, colour):
        """Return list of axial (q, r) for all cells of a given colour."""
        l = [(cell.q, cell.r) for cell in self.cells if cell.postype == colour]
        return [self.index_of[(q,r)] for (q,r) in l]






