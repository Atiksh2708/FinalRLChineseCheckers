import tkinter as tk
from tkinter import ttk
from checkers_board import BoardPosition,HexBoard
class BoardGUI:
    """Tkinter visualization of the hex board and pins."""
    def __init__(self, board: HexBoard, pins):
        self.board = board
        self.pins = pins

        self.root = tk.Tk()
        self.root.title("Hexagonal Board")

        # Compute canvas extents
        xs = [x for x, y in board.cartesian]
        ys = [y for x, y in board.cartesian]
        pad = 60
        width = int(max(xs) - min(xs) + 2 * pad)
        height = int(max(ys) - min(ys) + 2 * pad)

        # Shift all coords so they’re centered with padding
        self.offset_x = -min(xs) + pad
        self.offset_y = -min(ys) + pad

        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="white")



        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        scrollable_frame = ttk.Frame(self.canvas)

        #scrollable_frame.bind(command=scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


        self.draw_board()
        self.draw_pins()

    def _to_canvas(self, x, y):
        return (x + self.offset_x, y + self.offset_y)

    def draw_board(self):
        colurmatches ={'red':'rosybrown1', 'lawn green':'palegreen1', 'blue':'lightblue1', 'yellow':'lightgoldenrod1', 'purple':'plum1', 'gray0':'gray20'}
        r = self.board.hole_radius
        for cells in self.board.cells:
            cx, cy = self._to_canvas(cells.x, cells.y)
            if cells.postype == 'board':
                fill_color = "lightgray"
                fill_stipple = 'gray25'
            else:
                fill_color = colurmatches[cells.postype]
            #cx, cy = self._to_canvas(x, y)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=fill_color, outline="black"
            )

    def draw_pins(self):
        r = int(self.board.hole_radius * 0.7)
        for pin in self.pins:
            x, y = pin.position
            cx, cy = self._to_canvas(x, y)
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                fill=pin.color, outline="black"
            )

    def run(self):
        self.root.mainloop()

    def refresh(self, newpins):
        self.pins = newpins
        self.draw_board()
        self.draw_pins()
