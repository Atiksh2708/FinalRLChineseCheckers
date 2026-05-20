from checkers_board import BoardPosition,HexBoard
class Pin:
    """Represents a pin placed on the board by index."""
    def __init__(self, board: HexBoard, axialindex: int, id:int, color="red"):
        self.board = board
        self.axialindex = axialindex
        self.color = color
        self.id = id
        self.board.cells[axialindex].occupied = True

    @property
    def position(self):
        """Pixel coordinates for Tkinter drawing."""
        return self.board.cartesian[self.axialindex]
    
    
    def getPossibleMoves(self):
        """
        Return a sorted list of board indices (ints) representing empty cells
        that this pin can legally move to in one turn.

        Legal moves:
        Single-step: to any adjacent empty neighbor (6 axial directions).
        Multi-hop: one or more consecutive hops; each hop jumps over one
            occupied adjacent cell and lands on the cell immediately beyond,
            which must be empty. Hops may continue from the landing cell.

        Notes:
        This function only checks board bounds and occupancy; it does not
            enforce end-zone/color restrictions (if any). Use placePin to apply
            additional placement rules.
        """
        board = self.board
        start_idx = self.axialindex

        # Axial neighbor directions on a pointy-top hex grid:
        # (q, r) neighbors = (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

        def idx_of(q, r):
            # Safe lookup: returns None if (q, r) not on the board
            return board.index_of.get((q, r), None)

        # --- Single-step neighbors ---
        start_cell = board.cells[start_idx]
        q0, r0 = start_cell.q, start_cell.r
        possible = set()

        for dq, dr in directions:
            ni = idx_of(q0 + dq, r0 + dr)
            if ni is not None and not board.cells[ni].occupied:
                possible.add(ni)

        # --- Multi-hop search (BFS/DFS; here we use a stack = DFS) ---
        visited = {start_idx}   # landing cells we've already explored (avoid loops)
        stack = [start_idx]

        while stack:
            cur = stack.pop()
            cq, cr = board.cells[cur].q, board.cells[cur].r

            for dq, dr in directions:
                # Adjacent cell to jump over
                aq, ar = cq + dq, cr + dr
                # Landing cell immediately beyond
                bq, br = cq + 2*dq, cr + 2*dr

                adj_idx = idx_of(aq, ar)
                land_idx = idx_of(bq, br)

                if adj_idx is None or land_idx is None:
                    continue  # off-board in this direction

                # Hop rule: adjacent must be occupied; landing must be empty
                if board.cells[adj_idx].occupied and not board.cells[land_idx].occupied:
                    if land_idx not in visited:
                        possible.add(land_idx)
                        visited.add(land_idx)
                        stack.append(land_idx)  # continue chaining hops from here

        return sorted(possible)


    
    def placePin(self, new_axialindex:int):
        """Move pin to a new index on the board."""
        if int(new_axialindex) < 0 or int(new_axialindex) >= len(self.board.cells):
            print("Pin index out of bounds for this board.")
            return False
        #check if new_index is occupied?
        if self.board.cells[new_axialindex].occupied == True:
            print("Cannot place pin here; position occupied.")
            return False
        '''if self.board.cells[new_axialindex].postype!='board':
            if self.board.cells[new_axialindex].postype != self.board.colour_opposites[self.color] and self.board.cells[new_axialindex].postype != self.color:
                print("Cannot place pin here; Other colour position.", self.board.cells[new_axialindex].postype)
                return False'''
        #removed above rule to let pins move anywhere
        
        self.board.cells[self.axialindex].occupied = False
        self.axialindex = int(new_axialindex)
        self.board.cells[int(new_axialindex)].occupied = True
        print('Pin placed successfully.')
        return True

