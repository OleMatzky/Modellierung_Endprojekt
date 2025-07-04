import numpy as np
from numpy.random import Generator
from typing import Iterator

class Grid:
    def __init__(self):
        self.grid = None
        self.start = None
        self.goal = None
        self.width = 0
        self.height = 0

    def neighbors(self, cell: tuple[int, int], step: int = 1):
        y, x = cell
        directions = [(-step, 0), (step, 0), (0, -step), (0, step)]
        result = []
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= nx < self.width and 0 <= ny < self.height:
                result.append((ny, nx))
        return result

    def generate_maze_grid(self, width: int, height: int, rng:Generator, extra_openings: int = 10, animate: bool = False)-> Iterator[np.ndarray]:
        self.width = width if width % 2 == 1 else width + 1 # x-dimension
        self.height = height if height % 2 == 1 else height + 1 # y-dimension

        self.grid = np.ones((self.height, self.width), dtype=int)  # 1 = wall, 0 = path

        def rand_odd(rng: Generator, low: int, high: int) -> int:
            if low % 2 == 0:
                low += 1                
                high -= 1             
            if low > high:
                raise ValueError("Kein ungerader Wert in diesem Bereich")

            n = (high - low) // 2 + 1 
            return low + 2 * rng.integers(n)

        start_x = 0
        start_y = rand_odd(rng, 1, self.height // 4) # first quarter and odd row
        self.start = (start_y, start_x)
        self.grid[self.start] = 2

        end_x = self.width - 1
        end_lower_bound = 3 * self.height // 4
        end_y = rand_odd(rng, end_lower_bound, self.height - 2) # last quarter and odd row
        self.goal = (end_y, end_x)
        self.grid[self.goal] = 3

        #depth-first search to carve out the maze (iterative via stack to prevent recursion limit issues)
        visited = np.zeros(self.grid.shape, dtype=bool)
        stack = []

        # Ensure path from start to end
        first_cell = (start_y, 1)
        last_cell = (end_y, self.width - 2)
        self.grid[first_cell] = 0
        self.grid[last_cell] = 0
        
        visited[first_cell] = True
        stack.append(first_cell)

        # for animation
        if animate:
            yield self.grid.copy()

        while stack:
            current = stack[-1]
            neighbors = [
                n for n in self.neighbors(current,step=2)
                if not visited[n]
            ]
            if not neighbors:
                stack.pop()
                continue

            rng.shuffle(neighbors)

            next_cell = neighbors[0]
            self.grid[next_cell] = 0
            visited[next_cell] = True

            between_x = (next_cell[1] + current[1]) // 2
            between_y = (next_cell[0] + current[0]) // 2
            self.grid[between_y, between_x] = 0

            stack.append(next_cell)

            # for animation
            if animate:
                yield self.grid.copy()

        # Add extra openings
        inner_walls = np.argwhere(
            (self.grid == 1) &
            (np.arange(self.height)[:, None] > 0) & (np.arange(self.height)[:, None] < self.height-1) &   # y ≠ 0, h-1
            (np.arange(self.width)     > 0) & (np.arange(self.width)     < self.width-1)               # x ≠ 0, w-1
        )
        rng.shuffle(inner_walls)

        # gewünschte Anzahl tatsächlicher Öffnungen
        openings_left = int(self.width * self.height * extra_openings / 100)

        def is_candidate(y, x):
            neigh = self.neighbors((y, x))
            adj_walls = [n for n in neigh if self.grid[n] == 1]
            # No cross, lone  or T-shape wall
            if len(adj_walls) != 2:
                return False
            
            # Check if the two adjacent walls are in the same row or column -> no corner
            if adj_walls[0][0] != adj_walls[1][0] and adj_walls[0][1] != adj_walls[1][1]:
                return False

            # No lone wall afterwards
            if not all(sum(self.grid[n] == 1 for n in self.neighbors(w)) >= 2 for w in adj_walls):
                return False
            
            return True

        for y, x in inner_walls:
            if openings_left == 0:
                break
            if is_candidate(y, x):
                self.grid[y, x] = 0
                openings_left -= 1

            # for animation
            if animate:
                yield self.grid.copy()

        if not animate:
            yield self.grid.copy()
        else: 
            # Sentinel value to indicate end of generation
            yield np.array([np.nan]) 

    def get_grid(self):
        return self.grid

    def get_start_goal(self):
        return self.start, self.goal