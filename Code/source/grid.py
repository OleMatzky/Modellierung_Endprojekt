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

        start_x = 0
        start_y = rng.choice(range(1, self.height // 2 + 1, 2)) # Ensure start is in the upper half and odd row
        self.start = (start_y, start_x)
        self.grid[self.start] = 2

        end_x = self.width - 1
        end_y = rng.choice(range(self.height // 2 + 1, self.height - 1, 2)) # Ensure end is in the lower half and odd row
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
        tried = set()
        extra_openings = int(self.width * self.height * extra_openings / 100)  # Convert percentage to number of openings
        while extra_openings > 0:
            #get random position in the grid
            y = rng.integers(1, self.height - 1)
            x = rng.integers(1, self.width - 1)
            if self.grid[y, x] != 1:
                continue
            
            if (y, x) in tried:
                continue
            tried.add((y, x))

            neighbors = self.neighbors((y, x))
            adjacent_walls = [n for n in neighbors if self.grid[n] == 1]

            # No cross, lone  or T-shape wall
            if len(adjacent_walls) != 2:
                continue

            # Check if the two adjacent walls are in the same row or column -> no corner
            if adjacent_walls[0][0] != adjacent_walls[1][0] and adjacent_walls[0][1] != adjacent_walls[1][1]:
                continue

            # Dont carve a lone wall
            if len([n for n in self.neighbors(adjacent_walls[0]) if self.grid[n] == 0]) == 3 or len([n for n in self.neighbors(adjacent_walls[1]) if self.grid[n] == 0]) == 3:
                    continue
                
            # All test passed, carve the path
            self.grid[y, x] = 0
            extra_openings -= 1

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