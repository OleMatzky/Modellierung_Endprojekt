import numpy as np
import heapq
import time
from typing import Iterator, Tuple
from source.search_base import SearchAlgo

class GreedyBFS(SearchAlgo):
    name = "Greedy BFS"

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance as admissible heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def run(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Iterator[Tuple[np.ndarray, Tuple[int, int, int]]]:
        h, w = grid.shape
        open_heap = [(self._heuristic(start, goal), start)]
        parent = {}
        in_open = {start}
        closed = set()
        
        nodes_visited = 0
        start_time = time.perf_counter_ns()
        
        while open_heap:
            # Get node with minimum f_score
            _, current = heapq.heappop(open_heap)
            if current not in in_open:  # Skip outdated entries
                continue
                
            in_open.remove(current)
            closed.add(current)
            nodes_visited += 1

            # Visualize current best path (light blue)
            node = current
            while node in parent:
                if node not in (start, goal):
                    grid[node] = 5
                node = parent[node]

            # Yield current state
            elapsed_ms = int((time.perf_counter_ns() - start_time) // 1e6)
            yield grid, (nodes_visited, np.nan, elapsed_ms)

            # Check if goal is reached
            if current == goal:
                path_length = 0
                node = current
                # Color final path blue
                while node in parent:
                    if node not in (start, goal):
                        grid[node] = 6
                    node = parent[node]
                    path_length += 1

                elapsed_ms = int((time.perf_counter_ns() - start_time) // 1e6)
                yield grid, (nodes_visited, path_length, elapsed_ms)
                break

            # Expand neighbors
            cy, cx = current
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = cy + dy, cx + dx
                
                # Check bounds
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                    
                # Skip walls
                if grid[ny, nx] == 1:
                    continue
                
                # Skip already closed nodes
                if (ny, nx) in closed:
                    continue

                h_val = self._heuristic((ny, nx), goal)
                if (ny, nx) not in in_open:
                    parent[(ny, nx)] = current

                    heapq.heappush(open_heap, (h_val, (ny, nx)))

                    # Mark as open (yellow)
                    if (ny, nx) not in (start, goal):
                        grid[(ny, nx)] = 4
                    in_open.add((ny, nx))

         # Sentinel value to indicate end of generation
        yield np.array([np.nan]), (np.nan, np.nan, np.nan)