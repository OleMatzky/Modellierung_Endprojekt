import numpy as np
import heapq
import time
from typing import Iterator, Tuple
from source.search_base import SearchAlgo

class Dijkstra(SearchAlgo):
    name = "Dijkstra"

    def run(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> Iterator[Tuple[np.ndarray, Tuple[int, int, int]]]:
        """
        Generate live frames for animation using Dijkstra's pathfinding
        algorithm => similar to A* but without heuristic.

        Parameters
        ----------
        grid : np.ndarray
            2D grid where 0=free, 1=wall
        start : Tuple[int, int]
            Starting position (y, x)
        goal : Tuple[int, int]
            Goal position (y, x)

        Yields
        ------
        Tuple[np.ndarray, Tuple[int, int, int]]
            (grid, metrics) where:
            - grid: modified array with color-coded visualization
            - metrics: (nodes_visited, path_length, milliseconds)
              - path_length = np.nan until final path is found
              - milliseconds = runtime using perf_counter
        """
        h, w = grid.shape
        
        # Initialize data structures
        open_heap = [(0, start)]  # Priority queue: (f_score, node)
        g_score = {start: 0}
        parent = {}
        in_open = {start}
        
        nodes_visited = 0
        start_time = time.perf_counter_ns()

        while open_heap:
            # Get node with minimum f_score
            _, current = heapq.heappop(open_heap)
            if current not in in_open:  # Skip outdated entries
                continue
                
            in_open.remove(current)
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

                tentative_g = g_score[current] + 1
                
                # Update if better path found
                if tentative_g < g_score.get((ny, nx), float('inf')):
                    parent[(ny, nx)] = current
                    g_score[(ny, nx)] = tentative_g
                    f_score = tentative_g
                    
                    heapq.heappush(open_heap, (f_score, (ny, nx)))
                    
                    # Mark as open (yellow)
                    if (ny, nx) not in (start, goal):
                        grid[ny, nx] = 4
                    in_open.add((ny, nx))

        # Sentinel value to indicate end of generation
        yield np.array([np.nan]), (np.nan, np.nan, np.nan)
