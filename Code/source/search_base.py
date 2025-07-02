import numpy as np
from typing import Iterator, Any
class SearchAlgo:
    name = "Abstract"
    def run(self, grid, start, goal) -> Iterator[tuple[np.ndarray, Any]]:
        raise NotImplementedError