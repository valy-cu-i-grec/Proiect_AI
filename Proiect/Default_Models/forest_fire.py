import random
from grid import Grid
from engine import CellularRule

class ForestFireRule(CellularRule):
    def __init__(self, p_growth: float = 0.05, p_ignite: float = 0.0001):
        self.p_growth = p_growth
        self.p_ignite = p_ignite

    @property
    def neighborhood_type(self) -> str:
        return "moore"

    def apply(self, current_state: str, neighbors: list) -> str:
        if current_state == "fire":
            return "empty"

        elif current_state == "empty":
            if random.random() < self.p_growth:
                return "tree"
            return "empty"

        elif current_state == "tree":
            vecini_lista = list(neighbors)
            if vecini_lista.count("fire") > 0 or random.random() < self.p_ignite:
                return "fire"
            return "tree"
        
        return current_state

def setup_forest(height: int = 20, width: int = 20) -> Grid:
    grid = Grid(height, width, states=("empty", "tree", "fire"), boundary="fixed", fixed_value="empty")
    grid.init_random(states=("empty", "tree", "fire"), probabilities=[0.50, 0.49, 0.01])
    return grid