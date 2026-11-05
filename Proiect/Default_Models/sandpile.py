from grid import Grid
from engine import CellularRule

class SandpileRule(CellularRule):
    @property
    def neighborhood_type(self) -> str:
        return "von_neumann"

    def apply(self, current_state: int, neighbors: list) -> int:
        remains = current_state % 4
        
        received = sum(n // 4 for n in neighbors)
        
        return remains + received

def setup_sandpile(size: int = 15) -> Grid:
    grid = Grid(size, size, boundary="fixed", fixed_value=0, dtype=int)
    grid.init_empty(0)

    center = size // 2
    grid.set_cell(center, center, 10000)

    return grid