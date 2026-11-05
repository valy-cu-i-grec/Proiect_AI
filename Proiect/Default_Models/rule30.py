from grid import Grid
from engine import CellularRule

class Rule30(CellularRule):
    
    @property
    def neighborhood_type(self) -> str:
        return "1d"

    def apply(self, current_state: int, neighbors: list) -> int:
        left = neighbors[0]
        right = neighbors[1]
        
        pattern = (left, current_state, right)
        
        if pattern in [(1, 0, 0), (0, 1, 1), (0, 1, 0), (0, 0, 1)]:
            return 1
        return 0

def setup_rule30(width: int = 31) -> Grid:
    grid = Grid.one_dimensional(width, boundary="toroidal", dtype=int)
    grid.init_empty(0)
    
    center_col = width // 2
    grid.set_cell(0, center_col, 1)
    
    return grid