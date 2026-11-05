from abc import ABC, abstractmethod
from typing import List, Any, Generator


class CellularRule(ABC):
    @property
    @abstractmethod
    def neighborhood_type(self) -> str:
        pass

    @abstractmethod
    def apply(self, cell_state: Any, neighbors: List[Any]) -> Any:
        pass



class AutomatonEngine:
    
    def __init__(self, initial_grid: Any, rule: CellularRule):
        self.current_grid = initial_grid
        self.rule = rule

    def step(self) -> Any:
        next_grid = self.current_grid.clone()
        
        for r in range(self.current_grid.height):
            for c in range(self.current_grid.width):
                current_val = self.current_grid.get_cell(r, c)
                neighbors = self.current_grid.get_neighbors(r, c, self.rule.neighborhood_type)
                
                new_val = self.rule.apply(current_val, neighbors)
                next_grid.set_cell(r, c, new_val)
                
        self.current_grid = next_grid
        return self.current_grid

    def run(self, steps: int) -> Generator[Any, None, None]:
        yield self.current_grid 
        
        for _ in range(steps):
            yield self.step()
