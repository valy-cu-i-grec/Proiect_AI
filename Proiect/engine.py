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



if __name__ == "__main__":
    import copy

    class MockGrid:
        def __init__(self, data):
            self.data = data
            self.height = len(data)
            self.width = len(data[0]) if self.height > 0 else 0

        def get_cell(self, r, c):
            return self.data[r][c]

        def set_cell(self, r, c, val):
            self.data[r][c] = val

        def get_neighbors(self, r, c, n_type):
            left = self.data[r][c-1] if c > 0 else 0
            right = self.data[r][c+1] if c < self.width - 1 else 0
            return [left, right]

        def clone(self):
            return MockGrid(copy.deepcopy(self.data))
            
        def __str__(self):
            return "\n".join(str(row) for row in self.data)

    class SumRule(CellularRule):
        @property
        def neighborhood_type(self) -> str:
            return "1d"
            
        def apply(self, cell_state, neighbors):
            return cell_state + sum(neighbors)

    print("--- Inițializare ---")
    stare_initiala = MockGrid([[0, 1, 0, 0, 1]])
    regula = SumRule()
    engine = AutomatonEngine(stare_initiala, regula)
    
    for step_num, current_state in enumerate(engine.run(steps=3)):
        print(f"\nGenerația {step_num}:")
        print(current_state)