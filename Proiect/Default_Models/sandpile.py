from grid import Grid

def sandpile_transition(current_state: int, neighbors: list) -> int:
    remains = current_state % 4

    recieved = sum(n // 4 for n in neighbors)

    return remains + recieved

def setup_sandpile(size: int = 15) -> Grid:
    grid = Grid(size, size, boundary="fixed", fixed_value=0, dtype=int)
    grid.init_empty(0);

    center = size // 2;
    grid.set_cell(center, center, 10000);

    return grid;