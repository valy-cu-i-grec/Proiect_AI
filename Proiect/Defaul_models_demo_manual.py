import time
import sys
from grid import Grid
from Default_Models.rule30 import setup_rule30, rule30_transition
from Default_Models.forest_fire import setup_forest, fire_transition
from Default_Models.sandpile import setup_sandpile, sandpile_transition

def run_rule30(steps = 15):
    print("=== Testare regula30 ===")
    try:
        grid = setup_rule30(31)
    except NameError:
        return

    grid.pretty_print({0: " ", 1: "█"})

    for _ in range(steps):
        next_grid = grid.clone()
        for c in range(grid.width):
            neighbors = grid.get_neighbors(0, c, neighborhood="1d").tolist()
            new_state = rule30_transition(grid.get_cell(0,c), neighbors)
            next_grid.set_cell(0,c,new_state)
        grid = next_grid
        grid.pretty_print({0: " ", 1: "█"})
    print("\n")

def run_forest_fire(steps=5):
    print("=== Testare Forest Fire ===")
    try:
        grid = setup_forest(10, 10)
    except NameError:
        return
        
    symbols = {"empty": ".", "tree": "T", "fire": "F"}
    grid.pretty_print(symbols)
    print("-" * 20)
    
    for _ in range(steps):
        next_grid = grid.clone()
        for r in range(grid.height):
            for c in range(grid.width):
                counts = grid.count_neighbor_states(r, c, neighborhood="moore")
                new_state = fire_transition(grid.get_cell(r, c), counts)
                next_grid.set_cell(r, c, new_state)
        grid = next_grid
        grid.pretty_print(symbols)
        print("-" * 20)
    print("\n")

def run_sandpile(steps = 10):
    print("=== Test sandpile ===")
    try:
        grid = setup_sandpile(9)
    except NameError:
        return

    grid.pretty_print()
    print("\n")

    for _ in range(steps):
        next_grid = grid.clone()
        for r in range(grid.height):
            for c in range(grid.width):
                neighbors = grid.get_neighbors(r, c, neighborhood="von_neumann").tolist()
                new_state = sandpile_transition(grid.get_cell(r,c), neighbors)
                next_grid.set_cell(r,c,new_state)
        grid = next_grid
        grid.pretty_print()
        print("\n")

def run():
    print("Pornim simularile...")
    run_rule30(15)
    run_forest_fire(5)
    run_sandpile(10)

if __name__ == "__main__":
    run()
