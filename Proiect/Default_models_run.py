import os
from pathlib import Path
from grid import Grid
from engine import AutomatonEngine
from storage import SimulationHistory

from Default_Models.rule30 import setup_rule30, Rule30
from Default_Models.forest_fire import setup_forest, ForestFireRule
from Default_Models.sandpile import setup_sandpile, SandpileRule

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_rule30(steps = 15):
    print("=== Testare Regula 30 ===")
    grid = setup_rule30(31)
    regula = Rule30()
    engine = AutomatonEngine(grid, regula)
    istoric = SimulationHistory(metadata={"regula": "Rule30"})
    
    istoric.record(grid)
    grid.pretty_print({0: " ", 1: "█"})

    for pas, next_grid in enumerate(engine.run(steps)):
        if pas == 0:
            continue 
        istoric.record(next_grid)
        next_grid.pretty_print({0: " ", 1: "█"})
        

    file_path = OUTPUT_DIR / "rezultate_rule30.json"
    istoric.save_to_json(file_path)
    print(f"-> Istoricul a fost salvat cu succes in '{file_path}'")
    
    celule_active = istoric.count_state_at_step(-1, 1)
    print("-> Analiza simularii:")
    print(f"   Celule active la pasul final: {celule_active}\n")


def run_forest_fire(steps=5):
    print("=== Testare Forest Fire ===")
    grid = setup_forest(10, 10)
    regula = ForestFireRule()
    engine = AutomatonEngine(grid, regula)
    istoric = SimulationHistory(metadata={"regula": "ForestFire"})
        
    symbols = {"empty": ".", "tree": "T", "fire": "F"}
    istoric.record(grid)
    grid.pretty_print(symbols)
    print("-" * 20)
    
    for pas, next_grid in enumerate(engine.run(steps)):
        if pas == 0:
            continue
        istoric.record(next_grid)
        next_grid.pretty_print(symbols)
        print("-" * 20)
    

    file_path = OUTPUT_DIR / "rezultate_padure.json"
    istoric.save_to_json(file_path)
    print(f"-> Istoricul a fost salvat cu succes in '{file_path}'")
    
    copaci_initiali = istoric.count_state_at_step(0, "tree")
    copaci_finali = istoric.count_state_at_step(-1, "tree")
    
    print("-> Analiza simularii:")
    print(f"   Copaci la pasul 0: {copaci_initiali}")
    print(f"   Copaci la pasul final: {copaci_finali}")
    
    stabilitate = istoric.find_stability_step()
    if stabilitate:
        print(f"   Incendiul s-a oprit (stare stabila) la pasul: {stabilitate}\n")
    else:
        print(f"   La pasul {steps} focul inca arde sau padurea se modifica.\n")


def run_sandpile(steps = 10):
    print("=== Testare Sandpile ===")
    grid = setup_sandpile(9)
    regula = SandpileRule()
    engine = AutomatonEngine(grid, regula)
    istoric = SimulationHistory(metadata={"regula": "Sandpile"})

    istoric.record(grid)
    grid.pretty_print({}) 
    print("-" * 20)

    for pas, next_grid in enumerate(engine.run(steps)):
        if pas == 0:
            continue
        istoric.record(next_grid)
        next_grid.pretty_print({})
        print("-" * 20)

        
    file_path = OUTPUT_DIR / "rezultate_sandpile.json"
    istoric.save_to_json(file_path)
    print(f"-> Istoricul a fost salvat cu succes in '{file_path}'")
    
    stabilitate = istoric.find_stability_step()
    print("-> Analiza simularii:")
    if stabilitate:
        print(f"   Gramada de nisip s-a stabilizat la pasul: {stabilitate}\n")
    else:
        print(f"   Nisipul inca se prabuseste dupa {steps} pasi.\n")


def run():
    print("Pornim simularile si exportul de fisiere...\n")
    run_rule30(15)
    run_forest_fire(5)
    run_sandpile(10)

if __name__ == "__main__":
    run()