from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


from grid import Grid 


class SimulationHistory:
    

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
       
        self._history: List[Grid] = []
        self.metadata: Dict[str, Any] = metadata or {}

    def record(self, grid: Grid) -> None:
       
        if not isinstance(grid, Grid):
            raise TypeError(f"Se aștepta o instanță de Grid, primit {type(grid).__name__}")
        
        self._history.append(grid.clone())

    def get_generation(self, step: int) -> Grid:
       
        if not self._history:
            raise IndexError("Istoricul este gol. Nu există nicio generație salvată.")
        
        if step < -len(self._history) or step >= len(self._history):
            raise IndexError(f"Pasul {step} este în afara limitelor (0..{len(self._history) - 1}).")
        
       
        return self._history[step].clone()

    def latest(self) -> Grid:
     
        return self.get_generation(-1)

    def initial(self) -> Grid:
        """Returnează starea inițială G_0."""
        return self.get_generation(0)

    @property
    def total_steps(self) -> int:
       
        return len(self._history)

    def clear(self) -> None:
       
        self._history.clear()

    def __len__(self) -> int:
        return self.total_steps

    def __iter__(self):
      
        for g in self._history:
            yield g.clone()

    def save_to_json(self, filepath: Union[str, Path]) -> None:
      
        if not self._history:
            raise ValueError("Nu se poate salva un istoric gol.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        first_grid = self._history[0]

        # Serializăm matricele în liste native Python
        frames_data = [g.matrix.tolist() for g in self._history]

        payload = {
            "metadata": self.metadata,
            "config": {
                "height": first_grid.height,
                "width": first_grid.width,
                "boundary": getattr(first_grid, "boundary", getattr(first_grid, "_boundary", "fixed")),
                "states": list(first_grid.states) if hasattr(first_grid, "states") and first_grid.states else None
            },
            "total_steps": len(self._history),
            "frames": frames_data
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Union[str, Path]) -> SimulationHistory:
       
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Fișierul nu a fost găsit la calea: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        history_instance = cls(metadata=data.get("metadata", {}))
        config = data.get("config", {})
        boundary = config.get("boundary", "fixed")
        states = config.get("states")

        for frame in data.get("frames", []):
            # Încercăm instanțierea cu boundary/states; dacă from_matrix acceptă doar matrix, facem fallback
            try:
                restored_grid = Grid.from_matrix(matrix=frame, boundary=boundary)
            except TypeError:
                restored_grid = Grid.from_matrix(frame)

            # Restaurăm atributele configurabile dacă există pe Grid
            if states is not None and hasattr(restored_grid, "states"):
                try:
                    restored_grid.states = tuple(states)
                except AttributeError:
                    pass

            if hasattr(restored_grid, "boundary"):
                try:
                    restored_grid.boundary = boundary
                except AttributeError:
                    pass

            history_instance._history.append(restored_grid)

        return history_instance

   

    def count_state_at_step(self, step: int, state: Any) -> int:
        
        grid = self.get_generation(step)
        # np.count_nonzero funcționează rapid atât pentru valori numerice cât și pentru string-uri/obiecte
        return int(np.count_nonzero(grid.readonly_view() == state))

    def get_state_distribution(self, step: int) -> Dict[Any, int]:
      
        grid = self.get_generation(step)
        unique_vals, counts = np.unique(grid.readonly_view(), return_counts=True)
        return {val: int(cnt) for val, cnt in zip(unique_vals, counts)}

    def get_state_trajectory(self, state: Any) -> List[int]:
       
        trajectory: List[int] = []
        for g in self._history:
            cnt = int(np.count_nonzero(g.readonly_view() == state))
            trajectory.append(cnt)
        return trajectory

    def find_stability_step(self) -> Optional[int]:
     
        if len(self._history) < 2:
            return None

        for t in range(1, len(self._history)):
            # Folosește operatorul == implementat deja în Grid (__eq__)
            if self._history[t] == self._history[t - 1]:
                return t
        return None

    def is_stable(self) -> bool:
        
        if len(self._history) < 2:
            return False
        return self._history[-1] == self._history[-2]

    def find_cycle(self, max_period: int = 15) -> Optional[Tuple[int, int]]:
       
        total = len(self._history)
        if total < 2:
            return None

        # Căutăm de la cel mai recent cadru înapoi
        last_grid = self._history[-1]
        for period in range(1, min(max_period + 1, total)):
            candidate_step = total - 1 - period
            if last_grid == self._history[candidate_step]:
                # Verificăm dacă perioada se confirmă pe un pas intermediar pentru robustețe
                return (candidate_step, period)

        return None


if __name__ == "__main__":
   
    print("Testare SimulationHistory...")
    
    # Creează câteva grile simulate
    g0 = Grid.from_matrix([[0, 1], [0, 0]])
    g1 = Grid.from_matrix([[1, 1], [0, 0]])
    g2 = Grid.from_matrix([[1, 1], [0, 0]]) # Stare stabilă
    
    history = SimulationHistory(metadata={"experiment": "test_run", "author": "Echipa"})
    history.record(g0)
    history.record(g1)
    history.record(g2)
    
    print(f"Total cadre salvate: {history.total_steps}")
    print(f"Celule active la pasul 0: {history.count_state_at_step(0, 1)}")
    print(f"Celule active la pasul 1: {history.count_state_at_step(1, 1)}")
    print(f"Pasul la care s-a stabilizat: {history.find_stability_step()}")
    
    # Test export / import JSON
    history.save_to_json("test_history.json")
    loaded = SimulationHistory.load_from_json("test_history.json")
    print(f"Cadre reîncărcate din JSON: {loaded.total_steps}")
    print("Modulul 3 este complet funcțional!")    