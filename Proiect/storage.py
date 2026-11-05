from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


from grid import Grid 


class SimulationHistory:
    """
    Gestionează istoricul stărilor produse în timpul simulării unui automat celular.
    Permite stocarea generațiilor succesive, recuperarea stărilor la pași arbitrari
    și inspecția evoluției sistemului.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Inițializează un istoric gol.
        
        Args:
            metadata: Informații opționale despre experiment 
                      (ex. numele regulii, autor, data rulării).
        """
        self._history: List[Grid] = []
        self.metadata: Dict[str, Any] = metadata or {}

    def record(self, grid: Grid) -> None:
        """
        Adaugă o nouă generație în istoric.
        
        Folosește grid.clone() pentru a garanta independența memoriei: dacă engine-ul
        sau utilizatorul modifică instanța `grid` ulterior, starea salvată în istoric
        rămâne nealterată.
        """
        if not isinstance(grid, Grid):
            raise TypeError(f"Se aștepta o instanță de Grid, primit {type(grid).__name__}")
        
        self._history.append(grid.clone())

    def get_generation(self, step: int) -> Grid:
        """
        Returnează o copie a stării de la pasul cerut.
        
        Suportă indexare standard (0, 1, ..., N) și indexare negativă (-1 pentru ultima stare).
        """
        if not self._history:
            raise IndexError("Istoricul este gol. Nu există nicio generație salvată.")
        
        if step < -len(self._history) or step >= len(self._history):
            raise IndexError(f"Pasul {step} este în afara limitelor (0..{len(self._history) - 1}).")
        
       
        return self._history[step].clone()

    def latest(self) -> Grid:
        """Returnează ultima generație adăugată în istoric."""
        return self.get_generation(-1)

    def initial(self) -> Grid:
        """Returnează starea inițială G_0."""
        return self.get_generation(0)

    @property
    def total_steps(self) -> int:
        """Returnează numărul total de generații stocate (inclusiv starea inițială G_0)."""
        return len(self._history)

    def clear(self) -> None:
        """Golește istoricul curent."""
        self._history.clear()

    def __len__(self) -> int:
        return self.total_steps

    def __iter__(self):
        """Permite parcurgerea generațiilor: for step_grid in history: ..."""
        for g in self._history:
            yield g.clone()

    def save_to_json(self, filepath: Union[str, Path]) -> None:
        """
        Salvează întregul istoric al simulării într-un fișier JSON.
        """
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
        """
        Reconstruiește o instanță SimulationHistory dintr-un fișier JSON.
        """
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

    # ==========================================
    # Partea 3: Metode de Analiză și Interogare
    # ==========================================

    def count_state_at_step(self, step: int, state: Any) -> int:
        """
        Numără câte celule au o anumită stare la pasul specificat.
        
        Args:
            step: Indexul generației (poate fi și negativ, ex. -1 pentru ultima).
            state: Valoarea stării căutate (ex. 1, 'tree', 'fire', True).
        """
        grid = self.get_generation(step)
        # np.count_nonzero funcționează rapid atât pentru valori numerice cât și pentru string-uri/obiecte
        return int(np.count_nonzero(grid.readonly_view() == state))

    def get_state_distribution(self, step: int) -> Dict[Any, int]:
        """
        Returnează frecvența tuturor stărilor prezente în grilă la pasul dat.
        
        Exemplu output: {0: 2400, 1: 100} sau {'empty': 1500, 'tree': 900, 'fire': 100}
        """
        grid = self.get_generation(step)
        unique_vals, counts = np.unique(grid.readonly_view(), return_counts=True)
        return {val: int(cnt) for val, cnt in zip(unique_vals, counts)}

    def get_state_trajectory(self, state: Any) -> List[int]:
        """
        Construiește o listă cu numărul de celule de un anumit tip pentru fiecare pas (de la G_0 la G_N).
        Foarte util pentru grafice sau analiza vitezei de propagare (ex. curba de infecție / foc).
        """
        trajectory: List[int] = []
        for g in self._history:
            cnt = int(np.count_nonzero(g.readonly_view() == state))
            trajectory.append(cnt)
        return trajectory

    def find_stability_step(self) -> Optional[int]:
        """
        Detectează dacă și la ce pas sistemul a atins o stare staționară (G_t == G_{t-1}).
        
        Returns:
            Indexul primei generații unde starea devine identică cu precedenta, 
            sau None dacă sistemul este încă activ/în schimbare.
        """
        if len(self._history) < 2:
            return None

        for t in range(1, len(self._history)):
            # Folosește operatorul == implementat deja în Grid (__eq__)
            if self._history[t] == self._history[t - 1]:
                return t
        return None

    def is_stable(self) -> bool:
        """Verifică dacă ultimele două generații stocate sunt identice."""
        if len(self._history) < 2:
            return False
        return self._history[-1] == self._history[-2]

    def find_cycle(self, max_period: int = 15) -> Optional[Tuple[int, int]]:
        """
        Detectează dacă sistemul a intrat într-o buclă periodică (oscilator).
        Verifică dacă starea de la pasul curent t este identică cu o stare anterioară t - p,
        unde p este lungimea perioadei (1 <= p <= max_period).

        Returns:
            (start_step, period) - pasul de la care se repetă și perioada ciclului,
            sau None dacă nu este detectat niciun ciclu.
        """
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
