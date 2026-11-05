from __future__ import annotations

import copy
from collections import Counter
from functools import lru_cache
from typing import Any, Dict, Hashable, Mapping, Optional, Sequence, Tuple

import numpy as np

Offset = Tuple[int, int]

TOROIDAL = "toroidal"
FIXED = "fixed"
_BOUNDARY_ALIASES = {"toroidal": TOROIDAL, "fixed": FIXED, "dead": FIXED}

_DEFAULT_SYMBOLS: Mapping[Hashable, str] = {0: ".", 1: "#"}

@lru_cache(maxsize=None)
def neighborhood_offsets(neighborhood: str = "moore", radius: int = 1) -> Tuple[Offset, ...]:
    if radius < 1:
        raise ValueError(f"radius trebuie să fie >= 1, primit {radius}")
    kind = neighborhood.lower()
    span = range(-radius, radius + 1)
    if kind == "moore":
        offsets = [(dr, dc) for dr in span for dc in span if (dr, dc) != (0, 0)]
    elif kind in ("von_neumann", "neumann"):
        offsets = [
            (dr, dc)
            for dr in span
            for dc in span
            if (dr, dc) != (0, 0) and abs(dr) + abs(dc) <= radius
        ]
    elif kind in ("1d", "linear"):
        offsets = [(0, dc) for dc in span if dc != 0]
    else:
        raise ValueError(
            f"Vecinătate necunoscută: {neighborhood!r} "
            "(valori acceptate: 'moore', 'von_neumann', '1d')"
        )
    return tuple(offsets)


class Grid:

    def __init__(
        self,
        height: int,
        width: int,
        *,
        boundary: str = TOROIDAL,
        fixed_value: Any = None,
        dtype: Any = None,
        states: Optional[Sequence[Hashable]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if height < 1 or width < 1:
            raise ValueError(f"Dimensiuni invalide: ({height}, {width}); trebuie >= 1")
        self._height = int(height)
        self._width = int(width)
        self._boundary = self._normalize_boundary(boundary)

        self._states: Optional[Tuple[Hashable, ...]] = None
        if states is not None:
            self._states = tuple(states)
            if not self._states:
                raise ValueError("`states` nu poate fi gol")
            if len(set(self._states)) != len(self._states):
                raise ValueError("`states` conține valori duplicate")

        self._dtype = np.dtype(dtype) if dtype is not None else self._infer_dtype(self._states)

        default: Any = False if self._dtype.kind == "b" else 0
        if self._states is not None and default not in self._states:
            default = self._states[0]
        self._default = default

        self._fixed_value = default if fixed_value is None else fixed_value
        self._check_value(self._fixed_value)

        self._rng = np.random.default_rng(seed)
        self._cells = np.full((self._height, self._width), default, dtype=self._dtype)

    @classmethod
    def one_dimensional(cls, width: int, **kwargs: Any) -> "Grid":
        """Creează o grilă 1D (height == 1). Acceptă aceleași argumente ca `Grid`."""
        return cls(1, width, **kwargs)

    @classmethod
    def from_matrix(
        cls,
        matrix: Any,
        *,
        boundary: str = TOROIDAL,
        fixed_value: Any = None,
        dtype: Any = None,
        states: Optional[Sequence[Hashable]] = None,
        seed: Optional[int] = None,
    ) -> "Grid":
        """Creează o grilă dintr-o listă de liste sau un `np.ndarray`.

        O secvență 1D este interpretată ca grilă cu un singur rând. Datele sunt
        copiate, deci modificările ulterioare ale sursei nu afectează grila.

        Raises:
            ValueError: matrice goală, neregulată (rânduri de lungimi diferite),
                cu mai mult de 2 dimensiuni sau cu valori în afara `states`.
        """
        try:
            arr = np.asarray(matrix, dtype=dtype)
        except ValueError as exc:
            raise ValueError("Matricea trebuie să fie regulată (rânduri de aceeași lungime)") from exc
        if arr.dtype == object and arr.ndim == 1 and any(isinstance(x, (list, tuple)) for x in arr):
            raise ValueError("Matricea trebuie să fie regulată (rânduri de aceeași lungime)")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.size == 0:
            raise ValueError(f"Se așteaptă o matrice 2D nevidă, primit shape={arr.shape}")

        if dtype is None and states is None:
            # Șirurile de caractere devin etichete (object), nu tablouri de lungime fixă.
            dtype = np.dtype(object) if arr.dtype.kind in "USO" else arr.dtype

        grid = cls(
            arr.shape[0],
            arr.shape[1],
            boundary=boundary,
            fixed_value=fixed_value,
            dtype=dtype,
            states=states,
            seed=seed,
        )
        grid._check_array(arr)
        grid._cells[...] = arr  # atribuirea copiază și convertește la dtype-ul grilei
        return grid

    # ------------------------------------------------------------------ #
    # Proprietăți
    # ------------------------------------------------------------------ #
    @property
    def height(self) -> int:
        return self._height

    @property
    def width(self) -> int:
        return self._width

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._height, self._width)

    @property
    def is_1d(self) -> bool:
        return self._height == 1

    @property
    def boundary(self) -> str:
        """Condiția de frontieră normalizată: 'toroidal' sau 'fixed'."""
        return self._boundary

    @property
    def fixed_value(self) -> Any:
        return self._fixed_value

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def states(self) -> Optional[Tuple[Hashable, ...]]:
        return self._states

    @property
    def matrix(self) -> np.ndarray:
        """Copie independentă a matricei curente; modificarea ei nu afectează grila."""
        return self._cells.copy()

    def readonly_view(self) -> np.ndarray:
        """View read-only (fără copiere) al matricei interne, pentru citiri rapide.

        Reflectă modificările ulterioare ale grilei, dar nu poate fi scris.
        """
        view = self._cells.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------ #
    # Inițializare
    # ------------------------------------------------------------------ #
    def init_empty(self, default_val: Any = 0) -> "Grid":
        """Umple toată grila cu `default_val`. Returnează `self` (pentru înlănțuire)."""
        self._check_value(default_val)
        self._cells[...] = default_val
        return self

    def init_random(
        self,
        states: Sequence[Hashable],
        probabilities: Optional[Sequence[float]] = None,
    ) -> "Grid":
        """Umple grila aleator, independent pentru fiecare celulă.

        Args:
            states: stările din care se extrage.
            probabilities: probabilitățile asociate (suma = 1). Implicit: uniform.

        Returns:
            `self`, pentru înlănțuire.
        """
        states = list(states)
        if not states:
            raise ValueError("`states` nu poate fi gol")
        p: Optional[np.ndarray] = None
        if probabilities is not None:
            p = np.asarray(probabilities, dtype=float)
            if p.shape != (len(states),):
                raise ValueError("`probabilities` trebuie să aibă aceeași lungime ca `states`")
            if (p < 0).any() or not np.isclose(p.sum(), 1.0):
                raise ValueError("`probabilities` trebuie să fie nenegative și să însumeze 1")
        palette = np.empty(len(states), dtype=self._dtype)
        for i, s in enumerate(states):
            self._check_value(s)
            palette[i] = s
        # Se extrag indici și se mapează pe palete, ca etichetele (str) să rămână obiecte Python.
        idx = self._rng.choice(len(states), size=self.shape, p=p)
        self._cells[...] = palette[idx]
        return self

    def set_cell(self, r: int, c: int, val: Any) -> None:
        """Scrie `val` în celula (r, c). Coordonatele trebuie să fie în interiorul grilei."""
        self._check_coords(r, c)
        self._check_value(val)
        self._cells[r, c] = val

    def get_cell(self, r: int, c: int) -> Any:
        """Citește celula (r, c). Coordonatele trebuie să fie în interiorul grilei."""
        self._check_coords(r, c)
        return self._cells[r, c].item() if self._dtype.kind != "O" else self._cells[r, c]

    # ------------------------------------------------------------------ #
    # Vecinătăți
    # ------------------------------------------------------------------ #
    def get_neighbors(
        self, r: int, c: int, neighborhood: str = "moore", radius: int = 1
    ) -> np.ndarray:
        """Valorile vecinilor celulei (r, c), în ordinea din `neighborhood_offsets`.

        * 'toroidal': coordonatele se reîntorc prin modulo. Pe grile mai mici
          decât vecinătatea, aceeași celulă (inclusiv cea centrală) poate apărea
          de mai multe ori.
        * 'fixed': vecinii din afara grilei au valoarea `fixed_value`.

        Args:
            neighborhood: 'moore', 'von_neumann' sau '1d' (stânga/dreapta pe rândul r).
            radius: raza vecinătății (>= 1).
        """
        self._check_coords(r, c)
        offs = np.asarray(neighborhood_offsets(neighborhood, radius), dtype=np.intp)
        rows = r + offs[:, 0]
        cols = c + offs[:, 1]

        if self._boundary == TOROIDAL:
            return self._cells[rows % self._height, cols % self._width]

        inside = (rows >= 0) & (rows < self._height) & (cols >= 0) & (cols < self._width)
        out = np.full(len(offs), self._fixed_value, dtype=self._dtype)
        out[inside] = self._cells[rows[inside], cols[inside]]
        return out

    def count_neighbor_states(
        self, r: int, c: int, neighborhood: str = "moore", radius: int = 1
    ) -> Dict[Hashable, int]:
        """Frecvența fiecărei stări în vecinătatea celulei (r, c).

        Dacă grila are `states` definite, toate apar în rezultat (cu 0 unde lipsesc).
        """
        values = self.get_neighbors(r, c, neighborhood, radius).tolist()
        counts = Counter(values)
        if self._states is None:
            return dict(counts)
        result: Dict[Hashable, int] = {s: counts.get(s, 0) for s in self._states}
        result.update({k: v for k, v in counts.items() if k not in result})
        return result

    def clone(self) -> "Grid":
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new._cells = self._cells.copy()
        new._rng = copy.deepcopy(self._rng)
        return new

    copy = clone
    __copy__ = clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return self.shape == other.shape and bool(np.array_equal(self._cells, other._cells))

    __hash__ = None

    def render(self, symbols: Optional[Mapping[Hashable, str]] = None) -> str:
        table = _DEFAULT_SYMBOLS if symbols is None else symbols
        rows = [[table.get(v, str(v)) for v in row] for row in self._cells.tolist()]
        cell_width = max(len(s) for row in rows for s in row)
        sep = "" if cell_width == 1 else " "
        return "\n".join(sep.join(s.rjust(cell_width) for s in row) for row in rows)

    def pretty_print(self, symbols: Optional[Mapping[Hashable, str]] = None) -> None:
        print(self.render(symbols))

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return (
            f"Grid(height={self._height}, width={self._width}, "
            f"boundary={self._boundary!r}, dtype={self._dtype})"
        )

    @staticmethod
    def _normalize_boundary(boundary: str) -> str:
        try:
            return _BOUNDARY_ALIASES[boundary.lower()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"Frontieră necunoscută: {boundary!r} (valori acceptate: 'toroidal', 'fixed', 'dead')"
            ) from None

    @staticmethod
    def _infer_dtype(states: Optional[Tuple[Hashable, ...]]) -> np.dtype:
        if states is None:
            return np.dtype(int)
        if all(isinstance(s, (bool, np.bool_)) for s in states):
            return np.dtype(bool)
        if all(isinstance(s, (int, np.integer)) and not isinstance(s, (bool, np.bool_)) for s in states):
            return np.dtype(int)
        return np.dtype(object)  # etichete (str etc.)

    def _check_coords(self, r: int, c: int) -> None:
        if not (0 <= r < self._height and 0 <= c < self._width):
            raise IndexError(f"Celula ({r}, {c}) este în afara grilei {self.shape}")

    def _check_value(self, val: Any) -> None:
        if self._states is not None and val not in self._states:
            raise ValueError(f"Valoarea {val!r} nu este o stare permisă: {self._states}")

    def _check_array(self, arr: np.ndarray) -> None:
        if self._states is not None:
            invalid = set(arr.ravel().tolist()) - set(self._states)
            if invalid:
                raise ValueError(f"Valori care nu sunt stări permise: {sorted(invalid, key=repr)}")

def _demo() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=== 1. Inițializare ===")
    g = Grid(3, 4).init_empty(0)
    g.set_cell(1, 2, 1)
    print(g, "\n")
    assert g.get_cell(1, 2) == 1 and g.shape == (3, 4)

    rnd = Grid(4, 8, seed=42).init_random([0, 1], probabilities=[0.7, 0.3])
    print(rnd, "\n")
    assert rnd == Grid(4, 8, seed=42).init_random([0, 1], probabilities=[0.7, 0.3])  # reproductibil

    m = g.matrix
    m[0, 0] = 99 
    assert g.get_cell(0, 0) == 0
    try:
        g.readonly_view()[0, 0] = 5
        raise AssertionError("view-ul ar fi trebuit să fie read-only")
    except ValueError:
        pass

    print("=== 2. Vecini 2D: toroidal vs fixed (celula (0,0)) ===")
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    torus = Grid.from_matrix(data, boundary="toroidal")
    fixed = Grid.from_matrix(data, boundary="fixed") 
    print(torus.render({}), "\n") 
    for kind in ("moore", "von_neumann"):
        t = torus.get_neighbors(0, 0, kind).tolist()
        f = fixed.get_neighbors(0, 0, kind).tolist()
        print(f"{kind:12s} toroidal={t}  fixed={f}")
    assert torus.get_neighbors(0, 0, "moore").tolist() == [9, 7, 8, 3, 2, 6, 4, 5]
    assert fixed.get_neighbors(0, 0, "moore").tolist() == [0, 0, 0, 0, 2, 0, 4, 5]
    assert torus.get_neighbors(0, 0, "von_neumann").tolist() == [7, 3, 2, 4]
    assert fixed.get_neighbors(0, 0, "von_neumann").tolist() == [0, 0, 2, 4]
    assert len(torus.get_neighbors(1, 1, "moore", radius=1)) == 8
    assert Grid.from_matrix(data, boundary="dead").boundary == "fixed"

    print("\n=== 3. Vecini 1D ===")
    line_t = Grid.from_matrix([0, 1, 1, 0, 1], boundary="toroidal")
    line_f = Grid.from_matrix([0, 1, 1, 0, 1], boundary="fixed")
    print(line_t)
    print("c=0 toroidal:", line_t.get_neighbors(0, 0, "1d").tolist(),
          "| fixed:", line_f.get_neighbors(0, 0, "1d").tolist())
    assert line_t.is_1d
    assert line_t.get_neighbors(0, 0, "1d").tolist() == [1, 1]
    assert line_f.get_neighbors(0, 0, "1d").tolist() == [0, 1]
    assert line_f.get_neighbors(0, 4, "1d", radius=2).tolist() == [1, 0, 0, 0]

    print("\n=== 4. Distribuția stărilor din vecinătate ===")
    life = Grid.from_matrix([[1, 0, 1], [0, 1, 0], [1, 1, 0]], states=(0, 1))
    life_f = Grid.from_matrix([[1, 0, 1], [0, 1, 0], [1, 1, 0]], states=(0, 1), boundary="fixed")
    print("toroidal:", life.count_neighbor_states(0, 0), "| fixed:", life_f.count_neighbor_states(0, 0))
    assert life.count_neighbor_states(0, 0) == {0: 4, 1: 4}
    assert life_f.count_neighbor_states(0, 0) == {0: 7, 1: 1}

    print("\n=== 5. Clonare (double buffering) și egalitate ===")
    nxt = g.clone()
    assert nxt == g and nxt is not g
    nxt.set_cell(0, 0, 1)
    assert nxt != g and g.get_cell(0, 0) == 0 
    print("original:\n" + str(g) + "\nclonă modificată:\n" + str(nxt))

    print("\n=== 6. Etichete definite de utilizator ===")
    forest = Grid.from_matrix(
        [["tree", "fire"], ["empty", "tree"]], states=("empty", "tree", "fire"), boundary="fixed"
    )
    print(forest.render({"empty": " .", "tree": " T", "fire": " F"}))
    print(forest.count_neighbor_states(0, 0))
    assert forest.count_neighbor_states(0, 0) == {"empty": 6, "tree": 1, "fire": 1}

    print("\n=== 7. Validări ===")
    for bad, exc in [
        (lambda: g.set_cell(5, 5, 1), IndexError),
        (lambda: Grid(2, 2, states=(0, 1)).set_cell(0, 0, 7), ValueError),
        (lambda: Grid(2, 2, boundary="mirror"), ValueError),
        (lambda: Grid.from_matrix([[1, 2], [3]]), ValueError),
        (lambda: g.get_neighbors(0, 0, "hex"), ValueError),
    ]:
        try:
            bad()
            raise AssertionError("excepția așteptată nu a apărut")
        except exc as e:
            print(f"OK {type(e).__name__}: {e}")

    print("\nToate verificările au trecut.")

if __name__ == "__main__":
    _demo()
