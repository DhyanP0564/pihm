r"""Term intermediate representation of the descriptor's system rows.

The system rows of a descriptor residual, its equation, recurrences and padding, form one operator
from the state (a field register of :math:`n_f` qubits, then a Chebyshev register per axis) to a
register of the same shape: the row block that defines a state block sits at a field index of its
own (:func:`system_rows`). Its constraint rows are rank one, and are encoded apart. This module
writes the system rows as a linear combination of terms that a circuit implements directly
(:mod:`pihm.circuits.lcu`):

.. math::

   A_{\rm sys} = \sum_t c_t\, X^{x_t}Z^{z_t} \otimes \bigotimes_a F_{t,a},

a Pauli word on the field register and, on each axis, either a **band** (a truncated shift by
:math:`s` after a diagonal that is the identity, one :math:`Z_j`, or a reflection
:math:`R_m = I - 2|m\rangle\langle m|`) or a rank-one **outer** product
:math:`|u\rangle\langle\tau(x_s)|`, a source's coefficients against the feature vector at the
regular point.

Every band of the operators the descriptor is built from is affine in its column index but for a
few entries at its ends: :math:`\tilde{\mathbb{D}}`'s is :math:`2(c+1)` (so :math:`n` Z-string
terms, :math:`{\rm diag}(c) = \sum_j 2^j(I - Z_j)/2`), and :math:`\tilde{\mathbb{B}}`'s and the
multiplications' are constant; the normalisation at mode 0, the Hankel part of a product and the
Galerkin truncation give the ends (reflections). :func:`band_terms` finds that form from the
operator's matrix. Terms whose Chebyshev factors agree are merged across the row and column blocks
they join, so that the field register carries their pattern as one small matrix, expanded in Pauli
words: the recurrences' :math:`\tilde{\mathbb{D}}`, for example, is paid for once.

A first-order system's descriptor rows (:mod:`pihm.assemble.systems`) are its mass form,
:math:`I \otimes \tilde{\mathbb{D}} - \mathbb{A} \otimes \tilde{\mathbb{B}}` and the padding: three
groups of terms, :math:`\tilde{\mathbb{D}}`'s bands with the components' projector on the field
register, :math:`\tilde{\mathbb{B}}`'s with :math:`-\mathbb{A}`, and the identity with the padding.
The standard form's system rows put :math:`\mathbb{G}`, which is not banded, in place of
:math:`\tilde{\mathbb{D}}`, and the circuit composes their field-register matrices with
:math:`\mathbb{G}`'s own encoding.

**Matchings.** A coupling matrix :math:`\mathbb{A}`, which a Carleman lift makes large and
irregular, is not expanded in Pauli words but split into **matchings** (:func:`matching_terms`):
each has at most one entry per row and column, so it is a permutation of the field
register's basis states times a diagonal (:class:`Matching`), and there are as many as the
matrix's largest row or column degree.

:func:`system_ir` builds the representation from the problem and the operator library, not from
the assembled residual, so :meth:`SystemIR.to_sparse` against :func:`system_matrix` is an exact
check, at any :math:`n`.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray

from pihm import basis
from pihm.assemble import PADDING_WEIGHT, Residual, recurrence_name
from pihm.assemble.systems import coupling
from pihm.operators import SparseArray, banded_factors, multiplication
from pihm.problems.functions import polynomial_coefficients, source_coefficients
from pihm.problems.spec import Term as EquationTerm

__all__ = [
    "MAX_DEFECTS",
    "Band",
    "Diagonal",
    "DiagonalKind",
    "Matching",
    "Outer",
    "SystemIR",
    "Term",
    "band_terms",
    "field_terms",
    "matching_terms",
    "pauli_word",
    "system_ir",
    "system_matrix",
    "system_rows",
]

#: The diagonal factors a band may carry.
DiagonalKind: TypeAlias = Literal["identity", "z", "reflection"]

#: The most entries a band may leave off its affine fit before it is refused as unstructured.
MAX_DEFECTS: Final[int] = 24

#: Relative tolerance of a band's affine fit and of dropping a vanishing term.
_TOLERANCE: Final[float] = 1.0e-12


@dataclass(frozen=True)
class Diagonal:
    """A diagonal on one register: the identity, :math:`Z_j`, or :math:`R_m`."""

    kind: DiagonalKind
    index: int = 0

    def values(self, n_qubits: int) -> NDArray[np.float64]:
        """Return the diagonal's entries on ``n_qubits`` qubits."""
        k = np.arange(1 << n_qubits)
        if self.kind == "identity":
            return np.ones(k.size)
        if self.kind == "z":
            return np.asarray(1.0 - 2.0 * ((k >> self.index) & 1), dtype=np.float64)
        values = np.ones(k.size)
        values[self.index] = -1.0
        return values


@dataclass(frozen=True)
class Band:
    """One axis's factor of a banded term: the diagonal, then a truncated shift.

    Column :math:`c` of the input register goes to row :math:`c - s` of the output register,
    and vanishes where that row does not exist.
    """

    shift: int
    diagonal: Diagonal

    def matrix(self, n_in: int, n_out: int) -> sp.csr_array:
        """Return the factor as an ``(2**n_out, 2**n_in)`` matrix."""
        columns = np.arange(1 << n_in)
        rows = columns - self.shift
        keep = (rows >= 0) & (rows < (1 << n_out))
        values = self.diagonal.values(n_in)[keep]
        return sp.csr_array((values, (rows[keep], columns[keep])), shape=(1 << n_out, 1 << n_in))


@dataclass(frozen=True)
class Outer:
    r"""One axis's factor of a rank-one term, :math:`|u\rangle\langle\tau(x)|`.

    :math:`u` fills the output register; :math:`\tau(x)` is the raw feature vector
    (:func:`pihm.basis.tau`) at ``at``, which a circuit reads with the feature map.
    """

    left: tuple[float, ...]
    at: float

    def matrix(self, n_in: int, n_out: int) -> sp.csr_array:
        """Return the factor as an ``(2**n_out, 2**n_in)`` matrix."""
        u = np.asarray(self.left)
        if u.size != 1 << n_out:
            raise ValueError("an outer factor's coefficients must fill its output register")
        return sp.csr_array(np.outer(u, np.real(basis.tau(self.at, n_in))))


@dataclass(frozen=True)
class Matching:
    r"""A field-register matrix with at most one entry in each row and column.

    Column ``columns[i]`` goes to row ``rows[i]`` with entry ``values[i]``, the entries divided by
    their largest magnitude (a term's coefficient carries it), so that each lies in
    :math:`[-1, 1]`. It is a permutation of the basis states (:meth:`permutation`) followed by a
    diagonal on the rows, zero on every row it does not reach.
    """

    columns: tuple[int, ...]
    rows: tuple[int, ...]
    values: tuple[float, ...]

    def matrix(self, n_qubits: int) -> sp.csr_array:
        """Return the matching as a ``(2**n_qubits, 2**n_qubits)`` matrix."""
        size = 1 << n_qubits
        return sp.csr_array((self.values, (self.rows, self.columns)), shape=(size, size))

    def is_permutation(self, n_qubits: int) -> bool:
        """Whether the matching reaches every row with entry one: a permutation, unitary."""
        return len(self.rows) == 1 << n_qubits and all(v == 1.0 for v in self.values)

    def permutation(self, n_qubits: int) -> tuple[int, ...]:
        """Return a permutation of the basis states that sends each column to its row.

        Every state the matching leaves alone is fixed, and each chain of pairs (a column whose
        row is another pair's column) is closed into its own cycle, so that the permutation has
        as few transpositions as the pairs allow.
        """
        images = list(range(1 << n_qubits))
        following = dict(zip(self.columns, self.rows, strict=True))
        for column, row in following.items():
            images[column] = row
        for head in sorted(set(self.columns) - set(self.rows)):
            end = head
            while end in following:
                end = following[end]
            images[end] = head
        return tuple(images)


@dataclass(frozen=True)
class Term:
    r"""One term, :math:`c\,X^xZ^z \otimes \bigotimes_a F_a`, or with a matching for the Pauli word.

    A term with a :class:`Matching` has ``x_mask`` and ``z_mask`` zero, and the matching acts on the
    field register in their place.
    """

    coefficient: float
    x_mask: int
    z_mask: int
    factors: tuple[Band | Outer, ...]
    matching: Matching | None = None

    def field_matrix(self, n_qubits: int) -> sp.csr_array:
        """Return the term's matrix on the field register."""
        if self.matching is not None:
            return self.matching.matrix(n_qubits)
        return pauli_word(self.x_mask, self.z_mask, n_qubits)


@dataclass(frozen=True)
class SystemIR:
    """The descriptor's system rows as a sum of terms.

    Attributes
    ----------
    terms
        The terms.
    field_qubits
        The field register's qubits :math:`n_f`.
    qubits_in, qubits_out
        Each axis's register on the input and the output side (the output has one qubit more on
        a lifted axis).
    """

    terms: tuple[Term, ...]
    field_qubits: int
    qubits_in: tuple[int, ...]
    qubits_out: tuple[int, ...]

    @property
    def alpha(self) -> float:
        r"""The linear combination's subnormalisation, :math:`\sum_t|c_t|`."""
        return float(sum(abs(t.coefficient) for t in self.terms))

    def to_sparse(self) -> sp.csr_array:
        """Return the operator the terms sum to."""
        field = 1 << self.field_qubits
        rows = field * int(np.prod([1 << n for n in self.qubits_out]))
        columns = field * int(np.prod([1 << n for n in self.qubits_in]))
        total = sp.csr_array((rows, columns))
        for term in self.terms:
            parts: list[SparseArray] = [term.field_matrix(self.field_qubits)]
            for factor, n_in, n_out in zip(
                term.factors, self.qubits_in, self.qubits_out, strict=True
            ):
                parts.append(factor.matrix(n_in, n_out))
            product = parts[0]
            for part in parts[1:]:
                product = sp.kron(product, part, format="csr")
            total = sp.csr_array(total + term.coefficient * product)
        return total


def pauli_word(x_mask: int, z_mask: int, n_qubits: int) -> sp.csr_array:
    r"""Return :math:`X^xZ^z`, :math:`|b\rangle\mapsto(-1)^{z\cdot b}|b\oplus x\rangle`."""
    b = np.arange(1 << n_qubits)
    signs = 1.0 - 2.0 * (np.bitwise_count(b & z_mask) & 1)
    return sp.csr_array((signs, (b ^ x_mask, b)), shape=(1 << n_qubits, 1 << n_qubits))


def _pauli_expansion(F: NDArray[np.float64]) -> dict[tuple[int, int], float]:
    r"""Return the coefficients :math:`f_{xz}` of :math:`F = \sum f_{xz}X^xZ^z` (real F)."""
    size = F.shape[0]
    expansion: dict[tuple[int, int], float] = {}
    b = np.arange(size)
    scale = float(np.max(np.abs(F))) if F.size else 0.0
    for x in range(size):
        column_pairs = F[b ^ x, b]
        for z in range(size):
            signs = 1.0 - 2.0 * (np.bitwise_count(b & z) & 1)
            value = float(np.sum(signs * column_pairs)) / size
            if abs(value) > _TOLERANCE * max(scale, 1.0):
                expansion[(x, z)] = value
    return expansion


def field_terms(F: NDArray[np.float64], factors: tuple[Band | Outer, ...] = ()) -> list[Term]:
    r"""Return :math:`F = \sum f_{xz}X^xZ^z` on the field register as terms with ``factors``.

    With no factors the terms act on the field register alone, a block encoding of :math:`F`
    once compiled (:func:`pihm.circuits.lcu.compile_system`).
    """
    return [Term(value, x, z, factors) for (x, z), value in _pauli_expansion(F).items()]


def _matchings(F: NDArray[np.float64]) -> list[list[tuple[int, int, float]]]:
    """Return ``F``'s non-zeros as matchings, a greedy edge colouring of rows against columns.

    The entries are coloured in row order, each the least colour neither its row nor its column
    has, so that no two entries of a colour share a row or a column.
    """
    scale = float(np.max(np.abs(F))) if F.size else 0.0
    rows, columns = np.nonzero(np.abs(F) > _TOLERANCE * max(scale, 1.0))
    taken_rows: dict[int, set[int]] = {}
    taken_columns: dict[int, set[int]] = {}
    colours: list[list[tuple[int, int, float]]] = []
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        taken = taken_rows.setdefault(row, set()) | taken_columns.setdefault(column, set())
        colour = next(c for c in range(len(taken) + 1) if c not in taken)
        if colour == len(colours):
            colours.append([])
        colours[colour].append((row, column, float(F[row, column])))
        taken_rows[row].add(colour)
        taken_columns[column].add(colour)
    return colours


def matching_terms(F: NDArray[np.float64], factors: tuple[Band | Outer, ...] = ()) -> list[Term]:
    r"""Return :math:`F` on the field register as a sum of matchings with ``factors``.

    Each term's coefficient is its matching's largest entry in magnitude, so the terms'
    :math:`\alpha` is the sum of those (module documentation); where a matching's entries share
    one value, its sign too, and the matching's entries are one.
    """
    terms = []
    for entries in _matchings(F):
        rows, columns, values = zip(*entries, strict=True)
        largest = max(abs(v) for v in values)
        sign = np.sign(values[0]) if len(set(values)) == 1 else 1.0
        scaled = tuple(v / (sign * largest) for v in values)
        terms.append(Term(float(sign * largest), 0, 0, factors, Matching(columns, rows, scaled)))
    return terms


def band_terms(matrix: SparseArray, n_in: int, n_out: int) -> list[tuple[float, Band]]:
    r"""Return the banded ``matrix`` as :math:`\sum c\,\text{shift}^s\,{\rm diag}` terms.

    Each band's entries, by column, are fitted as affine in the column index; the entries off the
    fit (at most :data:`MAX_DEFECTS`) become reflections. Columns the truncated shift discards are
    free, and are fitted to whatever keeps the form shortest.

    Parameters
    ----------
    matrix
        ``(2**n_out, 2**n_in)``.
    n_in, n_out
        The input and output registers' qubits.

    Returns
    -------
    list of (float, Band)
        The terms, the identity first in each band.

    Raises
    ------
    ValueError
        If a band has more than :data:`MAX_DEFECTS` entries off its affine fit.
    """
    M = sp.coo_array(matrix)
    N_in, N_out = 1 << n_in, 1 << n_out
    if M.shape != (N_out, N_in):
        raise ValueError(f"a {M.shape} matrix does not map {n_in} qubits to {n_out}")
    scale = float(np.max(np.abs(M.data))) if M.nnz else 0.0
    terms: list[tuple[float, Band]] = []
    for s in sorted(set((M.col - M.row).tolist())):
        on_band = M.col - M.row == s
        columns = np.arange(N_in)
        valid = (columns - s >= 0) & (columns - s < N_out)
        w = np.zeros(N_in)
        w[M.col[on_band]] = M.data[on_band]
        c, v = columns[valid], w[valid]
        a, b, defects = _affine_with_defects(c.astype(np.float64), v, scale)
        identity = a + b * (N_in - 1) / 2.0 + sum(d for _, d in defects) / 2.0
        if abs(identity) > _TOLERANCE * scale:
            terms.append((identity, Band(s, Diagonal("identity"))))
        if abs(b) > _TOLERANCE * scale:
            terms.extend((-b * 2.0 ** (j - 1), Band(s, Diagonal("z", j))) for j in range(n_in))
        terms.extend((-d / 2.0, Band(s, Diagonal("reflection", m))) for m, d in defects)
    return terms


def _affine_with_defects(
    c: NDArray[np.float64], v: NDArray[np.float64], scale: float
) -> tuple[float, float, list[tuple[int, float]]]:
    """Return ``(a, b, [(m, v_m - a - b m)])``: the affine fit and the entries off it."""
    if c.size <= 2:
        # Two points always lie on a line.
        slope = float((v[1] - v[0]) / (c[1] - c[0])) if c.size == 2 else 0.0
        intercept = float(v[0] - slope * c[0]) if c.size else 0.0
        return intercept, slope, []
    tolerance = _TOLERANCE * max(scale, 1.0)
    slope = float(np.median(np.diff(v) / np.diff(c)))
    intercept = float(np.median(v - slope * c))
    for _ in range(3):
        bulk = np.abs(v - intercept - slope * c) <= 1e3 * tolerance
        if np.count_nonzero(bulk) >= 2:
            slope, intercept = (float(x) for x in np.polyfit(c[bulk], v[bulk], 1))
    residual = v - intercept - slope * c
    off = np.flatnonzero(np.abs(residual) > tolerance)
    if off.size > MAX_DEFECTS:
        raise ValueError(
            f"a band has {off.size} entries off its affine fit (at most {MAX_DEFECTS})"
        )
    snapped = [(int(c[i]), float(residual[i])) for i in off]
    return intercept, slope, snapped


def system_rows(residual: Residual) -> tuple[tuple[int, str], ...]:
    r"""Return the field index at which each system row block sits, with the block's name.

    A recurrence sits at its lower block's index while that is free, so that the dominant
    :math:`\tilde{\mathbb{D}}` is diagonal on the field register, and otherwise at its higher
    block's; the equation takes the lowest index left, and a padding row block its own.
    """
    index = {d: i for i, d in enumerate(residual.derivatives) if d is not None}
    taken: set[int] = set()
    placed: list[tuple[int, str]] = []
    for lower, higher, name in _recurrences(residual):
        at = index[lower] if index[lower] not in taken else index[higher]
        taken.add(at)
        placed.append((at, name))
    free = [i for i, d in enumerate(residual.derivatives) if d is not None and i not in taken]
    placed.insert(0, (min(free), "equation"))
    for i, d in enumerate(residual.derivatives):
        if d is None:
            placed.append((i, f"padding block {i}"))
    return tuple(placed)


def _recurrences(
    residual: Residual,
) -> list[tuple[tuple[int, ...], tuple[int, ...], str]]:
    """Return each recurrence's lower and higher derivative and its name, in assembly order."""
    axes = residual.problem.fields[0].axes
    carried = [d for d in residual.derivatives if d is not None]
    found = []
    for a, axis in enumerate(axes):
        order = max(d[a] for d in carried)
        for j in range(order):
            lower = tuple(j if b == a else 0 for b in range(len(axes)))
            higher = tuple(j + 1 if b == a else 0 for b in range(len(axes)))
            found.append((lower, higher, recurrence_name(j, axis)))
    return found


def system_matrix(residual: Residual) -> sp.csr_array:
    """Return the residual's system row blocks at their field indices (:func:`system_rows`).

    Row blocks on a register narrower than the widest (the recurrences and the padding, when the
    equation is lifted) take the lower rows of it. A system's equation rows are its components',
    in order, and its padding follows. This is the assembled operator the IR is checked against.
    """
    blocks = {block.name: block.matrix for block in residual.blocks}
    if residual.is_system:
        padding = [block.matrix for block in residual.blocks if block.kind == "padding"]
        return sp.csr_array(sp.vstack([blocks["equation"], *padding], format="csr"))
    widest = max(blocks["equation"].shape[0], residual.dimension // len(residual.derivatives))
    field = len(residual.derivatives)
    parts = []
    for at, name in sorted(system_rows(residual)):
        matrix = blocks[name]
        padded = sp.vstack([matrix, sp.csr_array((widest - matrix.shape[0], matrix.shape[1]))])
        parts.append((at, padded))
    rows: list[SparseArray] = [sp.csr_array((widest, residual.dimension))] * field
    for at, padded in parts:
        rows[at] = padded
    return sp.csr_array(sp.vstack(rows, format="csr"))


def _rows_on(matrix: SparseArray, n_out: int) -> sp.csr_array:
    """Return ``matrix`` with zero rows below it, to fill a register of ``n_out`` qubits."""
    M = sp.csr_array(matrix)
    missing = (1 << n_out) - M.shape[0]
    return M if missing == 0 else sp.csr_array(sp.vstack([M, sp.csr_array((missing, M.shape[1]))]))


def _source_factor(term: EquationTerm, residual: Residual, ns_out: Sequence[int]) -> Outer:
    r"""Return a source's rank-one factor :math:`|\hat r\rangle\langle\tau(x_s)|` (one axis)."""
    problem = residual.problem
    (n_out,) = ns_out
    if term.source is not None:
        target = source_coefficients(term.source, [a.name for a in problem.axes], [n_out])
    elif term.multiplier is not None:
        target = polynomial_coefficients(term.multiplier, n_out)
    else:
        target = np.zeros(1 << n_out)
        target[0] = np.sqrt(float(1 << n_out))
    if np.iscomplexobj(target) and np.any(np.imag(target) != 0):
        raise NotImplementedError("the term IR takes a real source")
    return Outer(tuple(float(x) for x in np.real(target)), float(problem.regular.at[0]))


def system_ir(residual: Residual) -> SystemIR:
    """Return a linear descriptor residual's system rows as terms (the module documentation).

    It is built from the problem and the operator library, with the residual's layout, lifts and
    block weights, but not from its assembled matrices, so :meth:`SystemIR.to_sparse` against
    :func:`system_matrix` checks the assembly.

    Raises
    ------
    ValueError
        For a residual that is not a descriptor residual, or is a doubled space.
    NotImplementedError
        For a source on several axes, or a complex one.
    """
    if residual.regime != "descriptor" or residual.nonlinearity == "doubled":
        raise ValueError("the term IR covers descriptor residuals, not a doubled space")
    if residual.is_system:
        return _mass_form_ir(residual)
    problem = residual.problem
    axes = problem.fields[0].axes
    ns = residual.qubits
    ns_out = tuple(n + int(lift) for n, lift in zip(ns, residual.lifted, strict=True))
    index = {d: i for i, d in enumerate(residual.derivatives) if d is not None}
    weights = residual.block_weights
    row_of = {name: at for at, name in system_rows(residual)}
    size = len(residual.derivatives)
    identity: list[tuple[float, Band | Outer]] = [(1.0, Band(0, Diagonal("identity")))]
    groups: dict[tuple[Band | Outer, ...], NDArray[np.float64]] = {}

    def add(
        row: int, column: int, per_axis: Sequence[Sequence[tuple[float, Band | Outer]]]
    ) -> None:
        for combination in itertools.product(*per_axis):
            value = weights[column] * float(np.prod([c for c, _ in combination]))
            key = tuple(factor for _, factor in combination)
            groups.setdefault(key, np.zeros((size, size)))[row, column] += value

    for lower, higher, name in _recurrences(residual):
        a = next(i for i, j in enumerate(higher) if j)
        B, D = banded_factors(ns[a])
        for column, operator in ((index[lower], -D), (index[higher], B)):
            factors = [
                band_terms(_rows_on(operator, ns_out[b]), ns[b], ns_out[b]) if b == a else identity
                for b in range(len(axes))
            ]
            add(row_of[name], column, factors)

    (equation,) = problem.equations
    for term in equation.terms:
        if not term.factors:
            if len(axes) != 1:
                raise NotImplementedError("the term IR takes sources on one axis")
            outer = _source_factor(term, residual, ns_out)
            scale = term.coefficient / residual.regular_value
            add(row_of["equation"], index[(0,)], [[(scale, outer)]])
            continue
        (factor,) = term.factors
        factors = []
        for b, (axis, n) in enumerate(zip(axes, ns, strict=True)):
            on_axis = term.multiplier is not None and term.multiplier.axis == axis
            polynomial = term.multiplier if on_axis else None
            coefficients = (1.0,) if polynomial is None else polynomial.coefficients
            if residual.lifted[b]:
                operator = multiplication(coefficients, n, mode="lift")
                factors.append(band_terms(operator, n, ns_out[b]))
            elif polynomial is not None and not polynomial.is_constant:
                operator = multiplication(coefficients, n, mode=residual.mult)
                factors.append(band_terms(operator, n, ns_out[b]))
            else:
                factors.append([(coefficients[0], Band(0, Diagonal("identity")))])
        factors[0] = [(term.coefficient * c, f) for c, f in factors[0]]
        add(row_of["equation"], index[factor.derivative], factors)

    for i, derivative in enumerate(residual.derivatives):
        if derivative is None:
            padding: list[tuple[float, Band | Outer]] = [
                (PADDING_WEIGHT, Band(0, Diagonal("identity")))
            ]
            add(i, i, [padding] + [identity] * (len(axes) - 1))

    terms = [
        Term(value, x, z, key)
        for key, F in groups.items()
        for (x, z), value in _pauli_expansion(F).items()
    ]
    return SystemIR(tuple(terms), residual.field_qubits, ns, ns_out)


def _mass_form_ir(residual: Residual) -> SystemIR:
    r"""Return a system's mass form as terms (module documentation).

    :math:`\Pi_C \otimes \tilde{\mathbb{D}} - \mathbb{A} \otimes \tilde{\mathbb{B}} +
    c_{\rm pad}\Pi_{\rm pad} \otimes I`: the bands of :math:`\tilde{\mathbb{D}}` and the
    padding's identity with their field-register matrices in Pauli words, and each band of
    :math:`\tilde{\mathbb{B}}` with :math:`-\mathbb{A}`'s matchings, which the bands share.
    """
    (n,) = residual.qubits
    size = len(residual.derivatives)
    components = np.array([m is not None for m in residual.monomials], dtype=np.float64)
    A = coupling(residual.problem, residual.carleman_order).toarray()
    padded = np.zeros((size, size))
    padded[: A.shape[0], : A.shape[1]] = A
    B, D = banded_factors(n)
    groups: dict[tuple[Band | Outer, ...], NDArray[np.float64]] = {}
    for coefficient, band in band_terms(D, n, n):
        groups[(band,)] = coefficient * np.diag(components)
    padding = PADDING_WEIGHT * np.diag(1.0 - components)
    if padding.any():
        key: tuple[Band | Outer, ...] = (Band(0, Diagonal("identity")),)
        groups[key] = groups.get(key, np.zeros((size, size))) + padding
    terms = [term for key, F in groups.items() for term in field_terms(F, key)]
    matchings = matching_terms(-padded)
    for coefficient, band in band_terms(B, n, n):
        terms.extend(
            Term(coefficient * m.coefficient, 0, 0, (band,), m.matching) for m in matchings
        )
    return SystemIR(tuple(terms), residual.field_qubits, (n,), (n,))
