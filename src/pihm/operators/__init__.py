r"""The operator dictionary: every operator sparse, every τ raw.

* :mod:`~pihm.operators.derivative`: :math:`\mathbb{G}`, the banded factors
  :math:`\tilde{\mathbb{B}}, \tilde{\mathbb{D}}` and the factorisation
  :math:`\mathbb{G} = R_0(2U_{\rm odd})\Lambda`.
* :mod:`~pihm.operators.multiplication`: :math:`\mathbb{M}_a` as an exact lift or square.
* :mod:`~pihm.operators.folds`: the product folds :math:`\mathbb{N}_1^{(p)}` and
  :math:`\mathbb{N}_x`, built by coefficient or nodal construction.
* :mod:`~pihm.operators.constraints`: the zero, collapse, ratio and datum rows.
* :mod:`~pihm.operators.tensor`: multi-axis (Kronecker-order) conventions.
"""

from pihm.operators.constraints import (
    CONSTRAINT_BUILDERS,
    ConstraintKind,
    collapse_row,
    datum_rows,
    ratio_rows,
    zero_row,
)
from pihm.operators.derivative import banded_factors, derivative_factorisation, derivative_matrix
from pihm.operators.folds import FoldBasis, FoldWeight, fold, fold_levels
from pihm.operators.multiplication import (
    CoefficientBasis,
    MultiplicationMode,
    embedding,
    multiplication,
    power_of_x,
    raw_product,
)
from pihm.operators.tensor import SparseArray, axis_sizes, embed_axis, kron_all

__all__ = [
    "CONSTRAINT_BUILDERS",
    "CoefficientBasis",
    "ConstraintKind",
    "FoldBasis",
    "FoldWeight",
    "MultiplicationMode",
    "SparseArray",
    "axis_sizes",
    "banded_factors",
    "collapse_row",
    "datum_rows",
    "derivative_factorisation",
    "derivative_matrix",
    "embed_axis",
    "embedding",
    "fold",
    "fold_levels",
    "kron_all",
    "multiplication",
    "power_of_x",
    "ratio_rows",
    "raw_product",
    "zero_row",
]
