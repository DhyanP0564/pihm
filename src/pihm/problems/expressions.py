r"""A small, safe expression language for the known functions of a problem.

Problems are declarative data, so the functions they mention (exact
solutions, source terms, boundary data and readout weights) are written as text, such as
``"0.5*(1 + x)*exp(-2*x)"``, rather than as Python callables. Text serialises to YAML, hashes
stably and reads the same in a record as in the paper.

An expression is parsed once and checked against a whitelist before it is compiled. It may use

* the problem's axis variables (for example ``x``, or ``t`` and ``x``);
* numbers, and the constants ``pi`` and ``e``;
* ``+``, ``-``, ``*``, ``/`` and ``**``;
* the functions in :data:`FUNCTIONS`, called with positional arguments only.

Anything else (attribute access, subscripts, keywords, comparisons, lambdas, names that are not
listed) is refused with a :class:`ValueError`, so evaluating an expression cannot run arbitrary
code.

Examples
--------
>>> f = compile_expression("0.5*(1 + x)*exp(-2*x)", ["x"])
>>> float(f(0.0))
0.5
>>> compile_expression("__import__('os')", ["x"])
Traceback (most recent call last):
    ...
ValueError: '__import__' is not an allowed name in expression "__import__('os')"
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
from numpy.polynomial import legendre as nplegendre
from numpy.typing import ArrayLike, NDArray
from scipy import special

__all__ = ["CONSTANTS", "FUNCTIONS", "Compiled", "compile_expression", "expression_variables"]

#: A compiled expression, called with one array (or scalar) per variable, in order.
Compiled = Callable[..., NDArray[Any]]


def _legendre(degree: ArrayLike, x: Any) -> NDArray[Any]:
    r"""Return the Legendre polynomial :math:`P_l(x)`."""
    return np.asarray(special.eval_legendre(_whole(degree), x))


def _legendre_derivative(degree: ArrayLike, x: Any) -> NDArray[Any]:
    r"""Return :math:`P_l'(x)`, exactly, from the Legendre series of :math:`P_l`."""
    series = np.zeros(_whole(degree) + 1)
    series[-1] = 1.0
    return np.asarray(nplegendre.legval(x, nplegendre.legder(series)))


def _associated_legendre(order: ArrayLike, degree: ArrayLike, x: Any) -> NDArray[Any]:
    r"""Return :math:`P_l^m(x)` with the Condon–Shortley phase (as :func:`scipy.special.lpmv`)."""
    return np.asarray(special.lpmv(_whole(order), _whole(degree), x))


def _whole(value: ArrayLike) -> int:
    """Return ``value`` as a non-negative integer, refusing anything else."""
    number = float(np.asarray(value))
    if not number.is_integer() or number < 0:
        raise ValueError(f"expected a non-negative integer order or degree, got {number}")
    return int(number)


#: The functions an expression may call. Elementary functions act elementwise (and accept
#: complex arguments); the Legendre functions take their integer order and degree first.
FUNCTIONS: Final[Mapping[str, Callable[..., Any]]] = {
    "abs": np.abs,
    "arctan": np.arctan,
    "assoc_legendre": _associated_legendre,
    "cos": np.cos,
    "cosh": np.cosh,
    "dlegendre": _legendre_derivative,
    "exp": np.exp,
    "legendre": _legendre,
    "log": np.log,
    "sin": np.sin,
    "sinh": np.sinh,
    "sqrt": np.sqrt,
    "tan": np.tan,
    "tanh": np.tanh,
}

#: Named constants an expression may use.
CONSTANTS: Final[Mapping[str, float]] = {"pi": float(np.pi), "e": float(np.e)}

_OPERATORS: Final = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)


def _check(node: ast.AST, variables: frozenset[str], text: str) -> None:
    """Refuse any node outside the whitelist (recursively)."""
    if isinstance(node, ast.Expression):
        _check(node.body, variables, text)
    elif isinstance(node, ast.BinOp):
        if not isinstance(node.op, _OPERATORS):
            raise ValueError(f"operator {type(node.op).__name__} is not allowed in {text!r}")
        _check(node.left, variables, text)
        _check(node.right, variables, text)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, _OPERATORS):
            raise ValueError(f"operator {type(node.op).__name__} is not allowed in {text!r}")
        _check(node.operand, variables, text)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            raise ValueError(f"{name!r} is not an allowed name in expression {text!r}")
        if node.keywords:
            raise ValueError(f"keyword arguments are not allowed in expression {text!r}")
        for argument in node.args:
            _check(argument, variables, text)
    elif isinstance(node, ast.Name):
        if node.id not in variables and node.id not in CONSTANTS:
            raise ValueError(f"{node.id!r} is not an allowed name in expression {text!r}")
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numeric constants are allowed in expression {text!r}")
    else:
        raise ValueError(f"{type(node).__name__} is not allowed in expression {text!r}")


def _parse(text: str) -> ast.Expression:
    """Parse ``text`` as a single expression."""
    try:
        return ast.parse(text.strip(), mode="eval")
    except SyntaxError as error:
        raise ValueError(f"cannot parse expression {text!r}: {error.msg}") from None


def expression_variables(text: str) -> frozenset[str]:
    """Return the free variables of ``text``: its names that are neither functions nor constants.

    Examples
    --------
    >>> sorted(expression_variables("exp(-t)*sin(2*pi*x)"))
    ['t', 'x']
    """
    names = {node.id for node in ast.walk(_parse(text)) if isinstance(node, ast.Name)}
    return frozenset(names - set(FUNCTIONS) - set(CONSTANTS))


def compile_expression(text: str, variables: Sequence[str]) -> Compiled:
    """Compile ``text`` into a vectorised function of ``variables``.

    Parameters
    ----------
    text
        The expression, in the language described in the module documentation.
    variables
        The names of the arguments, in call order (normally the problem's axis names). An
        expression need not use all of them.

    Returns
    -------
    Callable
        ``f(*arrays)`` returning an array with the broadcast shape of its arguments, so that a
        constant expression still returns one value per point.

    Raises
    ------
    ValueError
        If the expression does not parse, or uses anything outside the whitelist.
    """
    names = frozenset(variables)
    if len(names) != len(variables):
        raise ValueError(f"variable names must be distinct, got {list(variables)}")
    clash = names & (set(FUNCTIONS) | set(CONSTANTS))
    if clash:
        raise ValueError(f"variable names {sorted(clash)} clash with built-in names")
    tree = _parse(text)
    _check(tree, names, text)
    code = compile(tree, filename="<pihm expression>", mode="eval")
    order = tuple(variables)

    def evaluate(*values: ArrayLike) -> NDArray[Any]:
        if len(values) != len(order):
            raise TypeError(f"expected {len(order)} argument(s) {order}, got {len(values)}")
        arrays = [np.asarray(value) for value in values]
        namespace: dict[str, Any] = {**FUNCTIONS, **CONSTANTS}
        namespace.update(zip(order, arrays, strict=True))
        # Safe: the tree was checked against the whitelist and builtins are removed.
        result = np.asarray(eval(code, {"__builtins__": {}}, namespace))
        shape = np.broadcast_shapes(*(array.shape for array in arrays)) if arrays else ()
        return np.broadcast_to(result, np.broadcast_shapes(shape, result.shape)).copy()

    evaluate.__doc__ = f"Evaluate {text!r} at ({', '.join(order)})."
    return evaluate
