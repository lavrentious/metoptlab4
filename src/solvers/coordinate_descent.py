from typing import List, Tuple

import sympy as sp  # type: ignore

from config import PRECISION
from utils import compute_fn

MAX_ITERATIONS = 100


def coordinate_descent(
    fn: sp.Lambda,
    x0: Tuple[sp.Float, sp.Float],
    EPS: sp.Float = sp.Float("0.0001"),
    verbose: bool = False,
) -> List[sp.Float]:
    """
    fn - multivariable function
    x0 - starting point vector
    dfxs - list of corresponding partial derivatives
    """
    x1_symbol, x2_symbol = sp.symbols("x1, x2")
    M = list(x0)

    prev_M = [-sp.oo, -sp.oo]
    for i in range(MAX_ITERATIONS):
        if verbose:
            print(f"iteration={i+1}: {M=}")

        # x1=?, x2=M[1]
        if verbose:
            print(f"-fixating x2={M[1]}")
        dfx1 = sp.diff(fn.expr, x1_symbol).subs({x2_symbol: M[1]})
        x1 = sp.solve(dfx1, x1_symbol)
        assert len(x1) == 1
        M[0] = x1[0].evalf(PRECISION)
        if verbose:
            print(f"--x1={M[0]}")

        # x1=M[0], x2=?
        if verbose:
            print(f"-fixating x1={M[0]}")
        dfx2 = sp.diff(fn.expr, x2_symbol).subs({x1_symbol: M[0]})
        x2 = sp.solve(dfx2, x2_symbol)
        M[1] = x2[0].evalf(PRECISION)
        if verbose:
            print(f"--x2={M[1]}")

        if verbose:
            print(f"new M: {M}, f(M)={compute_fn(fn, M[0], M[1])}")

        # check break condition: M(i-1) - M(i) < EPS
        if (M[0] - prev_M[0]) ** 2 + (M[1] - prev_M[1]) ** 2 < EPS**2:
            break
        prev_M = M.copy()

    return M
    # raise ValueError("max iterations exceeded")
