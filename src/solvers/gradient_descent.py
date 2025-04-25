from typing import List, Tuple

import sympy as sp  # type: ignore

from config import GD_STEP
from utils import compute_fn

MAX_ITERATIONS = 10000


def gradient_descent(
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
    step = GD_STEP
    x1_symbol, x2_symbol = sp.symbols("x1, x2")
    M = list(x0)
    grad = [sp.diff(fn.expr, x1_symbol), sp.diff(fn.expr, x2_symbol)]
    if verbose:
        print(f"grad(M)={grad}")

    prev_f = -sp.oo
    for i in range(MAX_ITERATIONS):
        if verbose:
            print(f"iteration={i+1}: {M=}")

        M = [
            M[0] - step * grad[0].subs({x1_symbol: M[0], x2_symbol: M[1]}).evalf(),
            M[1] - step * grad[1].subs({x1_symbol: M[0], x2_symbol: M[1]}).evalf(),
        ]
        new_f = compute_fn(fn, M[0], M[1])
        if new_f >= prev_f and prev_f != -sp.oo:
            step /= 2
        if verbose:
            print(f"-new M: {M}, f(M)={compute_fn(fn, M[0], M[1])}")

        # check break condition
        if abs(new_f - prev_f) < EPS:
            break
        prev_f = new_f

    return M
    # raise ValueError("max iterations exceeded")
