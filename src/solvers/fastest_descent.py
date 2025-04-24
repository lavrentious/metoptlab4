from typing import List, Tuple

import sympy as sp  # type: ignore

from utils import compute_fn

MAX_ITERATIONS = 100


def fastest_descent(
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
    grad = [sp.diff(fn.expr, x1_symbol), sp.diff(fn.expr, x2_symbol)]
    if verbose:
        print(f"grad(M)={grad}")

    -sp.oo
    h = sp.symbols("h")
    for i in range(MAX_ITERATIONS):
        if verbose:
            print(f"iteration={i+1}: {M=}")

        g = [
            grad[0].subs({x1_symbol: M[0], x2_symbol: M[1]}).evalf(),
            grad[1].subs({x1_symbol: M[0], x2_symbol: M[1]}).evalf(),
        ]
        g_len = (g[0] ** 2 + g[1] ** 2) ** 0.5
        # check break condition
        if g_len < EPS:
            break
        normalized_grad = [
            g[0] / g_len,
            g[1] / g_len,
        ]
        # minimize f = f(x + h * normalized_grad) by lambda
        x1_new = M[0] - h * normalized_grad[0]
        x2_new = M[1] - h * normalized_grad[1]
        if verbose:
            print(f"x1_new={x1_new} ; x2_new={x2_new}")
        fh = fn.expr.subs({x1_symbol: x1_new, x2_symbol: x2_new})
        if verbose:
            print(f"-grad={g}")
            print(f"-{normalized_grad=}")
            print(f"-fh: {sp.collect(sp.expand(fh), h)}")
        h_min = sp.solve(sp.diff(fh, h), h)[0]
        if verbose:
            print(f"-{h_min=}")

        M[0] = M[0] - h_min * normalized_grad[0]
        M[1] = M[1] - h_min * normalized_grad[1]

        if verbose:
            print(f"-new M: {M}, f(M)={compute_fn(fn, M[0], M[1])}")

        # prev_f = new_f

    return M
    # raise ValueError("max iterations exceeded")
