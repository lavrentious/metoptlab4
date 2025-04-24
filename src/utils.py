import sympy as sp  # type: ignore

from config import PRECISION

x1_symbol, x2_symbol = sp.symbols("x1, x2")


def compute_fn(fn: sp.Lambda, x1: sp.Float, x2: sp.Float) -> sp.Float:
    return fn.expr.subs({x1_symbol: x1, x2_symbol: x2}).evalf(PRECISION)
