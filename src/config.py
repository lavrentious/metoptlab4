import sympy as sp  # type: ignore

PRECISION = 32


STARTING_POINT = (sp.Float("2", PRECISION), sp.Float("-2", PRECISION))

EPS = sp.Float("0.0001", PRECISION)
GD_STEP = sp.Float("0.01", PRECISION)  # шаг для метода градиентного спуска
QA_STEP = sp.Float("0.0001", PRECISION)  # шаг для метода квадратичной аппроксимации


x1, x2 = sp.symbols("x1, x2")
fn = sp.Lambda((x1, x2), 3 * x1**2 + 5 * x2**2 + 2 * x1 * x2 - 7 * x1 + x2 - 4)

# def fn(x1: Decimal, x2: Decimal) -> Decimal:
#     return 3 * x1**2 + 5 * x2**2 + 2 * x1 * x2 - 7 * x1 + x2 - 4


# def dfx1(x1: Decimal, x2: Decimal) -> Decimal:
#     return 6 * x1 + 2 * x2 - 7


# def dfx2(x1: Decimal, x2: Decimal) -> Decimal:
#     return 10 * x2 + 2 * x1 + 1
