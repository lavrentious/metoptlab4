from config import EPS, STARTING_POINT, fn
from solvers.coordinate_descent import coordinate_descent
from solvers.fastest_descent import fastest_descent
from solvers.gradient_descent import gradient_descent
from utils import compute_fn

VERBOSE = False


def run() -> None:
    print("1. Метод покоординатного спуска")
    ans = coordinate_descent(fn, STARTING_POINT, EPS, VERBOSE)
    print("===")
    print(f"ANS={ans}, f(M)={compute_fn(fn, ans[0], ans[1])}\n")

    print("2. Метод градиентного спуска")
    ans = gradient_descent(fn, STARTING_POINT, EPS, VERBOSE)
    print("===")
    print(f"ANS={ans}, f(M)={compute_fn(fn, ans[0], ans[1])}")

    print("3. Метод наискорейшего спуска")
    ans = fastest_descent(fn, STARTING_POINT, EPS, VERBOSE)
    print("===")
    print(f"ANS={ans}, f(M)={compute_fn(fn, ans[0], ans[1])}")


if __name__ == "__main__":
    run()
