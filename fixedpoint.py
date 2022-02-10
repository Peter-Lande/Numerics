import numpy as np
import argparse
import sys


def fixed_point(function, initial, error, iteration=0, previous_error=sys.maxsize):
    next_guess = (function(initial))
    iteration_error = np.abs((next_guess-initial))
    iteration += 1
    print("%13d %13.9f %13.9f %13.9f" %
          (iteration, initial, next_guess, iteration_error))
    if iteration_error < error:
        return next_guess
    if iteration_error >= previous_error:
        raise ValueError("Solution diverges.")
    if not (type(next_guess) == np.float64):
        raise ValueError("Solution is indeterminate.")
    return fixed_point(function, next_guess, error, iteration, iteration_error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solve for roots of functions using Newton-Raphson method.', prog='Fixed-Point')
    parser.add_argument('function', metavar='f(x)',
                        help='Function to be solved.')
    parser.add_argument('initial', type=float,
                        help='Initial guess of the root.')
    parser.add_argument('tolerance', type=float,
                        help='Tolerance level for Newton Raphson method.')
    args = parser.parse_args()
    try:
        def f(x): return eval(args.function)
    except NameError:
        print("f(x) does not follow python syntax.")
    try:
        print("%13s %13s %13s %13s" % ("Iteration", "p", "f(p)", "tolerance"))
        p = fixed_point(f, args.initial, args.tolerance)
        print("Found a root at p= %13.9f" % p)
    except ValueError as error:
        print(str(error))
