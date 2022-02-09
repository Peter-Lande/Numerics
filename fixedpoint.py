import numpy as np
import argparse


def fixed_point(function, initial, error):
    next_guess = (function(initial))
    iteration_error = np.abs((next_guess-initial)/initial)
    print("%11.6f %11.6f %11.6f" %
          (initial, next_guess, iteration_error))
    if iteration_error < error:
        return next_guess
    fixed_point(function, initial, error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solve for roots of functions using Newton-Raphson method.', prog='Newton-Raphson')
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
        p = fixed_point(f, args.initial, args.tolerance)
        print("Found a root at p= %11.6f" % p)
    except ValueError:
        print("No root in given bound.")
