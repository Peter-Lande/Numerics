import numpy as np
import argparse


def newton_raphson(function, derivative, initial, error):
    next_guess = initial - (function(initial))/(derivative(initial))
    if np.abs((next_guess-initial)/initial) < error:
        return next_guess
    newton_raphson(function, derivative, initial, error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solve for roots of functions using Newton-Raphson method.', prog='Newton-Raphson')
    parser.add_argument('function', metavar='f(x)',
                        help='Function to be solved.')
    parser.add_argument('derivative', metavar="f'(x)",
                        help='Derivative of the function.')
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
        def derivative(x): return eval(args.derivative)
    except NameError:
        print("f'(x) does not follow python syntax.")
    try:
        p = newton_raphson(f, derivative, args.initial, args.tolerance)
        print("Found a root at p= %11.6f" % p)
    except ValueError:
        print("No root in given bound.")
