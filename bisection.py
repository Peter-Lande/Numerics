import argparse
import numpy as np

# The numpy is used for functions like sqrt, sine, cosine.
# argparse is for making it easy to run this program in the command line


def bisection(function, initial, final, tolerance):
    p = (initial+final)/2
    pResult = function(p)
    initialResult = function(initial)
    finalResult = function(final)
    iterationTolerance = (final-initial)/2
    print("%11.6f %11.6f %11.6f %11.6f %11.6f" %
          (initial, final, p, pResult, iterationTolerance))
    if initialResult * finalResult > 0:
        raise ValueError
    elif iterationTolerance < tolerance:
        return p
    elif pResult == 0:
        return p
    if initialResult * pResult < 0:
        return bisection(function, initial, p, tolerance)
    else:
        return bisection(function, p, final, tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Solve for roots of functions.', prog='bisection')
    parser.add_argument('function', metavar='f(x)',
                        help='Function to be solved.')
    parser.add_argument('initial', type=float,
                        help='Initial point of bisection method.')
    parser.add_argument('final', type=float,
                        help='Final point of bisection method.')
    parser.add_argument('tolerance', type=float,
                        help='Tolerance level for biscetion method.')
    args = parser.parse_args()
    try:
        def f(x): return eval(args.function)
    except NameError:
        print("f(x) does not follow python syntax.")
    print("%11s %11s %11s %11s %11s" % ("a", "b", "p", "f(p)", "tolerance"))
    try:
        p = bisection(f, args.initial, args.final, args.tolerance)
        print("Found a root at p= %11.6f" % p)
    except ValueError:
        print("No root in given bound.")