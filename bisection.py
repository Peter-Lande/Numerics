import argparse
import numpy as np
import sys
# The numpy is used for functions like sqrt, sine, cosine.
# argparse is for making it easy to run this program in the command line


def bisection(function, initial, final, tolerance=0.0001, iteration=0, max_iteration=sys.maxsize):
    p = (initial+final)/2
    pResult = function(p)
    initialResult = function(initial)
    finalResult = function(final)
    iterationTolerance = (final-initial)/2
    iteration += 1
    print("%11d %11.6f %11.6f %11.6f %11.6f %11.6f" %
          (iteration, initial, final, p, pResult, iterationTolerance))
    if initialResult * finalResult > 0:
        raise ValueError
    elif (iterationTolerance <= tolerance) or (max_iteration <= iteration):
        return p
    elif pResult == 0:
        return p
    if initialResult * pResult < 0:
        return bisection(function, initial, p, tolerance, iteration)
    else:
        return bisection(function, p, final, tolerance, iteration)


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
    print("%11s %11s %11s %11s %11s %11s" %
          ("Iteration", "a", "b", "p", "f(p)", "tolerance"))
    try:
        p = bisection(f, args.initial, args.final, args.tolerance)
        print("Found a root at p= %11.6f" % p)
    except ValueError:
        print("No root in given bound.")
