import ivp_solvers
import matplotlib.pyplot as mp
import numpy as np

if __name__ == '__main__':
    def function(x, y):
        return -2.5*y

    def position(t, y, x):
        return x

    def velocity(t, y, x):
        return -(0.16)/(0.5)*x-(9.81)/(1.2)*np.sin(y*(np.pi/180))

    def xfunc(t, y, x):
        return 998*x-1998*y

    def yfunc(t, y, x):
        return 1000*x-2000*y

    def xderv(t, y, x):
        return 998

    def yderv(t, y, x):
        return -2000

    first = mp.figure()
    ax = mp.subplot(211)
    t = np.array(ivp_solvers.timeAxis(0, 3.4, 17))
    (x, y) = ivp_solvers.forwardEuler(0, 3.4, 17, 1, function)
    ax.plot(t, np.exp(-2.5*t), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=17")
    ax = mp.subplot(212)
    t = np.array(ivp_solvers.timeAxis(0, 3.4, 4))
    (x, y) = ivp_solvers.forwardEuler(0, 3.4, 4, 1, function)
    ax.plot(t, np.exp(-2.5*t), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=4")
    mp.tight_layout()
    mp.savefig('HW11_Problem2_1.png', format='png')
    third = mp.figure()
    ax = mp.subplot(211)
    t = np.array(ivp_solvers.timeAxis(0, 3.4, 17))
    (x, y) = ivp_solvers.forwardEuler(0, 3.4, 17, 1, function)
    ax.plot(t, np.exp(-2.5*t), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=17")
    ax = mp.subplot(212)
    t = np.array(ivp_solvers.timeAxis(0, 3.4, 4))
    (x, y) = ivp_solvers.forwardEuler(0, 3.4, 4, 1, function)
    ax.plot(t, np.exp(-2.5*t), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=17")
    mp.tight_layout()
    mp.savefig('HW11_Problem2_2.png', format='png')
    second = mp.figure()
    ax = mp.subplot(211)
    (t, y, x) = ivp_solvers.forwardEuler2D(
        0, 18, 180, 0, 90, velocity, position)
    ax.plot(t, y, label='Position')
    ax.plot(t, x, label='Velocity')
    ax.legend(loc="upper right")
    ax.set_title("N=180")
    ax = mp.subplot(212)
    (t, y, x) = ivp_solvers.forwardEuler2D(
        0, 18, 1800, 0, 90, velocity, position)
    ax.plot(t, y, label='Position')
    ax.plot(t, x, label='Velocity')
    ax.legend(loc="upper right")
    ax.set_title("N=1800")
    mp.tight_layout()
    mp.savefig('HW11_Problem1.png', format='png')
    (t, y, x) = ivp_solvers.forwardEuler2D(
        0, 0.1, 1, 1, 2, xfunc, yfunc)
    print("X: {}".format(x[-1]))
    print("Y: {}".format(y[-1]))
    (t, y, x) = ivp_solvers.forwardEuler2D(
        0, 0.1, 1000, 1, 2, xfunc, yfunc)
    print("X: {}".format(x[-1]))
    print("Y: {}".format(y[-1]))
    (t, y, x) = ivp_solvers.implicit2D(
        0, 0.1, 1, 1, 2, xfunc, yfunc, xderv, yderv)
    print("X: {}".format(x[-1]))
    print("Y: {}".format(y[-1]))
