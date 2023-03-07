import matplotlib.pyplot as mp
import scipy.special as sp
import numpy as np


def forwardEuler(start, end, stepNumber, initialCondition, function):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialCondition]
    for i in range(stepNumber):
        y.append(y[i]+h*function(t[i], y[i]))
        t.append(start+(i+1)*h)
    return (t, y)


def forwardEuler2D(start, end, stepNumber, initialConditionX, initialConditionY, functionX, functionY):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialConditionY]
    x = [initialConditionX]
    for i in range(stepNumber):
        y.append(y[i]+h*functionY(t[i], y[i], x[i]))
        x.append(x[i]+h*functionX(t[i], y[i], x[i]))
        t.append(start+(i+1)*h)
    return (t, y, x)


def modifiedEuler(start, end, stepNumber, initialCondition, function):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialCondition]
    for i in range(stepNumber):
        y.append(y[i]+(h/2)*(function(t[i], y[i]) +
                 function(start+(i+1)*h, y[i]+h*function(t[i], y[i]))))
        t.append(start+(i+1)*h)
    return (t, y)


def midpoint(start, end, stepNumber, initialCondition, function):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialCondition]
    for i in range(stepNumber):
        y.append(y[i]+(h)*(function(t[i]+h/2, y[i]+(h/2)*(function(t[i], y[i])))))
        t.append(start+(i+1)*h)
    return (t, y)


def rk4(start, end, stepNumber, initialCondition, function):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialCondition]
    for i in range(stepNumber):
        k1 = h*function(t[i], y[i])
        k2 = h*function(t[i]+h/2, y[i]+k1/2)
        k3 = h*function(t[i]+h/2, y[i]+k2/2)
        k4 = h*function(t[i], y[i]+k3)
        y.append(y[i]+(k1+2*k2+2*k3+k4)/6)
        t.append(start+(i+1)*h)
    return (t, y)


def timeAxis(start, end, stepNumber):
    h = (end-start)/stepNumber
    return [start+i*h for i in range(stepNumber)]


def newton_raphson(function, derivative, initial, t, inity, h, error):
    next_guess = initial - (initial - function(t, initial)
                            * h - inity)/(1+derivative(t, initial)*h)
    if np.abs((next_guess-initial)/initial) < error:
        return next_guess
    newton_raphson(function, derivative, next_guess, t, inity, h, error)


def newton_raphson2DY(function, derivative, initialX, initialY, t, inity, h, error):
    next_guess = initialY - (initialY - function(t, initialY, initialX)
                             * h - inity)/(1+derivative(t, initialY, initialX)*h)
    if np.abs((next_guess-initialY)/initialY) < error:
        return next_guess
    newton_raphson2DY(function, derivative, initialX,
                      next_guess, t, inity, h, error)


def newton_raphson2DX(function, derivative, initialX, initialY, t, inity, h, error):
    next_guess = initialY - (initialY - function(t, initialX, initialY)
                             * h - inity)/(1+derivative(t, initialX, initialY)*h)
    print(next_guess)
    if np.abs((next_guess-initialY)/initialY) < error:
        return next_guess
    newton_raphson2DX(function, derivative, initialX,
                      next_guess, t, inity, h, error)


def implicit(start, end, stepNumber, initialCondition, function, derivative):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialCondition]
    for i in range(stepNumber):
        y.append(newton_raphson(
            function, derivative, y[i], t[i], y[i], h, 0.1))
        t.append(start+(i+1)*h)
    return (t, y)


def implicit2D(start, end, stepNumber, initialConditionX, initialConditionY, functionX, functionY, derivativeX, derivativeY):
    h = (end-start)/stepNumber
    t = [start]
    y = [initialConditionY]
    x = [initialConditionX]
    for i in range(stepNumber):
        x.append(newton_raphson2DX(functionX, derivativeX,
                 y[i], x[i], t[i], x[i], h, 0.1))
        y.append(newton_raphson2DY(
            functionY, derivativeY, x[i], y[i], t[i], y[i], h, 0.1))

        t.append(start+(i+1)*h)
    return (t, y, x)


if __name__ == "__main__":
    def initFunc(t, y):
        return (y*(t+1))/(t*(y+1))

    def initDerivative(t, y):
        return ((t+1))/(t*(y+1)**2)

    euler = mp.figure()
    ax = mp.subplot(311)
    t = np.array(timeAxis(2, 4, 200))
    (x, y) = forwardEuler(2, 4, 200, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=200")
    ax = mp.subplot(312)
    t = np.array(timeAxis(2, 4, 400))
    (x, y) = forwardEuler(2, 4, 400, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=400")
    ax = mp.subplot(313)
    t = np.array(timeAxis(2, 4, 800))
    (x, y) = forwardEuler(2, 4, 800, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=800")
    mp.tight_layout()
    mp.savefig('Forward_Euler.png', format='png')
    modified = mp.figure()
    ax = mp.subplot(311)
    t = np.array(timeAxis(2, 4, 200))
    (x, y) = modifiedEuler(2, 4, 200, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=200")
    ax = mp.subplot(312)
    t = np.array(timeAxis(2, 4, 400))
    (x, y) = modifiedEuler(2, 4, 400, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=400")
    ax = mp.subplot(313)
    t = np.array(timeAxis(2, 4, 800))
    (x, y) = modifiedEuler(2, 4, 800, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=800")
    mp.tight_layout()
    mp.savefig('Modified_Euler.png', format='png')
    midpointfig = mp.figure()
    ax = mp.subplot(311)
    t = np.array(timeAxis(2, 4, 200))
    (x, y) = midpoint(2, 4, 200, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=200")
    ax = mp.subplot(312)
    t = np.array(timeAxis(2, 4, 400))
    (x, y) = midpoint(2, 4, 400, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=400")
    ax = mp.subplot(313)
    t = np.array(timeAxis(2, 4, 800))
    (x, y) = midpoint(2, 4, 800, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=800")
    mp.tight_layout()
    mp.savefig('Midpoint.png', format='png')
    rk = mp.figure()
    ax = mp.subplot(311)
    t = np.array(timeAxis(2, 4, 200))
    (x, y) = rk4(2, 4, 200, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=200")
    ax = mp.subplot(312)
    t = np.array(timeAxis(2, 4, 400))
    (x, y) = rk4(2, 4, 400, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=400")
    ax = mp.subplot(313)
    t = np.array(timeAxis(2, 4, 800))
    (x, y) = rk4(2, 4, 800, 4, initFunc)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=800")
    mp.tight_layout()
    mp.savefig('Runge_Kutta.png', format='png')
    backward = mp.figure()
    ax = mp.subplot(311)
    t = np.array(timeAxis(2, 4, 200))
    (x, y) = implicit(2, 4, 200, 4, initFunc, initDerivative)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=200")
    ax = mp.subplot(312)
    t = np.array(timeAxis(2, 4, 400))
    (x, y) = implicit(2, 4, 400, 4, initFunc, initDerivative)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=400")
    ax = mp.subplot(313)
    t = np.array(timeAxis(2, 4, 800))
    (x, y) = implicit(2, 4, 800, 4, initFunc, initDerivative)
    ax.plot(t, sp.lambertw(2*t*np.exp(t+2)), label='Analytic')
    ax.plot(x, y, label='Numerical')
    ax.legend(loc="upper right")
    ax.set_title("N=800")
    mp.tight_layout()
    mp.savefig('Backward_Euler.png', format='png')
