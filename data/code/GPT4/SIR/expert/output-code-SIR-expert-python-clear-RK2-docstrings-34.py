import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, beta, gamma):
    """
    SIR model differential equations.
    """
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I

    return dSdt, dIdt, dRdt

def rk2(y, t, dt, derivs):
    """
    Runge-Kutta 2 method.
    """
    k0 = dt*derivs(y, t)
    k1 = dt*derivs(y + k0, t + dt)
    y_next = y + 0.5*(k0 + k1)

    return y_next

# Initial conditions
S0, I0, R0 = 0.9, 0.1, 0.0 
beta, gamma = 0.35, 0.1

# Time grid
t = np.linspace(0, 100, 100)
dt = t[1] - t[0]

# Initial values
y = np.array([S0, I0, R0])

# Time evolution
S, I, R = [], [], []
for _t in t:
    _S, _I, _R = y
    S.append(_S)
    I.append(_I)
    R.append(_R)
    y = rk2(y, _t, dt, sir_model)

# Plot
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.legend()
plt.show()
