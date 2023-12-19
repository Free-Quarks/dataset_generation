import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# function that returns dy/dt

def SIRmodel(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# initial conditions
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0

# parameters
beta = 0.2
gamma = 0.1

# time points
t = np.linspace(0, 100, 100)

# solve ODE
y0 = S0, I0, R0
sol = odeint(SIRmodel, y0, t, args=(N, beta, gamma))
S, I, R = sol.T

# plot results
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
