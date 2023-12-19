import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# function that returns dy/dt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# initial conditions
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0

# contact rate, beta, and mean recovery rate, gamma, (in 1/days)
beta, gamma = 0.2, 1./10 

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot the data
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.show()
