import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# function that returns dy/dt

def SIR(y, t, N, beta, gamma):
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

# a grid of time points (in days)
t = np.linspace(0, 160, 160)

# solve ode
y0 = S0, I0, R0
sol = odeint(SIR, y0, t, args=(N, beta, gamma))
S, I, R = sol.T

# plot results
plt.figure(figsize=(10,6))
plt.plot(t, S/1000, 'b', label='Susceptible')
plt.plot(t, I/1000, 'r', label='Infected')
plt.plot(t, R/1000, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals (thousands)')
plt.title('SIR Model')
plt.legend()
plt.show()
