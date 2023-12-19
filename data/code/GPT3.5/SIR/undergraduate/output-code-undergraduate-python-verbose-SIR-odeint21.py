import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# function that returns dy/dt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# initial conditions
N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0

# contact rate, beta, and mean recovery rate, gamma
beta = 0.2
gamma = 1./10

# A grid of time points (in days)
t = np.linspace(0, 160, 160)

# initial condition vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
ret = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plotting
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.grid()
plt.show()

