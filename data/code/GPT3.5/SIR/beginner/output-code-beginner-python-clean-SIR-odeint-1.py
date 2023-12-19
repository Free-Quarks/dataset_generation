import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Function that returns dy/dt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Initial conditions
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0

# Parameters: contact rate and mean recovery rate
beta, gamma = 0.2, 1./10

# Time vector
t = np.linspace(0, 160, 160)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid
ret = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

# Plot
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model')
plt.legend()
plt.grid(True)
plt.show()
