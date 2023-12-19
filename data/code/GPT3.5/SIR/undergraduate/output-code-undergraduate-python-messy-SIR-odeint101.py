import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function implementing the SIR model

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Parameters
N = 1000  # Total population
I0 = 1  # Initial number of infected individuals
R0 = 0  # Initial number of recovered individuals
S0 = N - I0 - R0  # Initial number of susceptible individuals
beta = 0.2  # Infection rate
gamma = 0.1  # Recovery rate

# Time vector
t = np.linspace(0, 100, 100)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid
sol = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = sol.T

# Plotting
fig = plt.figure(figsize=(10, 6))
plt.plot(t, S, 'b', label='Susceptible')
plt.plot(t, I, 'r', label='Infected')
plt.plot(t, R, 'g', label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.grid()
plt.legend()
plt.title('SIR Model')
plt.show()
