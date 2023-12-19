import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR_model(S, I, R, t):
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


N = 1000   # total population
I0 = 1     # initial number of infected individuals
R0 = 0     # initial number of recovered individuals
S0 = N - I0 - R0   # initial number of susceptible individuals
beta = 0.2   # effective contact rate
gamma = 0.1  # recovery rate

# Time vector
t = np.linspace(0, 100, 1000)

# Initial conditions vector
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
results = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = results.T

# Plot the data
plot_SIR_model(S, I, R, t)
