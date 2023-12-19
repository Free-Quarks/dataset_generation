import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR(t, S, I, R):
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


N = 1000  # total population
I0 = 1  # initial infected individuals
R0 = 0  # initial recovered individuals
S0 = N - I0 - R0  # initial susceptible individuals
beta = 0.2  # infection rate
gamma = 0.1  # recovery rate

# Time points
t = np.linspace(0, 100, 100)

# Initial conditions
y0 = S0, I0, R0

# Integrate the SIR equations over the time grid, t.
result = odeint(SIR_model, y0, t, args=(N, beta, gamma))

# Unpack results
S, I, R = result.T

# Plot the SIR curves
plot_SIR(t, S, I, R)
