import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function that defines the SIR model equations


def sir_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


# Function to plot the SIR model

def plot_sir_model(t, S, I, R):
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


# Initial conditions
N = 1000  # Total population
I0 = 1     # Initial number of infected individuals
R0 = 0     # Initial number of recovered individuals
S0 = N - I0 - R0   # Initial number of susceptible individuals

# Parameters
beta = 0.2   # Contact rate
gamma = 0.1  # Recovery rate

# Time vector
t = np.linspace(0, 100, 100)

# Solve the SIR model equations
y = S0, I0, R0
sol = odeint(sir_model, y, t, args=(N, beta, gamma))
S, I, R = sol.T

# Plot the SIR model
plot_sir_model(t, S, I, R)
