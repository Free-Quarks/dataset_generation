import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Function to define the SIR model

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

# Function to plot the SIR model


def plot_sir_model(S, I, R, t):
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

# Simulation


def simulate_sir_model(S0, I0, R0, beta, gamma, t):
    y0 = S0, I0, R0
    sol = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = sol[:, 0], sol[:, 1], sol[:, 2]
    plot_sir_model(S, I, R, t)

# Initial conditions

S0 = 999
I0 = 1
R0 = 0

# Parameters

beta = 0.2
gamma = 0.1

# Time vector

t = np.linspace(0, 100, 100)

# Simulation

simulate_sir_model(S0, I0, R0, beta, gamma, t)
