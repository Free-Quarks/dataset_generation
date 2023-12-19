import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def plot_SIR_model(S, I, R):
    t = np.linspace(0, len(S), len(S))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()

# Initial conditions
S0 = 999
I0 = 1
R0 = 0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, num=100)

# Solve SIR model
y = odeint(SIR_model, [S0, I0, R0], t, args=(beta, gamma))
S, I, R = y[:, 0], y[:, 1], y[:, 2]

# Plot SIR model
plot_SIR_model(S, I, R)
