import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_results(t, S, I, R):
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


# Parameters
N = 1000
beta = 0.2
gamma = 0.1
S0, I0, R0 = N-1, 1, 0  # initial conditions

# Time vector
t = np.linspace(0, 100, 100)

# Solve model
result = odeint(SIR_model, [S0, I0, R0], t, args=(N, beta, gamma))
S, I, R = result.T

# Plot results
plot_results(t, S, I, R)
