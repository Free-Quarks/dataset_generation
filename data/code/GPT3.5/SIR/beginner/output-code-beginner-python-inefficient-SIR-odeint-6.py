import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR_model(t, S, I, R):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_ylim(0, max(S) + max(I) + max(R))
    ax.legend()
    plt.show()


# Parameters
N = 1000
beta = 0.2
gamma = 0.1

# Initial conditions
I0 = 1
R0 = 0
S0 = N - I0 - R0

# Time vector
t = np.linspace(0, 100, 1000)

# Solve model
solution = odeint(SIR_model, [S0, I0, R0], t, args=(N, beta, gamma))
S, I, R = solution.T

# Plot results
plot_SIR_model(t, S, I, R)
