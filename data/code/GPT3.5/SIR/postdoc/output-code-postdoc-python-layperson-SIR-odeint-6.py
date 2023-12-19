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
    plt.plot(t, S, '-', label='Susceptible')
    plt.plot(t, I, '-', label='Infected')
    plt.plot(t, R, '-', label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


N = 1000
beta = 0.2
gamma = 0.1
S0, I0, R0 = N-1, 1, 0

# Create the time vector
t = np.linspace(0, 100, 1000)

# Integrate the SIR equations over the time grid
sol = odeint(SIR_model, [S0, I0, R0], t, args=(N, beta, gamma))
S, I, R = sol.T

# Plot the results
plot_results(t, S, I, R)
