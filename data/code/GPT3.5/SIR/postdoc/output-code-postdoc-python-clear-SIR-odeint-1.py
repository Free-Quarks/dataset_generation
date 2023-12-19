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
    plt.title('SIR Model')
    plt.legend()
    plt.show()


# Initial conditions
N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0

# Parameters
beta = 0.2
gamma = 0.1

# Time vector
t = np.linspace(0, 100, 1000)

# Solve the SIR model
y0 = S0, I0, R0
result = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = result.T

# Plot the results
plot_results(t, S, I, R)

