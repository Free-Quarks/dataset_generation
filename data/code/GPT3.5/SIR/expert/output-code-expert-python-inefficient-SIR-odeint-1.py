import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_sir_model(S, I, R, t):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model Simulation')
    plt.legend()
    plt.show()


def simulate_sir_model(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    ret = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = ret.T
    plot_sir_model(S, I, R, t)


# example usage
S0 = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t = np.linspace(0, 100, 100)

simulate_sir_model(S0, I0, R0, beta, gamma, t)
