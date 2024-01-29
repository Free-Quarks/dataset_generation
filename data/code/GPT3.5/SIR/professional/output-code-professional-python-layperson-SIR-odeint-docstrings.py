import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]


def plot_results(t, S, I, R):
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    N = 1000
    I0 = 1
    R0 = 0
    S0 = N - I0 - R0
    beta = 0.2
    gamma = 0.1
    t = np.linspace(0, 100, 1000)
    y0 = [S0, I0, R0]
    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = solution.T
    plot_results(t, S, I, R)