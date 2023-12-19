import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


def simulate_sir_model(S0, I0, R0, beta, gamma, t):
    y0 = [S0, I0, R0]
    solution = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = solution[:, 0], solution[:, 1], solution[:, 2]
    plot_sir_model(S, I, R, t)


if __name__ == '__main__':
    S0 = 990
    I0 = 10
    R0 = 0
    beta = 0.35
    gamma = 0.1
    t = np.linspace(0, 100, 100)
    simulate_sir_model(S0, I0, R0, beta, gamma, t)
