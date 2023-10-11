import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def seir_model(y, t, N, beta, gamma, sigma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt


def run_seir_model(N, E0, I0, R0, beta, gamma, sigma, days):
    S0 = N - E0 - I0 - R0
    y0 = S0, E0, I0, R0
    t = np.linspace(0, days, days)

    result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = result.T

    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


N = 10000
E0, I0, R0 = 1, 0, 0
beta, gamma, sigma = 0.2, 0.1, 1/5
run_seir_model(N, E0, I0, R0, beta, gamma, sigma, 100)
