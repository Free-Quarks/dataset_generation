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


def seir_simulation(N, E0, I0, R0, beta, gamma, sigma, days):
    S0 = N - E0 - I0 - R0
    t = np.linspace(0, days, days)
    y0 = S0, E0, I0, R0
    result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
    S, E, I, R = result.T

    plt.plot(t, S, color='blue', label='Susceptible')
    plt.plot(t, E, color='orange', label='Exposed')
    plt.plot(t, I, color='red', label='Infected')
    plt.plot(t, R, color='green', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model Simulation')
    plt.legend()
    plt.grid(True)
    plt.show()


seir_simulation(1000, 10, 1, 0, 0.8, 0.2, 0.1, 150)
