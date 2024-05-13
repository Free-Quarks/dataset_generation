import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def serid_model(y, t, beta, gamma):
    S, E, I, R, D = y
    N = S + E + I + R + D

    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dIdt = gamma * E - (1 - gamma) * I
    dRdt = (1 - gamma) * I
    dDdt = 0.01 * I

    return [dSdt, dEdt, dIdt, dRdt, dDdt]


def plot_serid_model(N, beta, gamma):
    S0, E0, I0, R0, D0 = N - 1, 1, 0, 0, 0
    t = np.linspace(0, 100, 100)
    y0 = [S0, E0, I0, R0, D0]

    result = odeint(serid_model, y0, t, args=(beta, gamma))
    S, E, I, R, D = result.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, E, 'y', label='Exposed')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.plot(t, D, 'k', label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SERID Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}

