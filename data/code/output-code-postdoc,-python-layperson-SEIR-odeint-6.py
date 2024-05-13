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


def plot_seir_model(t, S, E, I, R):
    plt.figure(figsize=(8, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SEIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


N = 1000
beta = 0.2
gamma = 0.1
sigma = 0.1
E0, I0, R0 = 1, 0, 0
S0 = N - E0 - I0 - R0
y0 = S0, E0, I0, R0


t = np.linspace(0, 100, 100)


result = odeint(seir_model, y0, t, args=(N, beta, gamma, sigma))
S, E, I, R = result.T


plot_seir_model(t, S, E, I, R)
