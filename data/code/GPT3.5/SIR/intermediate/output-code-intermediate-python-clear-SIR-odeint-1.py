import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def SIR_model(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR_model(S, I, R, t):
    plt.figure(figsize=(8, 6))
    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


N = 1000
I0 = 1
R0 = 0
S0 = N - I0 - R0
beta = 0.2
gamma = 0.1

t = np.linspace(0, 100, 100)

y0 = S0, I0, R0

solution = odeint(SIR_model, y0, t, args=(N, beta, gamma))

S, I, R = solution.T

plot_SIR_model(S, I, R, t)
