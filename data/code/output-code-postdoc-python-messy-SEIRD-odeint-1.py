import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


def seird_model(y, t, N, beta, gamma, alpha, delta):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - alpha * E
    dIdt = alpha * E - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def plot_seird(t, S, E, I, R, D):
    plt.figure(figsize=(10,6))
    plt.plot(t, S, 'b-', label='Susceptible')
    plt.plot(t, E, 'm-', label='Exposed')
    plt.plot(t, I, 'r-', label='Infected')
    plt.plot(t, R, 'g-', label='Recovered')
    plt.plot(t, D, 'k-', label='Dead')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SEIRD Model')
    plt.legend()
    plt.grid(True)
    plt.show()


N = 1000
beta = 0.2
gamma = 0.1
alpha = 0.02
delta = 0.01
S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0
y0 = S0, E0, I0, R0, D0


t = np.linspace(0, 100, 100)
solution = odeint(seird_model, y0, t, args=(N, beta, gamma, alpha, delta))
S, E, I, R, D = solution.T


plot_seird(t, S, E, I, R, D)

