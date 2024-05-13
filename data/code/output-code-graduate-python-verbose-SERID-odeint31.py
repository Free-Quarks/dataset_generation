import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def serid_model(y, t, N, beta, gamma):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - gamma * E
    dIdt = gamma * E - D * I
    dRdt = (1 - D) * gamma * I
    dDdt = D * gamma * I
    return dSdt, dEdt, dIdt, dRdt, dDdt


def plot_serid_model(S, E, I, R, D, t):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')

    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of Individuals')
    ax.set_ylim(0, max(max(S), max(E), max(I), max(R), max(D)) * 1.1)
    ax.set_title('SERID Model')
    ax.legend()

    plt.show()


N = 1000
beta = 0.3
gamma = 0.1
D = 0.05
S0, E0, I0, R0, D0 = N-1, 1, 0, 0, 0

y0 = S0, E0, I0, R0, D0

t = np.linspace(0, 100, 1000)


result = odeint(serid_model, y0, t, args=(N, beta, gamma))
S, E, I, R, D = result.T

plot_serid_model(S, E, I, R, D, t)
