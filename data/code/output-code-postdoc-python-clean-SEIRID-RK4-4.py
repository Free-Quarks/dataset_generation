import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def seirid(y, t, N, beta, sigma, gamma, mu, p, q):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E - mu * E
    dIdt = sigma * E - gamma * I - p * I
    dRdt = gamma * I
    dDdt = p * I + mu * E
    return dSdt, dEdt, dIdt, dRdt, dDdt


def solve_seirid_model(N, E0, I0, R0, D0, beta, sigma, gamma, mu, p, q, t_end, dt):
    t = np.linspace(0, t_end, int(t_end/dt) + 1)
    y0 = N - E0 - I0 - R0 - D0
    y0 = y0, E0, I0, R0, D0
    result = odeint(seirid, y0, t, args=(N, beta, sigma, gamma, mu, p, q))
    S, E, I, R, D = result.T

    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Dead')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.legend()
    plt.show()


solve_seirid_model(N=1000, E0=1, I0=10, R0=0, D0=0, beta=0.2, sigma=0.1, gamma=0.1, mu=0.01, p=0.05, q=0, t_end=365, dt=1)
