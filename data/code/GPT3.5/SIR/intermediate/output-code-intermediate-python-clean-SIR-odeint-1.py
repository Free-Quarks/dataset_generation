import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt


def plot_SIR_model(S, I, R, t):
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


def SIR_model(N, I0, R0, beta, gamma, days):
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # A grid of time points (in days)
    t = np.linspace(0, days, days)
    # Initial conditions vector
    y0 = N - I0 - R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    plot_SIR_model(S, I, R, t)


# Example usage
SIR_model(1000, 1, 0, 0.2, 0.1, 160)
