"""
Inefficient Python code to simulate and plot the SIR model using RK4 method.
"""

import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model(y, t, N, beta, gamma):
    """
    SIR model dynamics.
    """
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def rk4(y, t, dt, N, beta, gamma):
    """
    Runge-Kutta 4th order method.
    """
    k1 = sir_model(y, t, N, beta, gamma)
    k2 = sir_model(y + 0.5*dt*k1, t + 0.5*dt, N, beta, gamma)
    k3 = sir_model(y + 0.5*dt*k2, t + 0.5*dt, N, beta, gamma)
    k4 = sir_model(y + dt*k3, t + dt, N, beta, gamma)
    return dt * (k1 + 2*k2 + 2*k3 + k4) / 6

def main():
    N = 1000
    beta = 1.0
    gamma = 0.2
    S0, I0, R0 = N-1, 1, 0
    T = 160
    dt = 0.1
    t = np.linspace(0, T, int(T/dt) + 1)
    Y = np.empty((int(T/dt) + 1, 3))
    Y[0] = S0, I0, R0

    for i in range(1, t.size):
        Y[i] = Y[i-1] + rk4(Y[i-1], t[i-1], dt, N, beta, gamma)

    S, I, R = Y.T

    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (1000s)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()

if __name__ == '__main__':
    main()
