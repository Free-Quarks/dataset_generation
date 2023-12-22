# Required Libraries
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

def _cc(a, b, c, d): 
    """
    SIR model differential equations.
    """
    # Variables
    S, I, R = a
    N=S+I+R

    # Equations
    dSdt = -b * S * I / N
    dIdt = b * S * I / N - c * I
    dRdt = c * I

    return dSdt, dIdt, dRdt

def _dd(e, f, g, h, i):
    """
    Integrate the SIR equations over the time grid, t.
    """
    # Initial conditions vector
    a0 = e, f, g
    # Integrate the SIR equations over the time grid, t.
    return integrate.odeint(_cc, a0, h, args=(i[0], i[1]))

def _ee(j, k, l, m, n, o):
    """
    Plot the data on three separate curves for S(t), I(t) and R(t)
    """
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(j, k/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(j, l/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(j, m/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
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

# Parameters
N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.2, 1./10 
t = np.linspace(0, 160, 160)

# Integrate the SIR equations over the time grid, t
ret = _dd(S0, I0, R0, t, (beta, gamma))
S, I, R = ret.T

_ee(t, S, I, R, N, (beta, gamma))
