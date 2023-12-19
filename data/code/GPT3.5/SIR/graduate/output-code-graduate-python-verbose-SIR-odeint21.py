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
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axis_bgcolor='#dddddd', axisbelow=True)
    ax.plot(t, S / N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I / N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R / N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (Normalized)')
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


N = 1000
I0, R0 = 1, 0
S0 = N - I0 - R0
beta, gamma = 0.2, 1.0 / 10


t = np.linspace(0, 160, 160)

y0 = S0, I0, R0
ret = odeint(SIR_model, y0, t, args=(N, beta, gamma))
S, I, R = ret.T

plot_SIR_model(S, I, R, t)
