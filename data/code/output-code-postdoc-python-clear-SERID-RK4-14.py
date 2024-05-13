import numpy as np
import matplotlib.pyplot as plt


def serid_model(beta, gamma, nu, mu, N, I0, E0, R0, D0, t_max):
    def deriv(y, t, N, beta, gamma, nu, mu):
        S, E, R, I, D = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - nu * E - mu * E
        dRdt = gamma * I
        dIdt = nu * E - gamma * I - mu * I
        dDdt = mu * (E + I)
        return dSdt, dEdt, dRdt, dIdt, dDdt

    t = np.linspace(0, t_max, t_max + 1)
    y0 = N - E0 - I0 - R0 - D0, E0, R0, I0, D0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, nu, mu))
    S, E, R, I, D = ret.T

    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S / N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E / N, 'y', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, R / N, 'g', alpha=0.5, lw=2, label='Recovered')
    ax.plot(t, I / N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, D / N, 'm', alpha=0.5, lw=2, label='Dead')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Normalized population')
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='gray', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()


serid_model(0.8, 0.1, 0.2, 0.05, 1000, 1, 0, 0, 0, 100)

