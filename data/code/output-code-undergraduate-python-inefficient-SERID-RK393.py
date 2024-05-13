import numpy as np
import matplotlib.pyplot as plt


def SERID(beta, gamma, rho, delta, N, I0, R0, E0, D0, T):
    # Total population, N.
    # Initial conditions vector
    y0 = S0, E0, I0, R0, D0
    # Integrate the SERID equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, rho, delta))
    S, E, I, R, D = ret.T
    
    # Plotting
    fig, ax = plt.subplots()    
    ax.plot(t, S, 'b', alpha=0.7, linewidth=2, label='Susceptible')
    ax.plot(t, E, 'y', alpha=0.7, linewidth=2, label='Exposed')
    ax.plot(t, I, 'r', alpha=0.7, linewidth=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.7, linewidth=2, label='Recovered')
    ax.plot(t, D, 'k', alpha=0.7, linewidth=2, label='Deceased')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Population')
    ax.set_ylim([0, N])
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.title('SERID Model')
    plt.show()


def deriv(y, t, N, beta, gamma, rho, delta):
    S, E, I, R, D = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - rho * E
    dIdt = rho * E - gamma * I - delta * I
    dRdt = gamma * I
    dDdt = delta * I
    return dSdt, dEdt, dIdt, dRdt, dDdt
