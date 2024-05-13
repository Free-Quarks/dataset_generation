import numpy as np
import matplotlib.pyplot as plt

def seir_model(beta, sigma, gamma, N, I0, E0, R0, T):
    # Total population, N.
    # Initial conditions
    S0 = N - E0 - I0 - R0
    
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # A grid of time points (in days)
    t = np.linspace(0, T, T)
    
    # The SEIR model differential equations.
    def deriv(y, t, N, beta, sigma, gamma):
        S, E, I, R = y
        dSdt = -beta * S * I / N
        dEdt = beta * S * I / N - sigma * E
        dIdt = sigma * E - gamma * I
        dRdt = gamma * I
        return dSdt, dEdt, dIdt, dRdt
    
    # Initial conditions vector
    y0 = S0, E0, I0, R0
    
    # Integrate the SEIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, sigma, gamma))
    S, E, I, R = ret.T
    
    # Plot the data on three separate curves for S(t), E(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, E/1000, 'y', alpha=0.5, lw=2, label='Exposed')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number (thousands)')
    ax.set_ylim(0,1.2)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    
    return S, E, I, R
}

