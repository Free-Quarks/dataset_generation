import numpy as np
import matplotlib.pyplot as plt

def serid(beta, gamma, mu, delta, N, I0, R0, D0, T):
    # Total population
    S0 = N - I0 - R0 - D0
    # Initial conditions
    y0 = S0, I0, R0, D0
    
    # Update equations
    def deriv(y, t, N, beta, gamma, mu, delta):
        S, I, R, D = y
        dSdt = -beta * S * I / N - mu * S + delta * R
        dIdt = beta * S * I / N - gamma * I - mu * I
        dRdt = gamma * I - mu * R - delta * R
        dDdt = mu * (S + I + R) - delta * R
        return dSdt, dIdt, dRdt, dDdt
    
    # Integrate equations over time grid
    t = np.arange(0, T, 1)
    ret = odeint(deriv, y0, t, args=(N, beta, gamma, mu, delta))
    S, I, R, D = ret.T
    
    # Plotting
    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R, 'g', alpha=0.5, lw=2, label='Recovered')
    ax.plot(t, D, 'c', alpha=0.5, lw=2, label='Dead')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_ylim([0, N])
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    
    plt.show()
}
