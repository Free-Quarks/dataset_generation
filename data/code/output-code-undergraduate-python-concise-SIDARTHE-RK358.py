import numpy as np
import matplotlib.pyplot as plt

def sidarthe_model(N, I0, D0, A0, R0, T0, H0, E0, beta, rho, sigma, gamma, mu, theta, delta, alpha, delta2, t_max):
    
    def deriv(y, t, N, beta, rho, sigma, gamma, mu, theta, delta, alpha, delta2):
        S, I, D, A, R, T, H, E = y
        dSdt = -beta * S * (I + rho * A) / N
        dIdt = beta * S * (I + rho * A) / N - (sigma + mu + gamma) * I
        dDdt = gamma * theta * I - delta * D
        dAdt = sigma * I - alpha * A
        dRdt = gamma * (1 - theta) * I + delta * D
        dTdt = delta2 * D
        dHdt = (sigma + mu) * I - delta2 * H
        dEdt = beta * S * (I + rho * A) / N
        return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt
    
    # Initial conditions vector
    y0 = N-I0-D0-A0-R0-T0-H0-E0
    
    # Integrate the SIR equations over the time grid, t.
    t = np.linspace(0, t_max, t_max)
    y = odeint(deriv, y0, t, args=(N, beta, rho, sigma, gamma, mu, theta, delta, alpha, delta2))
    S, I, D, A, R, T, H, E = y.T
    
    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/1000, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/1000, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/1000, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
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
}

