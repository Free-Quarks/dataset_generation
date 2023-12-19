import numpy as np
import matplotlib.pyplot as plt


def sir_model(beta, gamma, N, I0, R0, t_end, num_steps):
    def sir_deriv(y, t, beta, gamma, N):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, t_end, num_steps)
    y0 = N - I0 - R0
    S0, I0, R0 = y0, I0, R0
    y0 = S0, I0, R0
    ret = odeint(sir_deriv, y0, t, args=(beta, gamma, N))
    S, I, R = ret.T
    
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)
    ax.plot(t, S/N, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/N, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/N, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Fraction of population')
    ax.set_title('SIR Model')
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='gray', lw=0.5, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    plt.show()

