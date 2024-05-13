import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(beta, delta, gamma, theta, rho, epsilon, N, I0, E0, R0, T):
    def derivs(state, t):
        S, I, D, A, R, T, H, E = state
        N = S + I + D + A + R + T + H + E
        dSdt = -beta * S * (I + theta * A) / N
        dEdt = beta * S * (I + theta * A) / N - delta * E
        dIdt = delta * epsilon * E - (1 - rho) * gamma * I - rho * H
        dAdt = delta * (1 - epsilon) * E - gamma * (1 - rho) * A
        dRdt = (1 - rho) * gamma * I + gamma * (1 - rho) * A
        dTdt = theta * beta * S * (I + theta * A) / N
        dHdt = rho * H
        dDdt = rho * gamma * I
        return dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt

    t = np.arange(0, T + 1, 1)
    y0 = N - I0 - E0 - R0 - T - H0 - D0
    state0 = y0, I0, D0, A0, R0, T0, H0, E0
    ys = odeint(derivs, state0, t)
    S, I, D, A, R, T, H, E = ys.T

    fig, ax = plt.subplots()
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, D, 'g', label='Deceased')
    ax.plot(t, A, 'm', label='Asymptomatic')
    ax.plot(t, R, 'c', label='Recovered')
    ax.plot(t, T, 'y', label='Transferred')
    ax.plot(t, H, 'k', label='Hospitalized')
    ax.plot(t, E, 'y', label='Exposed')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_title('SIDARTHE Model')
    ax.legend()
    plt.show()
}

