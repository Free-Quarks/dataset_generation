import numpy as np
import matplotlib.pyplot as plt


def sidarthe_model(t, y, params):
    beta, gamma, delta, alpha, epsilon, theta, rho, population = params
    S, I, D, A, R, T, H, E = y
    N = np.sum(y)

    dSdt = -beta * S * (I + alpha * A + theta * T + rho * H) / N
    dIdt = beta * S * (I + alpha * A + theta * T + rho * H) / N - (gamma + delta) * I
    dDdt = delta * I
    dAdt = epsilon * delta * I - (gamma + alpha + epsilon) * A
    dRdt = gamma * (I + A + T + H)
    dTdt = alpha * A - (gamma + theta) * T
    dHdt = theta * T - (gamma + rho) * H
    dEdt = epsilon * A

    return [dSdt, dIdt, dDdt, dAdt, dRdt, dTdt, dHdt, dEdt]


def run_sidarthe_model(init, params, t_max, dt):
    t = np.arange(0, t_max, dt)

    sol = odeint(sidarthe_model, init, t, args=(params,))
    S, I, D, A, R, T, H, E = sol.T

    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Tested')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.title('SIDARTHE Model')
    plt.legend()
    plt.grid(True)
    plt.show()
}

