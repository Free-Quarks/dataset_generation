import numpy as np
import matplotlib.pyplot as plt


def SIDARTHE_RK3(N, beta, sigma, gamma, mu, delta, alpha, rho, theta, tmax, dt):
    def f(y):
        S, I, D, A, R, T, H, E = y
        Sp = -beta * S * I / N
        Ip = beta * S * I / N - sigma * I
        Dp = gamma * R
        Ap = sigma * I - (mu + delta) * A
        Rp = mu * A - gamma * R
        Tp = delta * A
        Hp = rho * (mu + delta) * A - theta * H
        Ep = (1 - rho) * (mu + delta) * A
        return np.array([Sp, Ip, Dp, Ap, Rp, Tp, Hp, Ep])

    t = np.linspace(0, tmax, int(tmax / dt) + 1)
    y = np.zeros((int(tmax / dt) + 1, 8))
    y[0] = N - 1, 1, 0, 0, 0, 0, 0, 0
    y[1] = y[0] - dt * f(y[0])
    y[2] = (3 * y[0] - y[1] - dt * f(y[1])) / 4
    for i in range(2, len(t) - 1):
        y[i + 1] = (y[i-1] + 2 * (1 + dt / dt_prev) * dt * f(y[i])) / (1 + dt / dt_prev + dt ** 2 / dt_prev ** 2)
    S, I, D, A, R, T, H, E = y.T

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Deceased')
    plt.plot(t, A, label='Asymptomatic')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, T, label='Tested')
    plt.plot(t, H, label='Hospitalized')
    plt.plot(t, E, label='Exposed')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.legend()
    plt.show()


SIDARTHE_RK3(10000, 1, 0.2, 0.05, 0.01, 0.01, 0.5, 0.5, 0.5, 100, 0.1)
