import numpy as np
import matplotlib.pyplot as plt


def rk3_solver(func, y0, t0, t_end, h):
    t = np.arange(t0, t_end + h, h)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        k1 = func(t[i-1], y[i-1])
        k2 = func(t[i-1] + h/2, y[i-1] + h/2 * k1)
        k3 = func(t[i-1] + h, y[i-1] - h * k1 + 2 * h * k2)
        y[i] = y[i-1] + h/6 * (k1 + 4 * k2 + k3)
    return y


def serid_model(t, y):
    S, E, R, I, D = y
    N = S + E + R + I + D
    beta = 0.2
    gamma = 0.1
    sigma = 0.1
    mu = 0.05
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dRdt = gamma * I
    dIdt = sigma * E - gamma * I - mu * I
    dDdt = mu * I
    return [dSdt, dEdt, dRdt, dIdt, dDdt]


def main():
    t0 = 0
    t_end = 10
    h = 0.1
    y0 = [990, 10, 0, 0, 0]
    y = rk3_solver(serid_model, y0, t0, t_end, h)
    t = np.arange(t0, t_end + h, h)
    S = y[:, 0]
    E = y[:, 1]
    R = y[:, 2]
    I = y[:, 3]
    D = y[:, 4]
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, E, label='Exposed')
    plt.plot(t, R, label='Recovered')
    plt.plot(t, I, label='Infected')
    plt.plot(t, D, label='Dead')
    plt.xlabel('Time')
    plt.ylabel('Number of individuals')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
