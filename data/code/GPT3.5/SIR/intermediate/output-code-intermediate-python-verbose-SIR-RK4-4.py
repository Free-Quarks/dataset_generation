import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(beta, gamma, N, I0, R0, t_end, dt):
    def derivative(y, t):
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.arange(0, t_end, dt)
    y0 = N - I0 - R0
    y = np.zeros((len(t), 3))
    y[0] = y0, I0, R0

    for i in range(1, len(t)):
        k1 = dt * derivative(y[i - 1], t[i - 1])
        k2 = dt * derivative(y[i - 1] + 0.5 * k1, t[i - 1] + 0.5 * dt)
        k3 = dt * derivative(y[i - 1] + 0.5 * k2, t[i - 1] + 0.5 * dt)
        k4 = dt * derivative(y[i - 1] + k3, t[i - 1] + dt)
        y[i] = y[i - 1] + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    S, I, R = y.T

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK4')
    plt.legend()
    plt.show()


SIR_RK4(0.5, 0.1, 1000, 1, 0, 100, 0.1)
