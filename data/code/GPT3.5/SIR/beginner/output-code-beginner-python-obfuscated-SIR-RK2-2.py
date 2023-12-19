import numpy as np
import matplotlib.pyplot as plt

def sir_rk2(beta, gamma, N, I0, R0, duration, dt):
    def deriv(y, t, beta, gamma, N):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = (beta * S * I / N) - (gamma * I)
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.arange(0, duration, dt)
    y = np.zeros((len(t), 3))
    y[0] = N - I0 - R0, I0, R0

    for i in range(1, len(t)):
        k1 = deriv(y[i-1], t[i-1], beta, gamma, N)
        k2 = deriv(y[i-1] + dt * k1 / 2, t[i-1] + dt / 2, beta, gamma, N)
        y[i] = y[i-1] + dt * k2

    S, I, R = y[:, 0], y[:, 1], y[:, 2]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()

