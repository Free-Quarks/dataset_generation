import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, t_max, dt):
    def f(t, y):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])

    t = np.arange(0, t_max, dt)
    y = np.zeros((len(t), 3))
    y[0] = np.array([N - I0 - R0, I0, R0])

    for i in range(1, len(t)):
        k1 = dt * f(t[i-1], y[i-1])
        k2 = dt * f(t[i-1] + dt/2, y[i-1] + k1/2)
        k3 = dt * f(t[i-1] + dt, y[i-1] - k1 + 2*k2)
        y[i] = y[i-1] + (k1 + 4*k2 + k3) / 6

    S, I, R = y.T

    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using Runge-Kutta 3')
    plt.legend()
    plt.grid(True)
    plt.show()


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0

SIR_RK3(beta, gamma, N, I0, R0, t_max=100, dt=0.1)
