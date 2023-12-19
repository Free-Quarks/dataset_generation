import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, N, I0, R0, T):
    def model(y, t):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    h = T / 1000
    t = np.linspace(0, T, int(T/h)+1)

    y = np.zeros((len(t), 3))
    y[0] = N-I0-R0, I0, R0

    for i in range(len(t)-1):
        k1 = model(y[i], t[i])
        k2 = model(y[i] + h * k1, t[i] + h)
        y[i+1] = y[i] + h * (k1 + k2) / 2

    S, I, R = y[:, 0], y[:, 1], y[:, 2]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 100

SIR_RK2(beta, gamma, N, I0, R0, T)
