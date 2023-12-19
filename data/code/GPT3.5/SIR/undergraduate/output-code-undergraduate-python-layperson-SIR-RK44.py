import numpy as np
import matplotlib.pyplot as plt


def SIR_RK4(N, beta, gamma, I0, T):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = N - I0, I0, 0
    res = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = res.T

    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model using RK4')
    plt.legend()
    plt.show()


N = 1000
beta = 0.2
gamma = 0.1
I0 = 1
T = 160

SIR_RK4(N, beta, gamma, I0, T)
