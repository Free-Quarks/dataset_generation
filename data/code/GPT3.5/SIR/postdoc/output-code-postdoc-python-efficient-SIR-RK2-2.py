import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, tmax, dt):
    def derivs(y, t):
        S, I, R = y
        dS = -beta * S * I / N
        dI = beta * S * I / N - gamma * I
        dR = gamma * I
        return np.array([dS, dI, dR])
    
    S0 = N - 1
    I0 = 1
    R0 = 0
    y0 = np.array([S0, I0, R0])
    t = np.arange(0, tmax+dt, dt)
    y = np.zeros((len(t), 3))
    y[0] = y0
    
    for i in range(len(t)-1):
        k1 = derivs(y[i], t[i])
        k2 = derivs(y[i] + dt * k1, t[i] + dt)
        y[i+1] = y[i] + 0.5 * dt * (k1 + k2)

    plt.plot(t, y[:, 0], label='Susceptible')
    plt.plot(t, y[:, 1], label='Infected')
    plt.plot(t, y[:, 2], label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.title('SIR Model')
    plt.show()


SIR_RK2(0.25, 0.1, 1000, 100, 0.1)
