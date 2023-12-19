import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, S0, I0, R0, t_end, N, dt):
    def SIR_model(t, X):
        S, I, R = X
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])

    t = np.arange(0, t_end, dt)
    X = np.zeros((len(t), 3))
    X[0] = np.array([S0, I0, R0])

    for i in range(1, len(t)):
        k1 = SIR_model(t[i-1], X[i-1])
        k2 = SIR_model(t[i-1] + dt/2, X[i-1] + dt/2 * k1)
        k3 = SIR_model(t[i-1] + dt/2, X[i-1] + dt/2 * k2)
        k4 = SIR_model(t[i-1] + dt, X[i-1] + dt * k3)
        X[i] = X[i-1] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    plt.plot(t, X[:, 0], label='Susceptible')
    plt.plot(t, X[:, 1], label='Infected')
    plt.plot(t, X[:, 2], label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

SIR_RK4(0.3, 0.1, 999, 1, 0, 100, 1000, 0.1)
