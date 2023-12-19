import numpy as np
import matplotlib.pyplot as plt


def SIR_model(N, beta, gamma, I0, R0, T):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, T, T+1)
    y0 = N - I0 - R0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of individuals')
    plt.title('SIR Model')
    plt.legend()
    plt.grid(True)
    plt.show()


SIR_model(1000, 0.2, 0.1, 1, 0, 100)
