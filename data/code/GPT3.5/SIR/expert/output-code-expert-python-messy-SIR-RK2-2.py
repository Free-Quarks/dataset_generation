import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, beta, gamma, I0, T):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = N - I0, I0, 0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, S, 'b', label='Susceptible')
    ax.plot(t, I, 'r', label='Infected')
    ax.plot(t, R, 'g', label='Recovered')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of individuals')
    ax.legend()
    plt.show()


SIR_RK2(1000, 0.2, 0.1, 1, 100)
