import numpy as np
import matplotlib.pyplot as plt

def SIR_RK4(beta, gamma, N, I0, R0, T):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, T, T)
    y0 = N - I0, I0, R0
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(t, S, label='Susceptible')
    ax.plot(t, I, label='Infected')
    ax.plot(t, R, label='Recovered')
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Number of individuals')
    ax.set_title('SIR Model using RK4')
    ax.legend()
    plt.show()


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
T = 160

SIR_RK4(beta, gamma, N, I0, R0, T)
