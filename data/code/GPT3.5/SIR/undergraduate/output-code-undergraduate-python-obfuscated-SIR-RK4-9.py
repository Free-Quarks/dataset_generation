import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, t):
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    S0 = N - I0 - R0
    y0 = S0, I0, R0

    t = np.linspace(0, t, t)
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    plt.plot(t, S, 'b', label='Susceptible')
    plt.plot(t, I, 'r', label='Infected')
    plt.plot(t, R, 'g', label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t = 160

SIR_model(beta, gamma, N, I0, R0, t)
