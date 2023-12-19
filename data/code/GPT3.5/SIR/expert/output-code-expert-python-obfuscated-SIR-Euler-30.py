import numpy as np
import matplotlib.pyplot as plt


def euler_sir_model(beta, gamma, S0, I0, R0, N, t_end, h):
    
    def deriv(y, t, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    t = np.linspace(0, t_end, int(t_end/h)+1)
    y0 = S0, I0, R0
    y = np.zeros((len(t),len(y0)))
    y[0] = y0

    for i in range(1,len(t)):
        y_prev = y[i-1]
        y[i] = y_prev + h * deriv(y_prev, t[i-1], beta, gamma)

    S, I, R = y[:,0], y[:,1], y[:,2]

    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model')
    plt.legend()
    plt.show()


beta = 0.2
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
N = S0 + I0 + R0

euler_sir_model(beta, gamma, S0, I0, R0, N, 100, 0.1)
