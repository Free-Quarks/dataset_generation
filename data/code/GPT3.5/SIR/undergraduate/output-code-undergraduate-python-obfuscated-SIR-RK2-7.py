import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    n_steps = int(t_max / dt)
    S = np.zeros(n_steps)
    I = np.zeros(n_steps)
    R = np.zeros(n_steps)
    t = np.linspace(0, t_max, n_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, n_steps):
        k1 = dt * (-beta * S[i-1] * I[i-1] / N)
        l1 = dt * (beta * S[i-1] * I[i-1] / N - gamma * I[i-1])
        k2 = dt * (-beta * (S[i-1] + k1/2) * (I[i-1] + l1/2) / N)
        l2 = dt * (beta * (S[i-1] + k1/2) * (I[i-1] + l1/2) / N - gamma * (I[i-1] + l1/2))
        S[i] = S[i-1] + k2
        I[i] = I[i-1] + l2
        R[i] = N - S[i] - I[i]
    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

SIR_RK2(990, 10, 0, 0.3, 0.1, 100, 0.1)
