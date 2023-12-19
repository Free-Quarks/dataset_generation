import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, N, I0, R0, T):
    dt = 0.1
    num_steps = int(T / dt)
    t = np.linspace(0, T, num_steps + 1)
    S = np.zeros(num_steps + 1)
    I = np.zeros(num_steps + 1)
    R = np.zeros(num_steps + 1)
    S[0] = N - I0
    I[0] = I0
    R[0] = R0

    for i in range(num_steps):
        k1 = dt * (-beta * S[i] * I[i] / N)
        l1 = dt * (beta * S[i] * I[i] / N - gamma * I[i])

        k2 = dt * (-beta * (S[i] + k1 / 2) * (I[i] + l1 / 2) / N)
        l2 = dt * (beta * (S[i] + k1 / 2) * (I[i] + l1 / 2) / N - gamma * (I[i] + l1 / 2))

        S[i+1] = S[i] + k2
        I[i+1] = I[i] + l2
        R[i+1] = N - S[i+1] - I[i+1]

    plt.plot(t, S, label='S')
    plt.plot(t, I, label='I')
    plt.plot(t, R, label='R')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()


SIR_RK2(0.3, 0.1, 1000, 1, 0, 10)
