import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, N, I0, T):
    S0 = N - I0
    R0 = 0
    dt = 0.1
    t = np.linspace(0, T, int(T/dt) + 1)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        k2 = -beta * (S[i-1] + dt/2*k1) * (I[i-1] + dt/2*k1) / N
        l1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        l2 = beta * (S[i-1] + dt/2*k1) * (I[i-1] + dt/2*k1) / N - gamma * (I[i-1] + dt/2*l1)

        S[i] = S[i-1] + dt * (k1 + k2) / 2
        I[i] = I[i-1] + dt * (l1 + l2) / 2
        R[i] = R[i-1] + dt * (gamma * (I[i-1] + dt/2*l1))

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()
}

