import numpy as np
import matplotlib.pyplot as plt
def SIR_RK2(beta, gamma, N, I0, T):
    dt = 0.1
    t = np.linspace(0, T, int(T/dt) + 1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0
    I[0] = I0

    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + dt * k1_S/2) * (I[i-1] + dt * k1_I/2) / N
        k2_I = beta * (S[i-1] + dt * k1_S/2) * (I[i-1] + dt * k1_I/2) / N - gamma * (I[i-1] + dt * k1_I/2)

        S[i] = S[i-1] + dt * (k1_S + k2_S)/2
        I[i] = I[i-1] + dt * (k1_I + k2_I)/2
        R[i] = N - S[i] - I[i]

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time (days)')
    plt.ylabel('Number of Individuals')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.grid(True)
    plt.show()
