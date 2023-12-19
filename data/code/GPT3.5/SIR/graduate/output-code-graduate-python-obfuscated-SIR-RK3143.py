import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, T):
    dt = 0.1
    num_steps = int(T / dt)
    t = np.arange(0, T, dt)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0

    for i in range(num_steps - 1):
        k1_S = -beta * S[i] * I[i] / N
        k1_I = beta * S[i] * I[i] / N - gamma * I[i]
        k1_R = gamma * I[i]
        k2_S = -beta * (S[i] + 0.5 * dt * k1_S) * (I[i] + 0.5 * dt * k1_I) / N
        k2_I = beta * (S[i] + 0.5 * dt * k1_S) * (I[i] + 0.5 * dt * k1_I) / N - gamma * (I[i] + 0.5 * dt * k1_I)
        k2_R = gamma * (I[i] + 0.5 * dt * k1_I)
        k3_S = -beta * (S[i] - dt * k1_S + 2 * dt * k2_S) * (I[i] - dt * k1_I + 2 * dt * k2_I) / N
        k3_I = beta * (S[i] - dt * k1_S + 2 * dt * k2_S) * (I[i] - dt * k1_I + 2 * dt * k2_I) / N - gamma * (I[i] - dt * k1_I + 2 * dt * k2_I)
        k3_R = gamma * (I[i] - dt * k1_I + 2 * dt * k2_I)
        S[i+1] = S[i] + dt * (1/6 * k1_S + 2/3 * k2_S + 1/6 * k3_S)
        I[i+1] = I[i] + dt * (1/6 * k1_I + 2/3 * k2_I + 1/6 * k3_I)
        R[i+1] = R[i] + dt * (1/6 * k1_R + 2/3 * k2_R + 1/6 * k3_R)

    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK3')
    plt.legend()
    plt.show()
