import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, tmax, dt):
    N = S0 + I0 + R0
    t = np.arange(0, tmax, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + 0.5 * dt * k1_S) * (I[i-1] + 0.5 * dt * k1_I) / N
        k2_I = beta * (S[i-1] + 0.5 * dt * k1_S) * (I[i-1] + 0.5 * dt * k1_I) / N - gamma * (I[i-1] + 0.5 * dt * k1_I)
        S[i] = S[i-1] + dt * k2_S
        I[i] = I[i-1] + dt * k2_I
        R[i] = R[i-1] + dt * gamma * (I[i-1] + 0.5 * dt * k1_I)
    plt.figure()
    plt.plot(t, S, label='Susceptible')
    plt.plot(t, I, label='Infected')
    plt.plot(t, R, label='Recovered')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.title('SIR Model using RK2')
    plt.legend()
    plt.show()


# Example usage
beta = 0.3
gamma = 0.1
S0 = 990
I0 = 10
R0 = 0
tmax = 100
dt = 0.1
SIR_RK2(beta, gamma, S0, I0, R0, tmax, dt)
