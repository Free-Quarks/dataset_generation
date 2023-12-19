import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(S0, I0, R0, beta, gamma, t_end, N, dt):
    t = np.arange(0, t_end, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1 = -beta * S[i-1] * I[i-1] / N
        l1 = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        m1 = gamma * I[i-1]
        k2 = -beta * (S[i-1] + dt * k1 / 2) * (I[i-1] + dt * l1 / 2) / N
        l2 = beta * (S[i-1] + dt * k1 / 2) * (I[i-1] + dt * l1 / 2) / N - gamma * (I[i-1] + dt * l1 / 2)
        m2 = gamma * (I[i-1] + dt * l1 / 2)
        k3 = -beta * (S[i-1] + dt * k2) * (I[i-1] + dt * l2) / N
        l3 = beta * (S[i-1] + dt * k2) * (I[i-1] + dt * l2) / N - gamma * (I[i-1] + dt * l2)
        m3 = gamma * (I[i-1] + dt * l2)

        S[i] = S[i-1] + dt * (k1 + 4*k2 + k3) / 6
        I[i] = I[i-1] + dt * (l1 + 4*l2 + l3) / 6
        R[i] = R[i-1] + dt * (m1 + 4*m2 + m3) / 6

    return S, I, R

S0 = 900
I0 = 100
R0 = 0
beta = 0.3
gamma = 0.05
t_end = 100
dt = 0.1
N = S0 + I0 + R0

S, I, R = SIR_RK3(S0, I0, R0, beta, gamma, t_end, N, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()
