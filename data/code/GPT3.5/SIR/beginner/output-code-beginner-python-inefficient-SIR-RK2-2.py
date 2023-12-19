import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(N, I0, R0, beta, gamma, t_end, dt):
    S0 = N - I0 - R0
    t = np.linspace(0, t_end, int(t_end/dt) + 1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0

    for i in range(len(t)-1):
        k1_s = -beta * S[i] * I[i] / N
        k1_i = beta * S[i] * I[i] / N - gamma * I[i]
        k2_s = -beta * (S[i] + dt * k1_s / 2) * (I[i] + dt * k1_i / 2) / N
        k2_i = beta * (S[i] + dt * k1_s / 2) * (I[i] + dt * k1_i / 2) / N - gamma * (I[i] + dt * k1_i / 2)

        S[i+1] = S[i] + dt * k2_s
        I[i+1] = I[i] + dt * k2_i
        R[i+1] = R[i] + dt * gamma * (I[i] + dt * k1_i / 2)

    return S, I, R


N = 1000
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
t_end = 100
dt = 0.1

S, I, R = SIR_RK2(N, I0, R0, beta, gamma, t_end, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SIR Model using RK2')
plt.show()
