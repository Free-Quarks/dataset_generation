import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, T):
    dt = 0.1
    t = np.arange(0, T, dt)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    for i in range(1, len(t)):
        k1_s = -beta * S[i-1] * I[i-1] / N
        k1_i = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k1_r = gamma * I[i-1]
        k2_s = -beta * (S[i-1] + dt/2 * k1_s) * (I[i-1] + dt/2 * k1_i) / N
        k2_i = beta * (S[i-1] + dt/2 * k1_s) * (I[i-1] + dt/2 * k1_i) / N - gamma * (I[i-1] + dt/2 * k1_i)
        k2_r = gamma * (I[i-1] + dt/2 * k1_i)
        k3_s = -beta * (S[i-1] - dt * k1_s + 2 * dt * k2_s) * (I[i-1] - dt * k1_i + 2 * dt * k2_i) / N
        k3_i = beta * (S[i-1] - dt * k1_s + 2 * dt * k2_s) * (I[i-1] - dt * k1_i + 2 * dt * k2_i) / N - gamma * (I[i-1] - dt * k1_i + 2 * dt * k2_i)
        k3_r = gamma * (I[i-1] - dt * k1_i + 2 * dt * k2_i)
        S[i] = S[i-1] + dt/6 * (k1_s + 4 * k2_s + k3_s)
        I[i] = I[i-1] + dt/6 * (k1_i + 4 * k2_i + k3_i)
        R[i] = R[i-1] + dt/6 * (k1_r + 4 * k2_r + k3_r)
    return S, I, R

beta = 0.3
gamma = 0.1
N = 1000
I0 = 10
T = 100

S, I, R = SIR_RK3(beta, gamma, N, I0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
