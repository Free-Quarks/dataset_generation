import numpy as np
import matplotlib.pyplot as plt


def SIR_RK3(beta, gamma, N, I0, R0, t_max, dt):
    t = np.arange(0, t_max, dt)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0

    for i in range(1, len(t)):
        k1_s = -beta * S[i-1] * I[i-1] / N
        k1_i = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k1_r = gamma * I[i-1]

        s_temp = S[i-1] + k1_s * dt
        i_temp = I[i-1] + k1_i * dt
        r_temp = R[i-1] + k1_r * dt

        k2_s = -beta * s_temp * i_temp / N
        k2_i = beta * s_temp * i_temp / N - gamma * i_temp
        k2_r = gamma * i_temp

        s_temp = S[i-1] + 0.75 * k1_s * dt + 0.25 * k2_s * dt
        i_temp = I[i-1] + 0.75 * k1_i * dt + 0.25 * k2_i * dt
        r_temp = R[i-1] + 0.75 * k1_r * dt + 0.25 * k2_r * dt

        k3_s = -beta * s_temp * i_temp / N
        k3_i = beta * s_temp * i_temp / N - gamma * i_temp
        k3_r = gamma * i_temp

        S[i] = S[i-1] + (2/9 * k1_s + 1/3 * k2_s + 4/9 * k3_s) * dt
        I[i] = I[i-1] + (2/9 * k1_i + 1/3 * k2_i + 4/9 * k3_i) * dt
        R[i] = R[i-1] + (2/9 * k1_r + 1/3 * k2_r + 4/9 * k3_r) * dt

    return S, I, R


beta = 0.3
gamma = 0.1
N = 1000
I0 = 1
R0 = 0

S, I, R = SIR_RK3(beta, gamma, N, I0, R0, t_max=100, dt=0.1)

plt.plot(S, label='S')
plt.plot(I, label='I')
plt.plot(R, label='R')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
