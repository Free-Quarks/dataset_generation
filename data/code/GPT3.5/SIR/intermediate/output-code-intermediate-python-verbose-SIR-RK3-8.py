import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(S0, I0, R0, beta, gamma, t_max):
    N = S0 + I0 + R0
    h = 0.01
    t = np.arange(0, t_max+h, h)
    S = np.zeros(t.shape)
    I = np.zeros(t.shape)
    R = np.zeros(t.shape)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, t.shape[0]):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = (beta * S[i-1] * I[i-1] / N) - gamma * I[i-1]
        k1_R = gamma * I[i-1]
        k2_S = -beta * (S[i-1] + (0.5 * h * k1_S)) * (I[i-1] + (0.5 * h * k1_I)) / N
        k2_I = (beta * (S[i-1] + (0.5 * h * k1_S)) * (I[i-1] + (0.5 * h * k1_I)) / N) - gamma * (I[i-1] + (0.5 * h * k1_I))
        k2_R = gamma * (I[i-1] + (0.5 * h * k1_I))
        k3_S = -beta * (S[i-1] - (h * k1_S) + (2 * h * k2_S)) * (I[i-1] - (h * k1_I) + (2 * h * k2_I)) / N
        k3_I = (beta * (S[i-1] - (h * k1_S) + (2 * h * k2_S)) * (I[i-1] - (h * k1_I) + (2 * h * k2_I)) / N) - gamma * (I[i-1] - (h * k1_I) + (2 * h * k2_I))
        k3_R = gamma * (I[i-1] - (h * k1_I) + (2 * h * k2_I))
        S[i] = S[i-1] + (h / 6) * (k1_S + 4*k2_S + k3_S)
        I[i] = I[i-1] + (h / 6) * (k1_I + 4*k2_I + k3_I)
        R[i] = R[i-1] + (h / 6) * (k1_R + 4*k2_R + k3_R)
    return S, I, R

S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100

S, I, R = SIR_RK3(S0, I0, R0, beta, gamma, t_max)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
