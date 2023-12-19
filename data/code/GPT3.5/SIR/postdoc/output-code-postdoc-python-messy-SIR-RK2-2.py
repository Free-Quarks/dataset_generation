import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    t = np.arange(0, t_max, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + dt * k1_S / 2) * (I[i-1] + dt * k1_I / 2) / N
        k2_I = beta * (S[i-1] + dt * k1_S / 2) * (I[i-1] + dt * k1_I / 2) / N - gamma * (I[i-1] + dt * k1_I / 2)
        S[i] = S[i-1] + dt * (k1_S + k2_S) / 2
        I[i] = I[i-1] + dt * (k1_I + k2_I) / 2
        R[i] = R[i-1] + dt * gamma * (I[i-1] + dt * k1_I / 2)
    return S, I, R

S0 = 990
I0 = 10
R0 = 0
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_RK2(S0, I0, R0, beta, gamma, t_max, dt)

plt.figure()
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

