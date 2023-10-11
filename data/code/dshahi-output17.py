import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt):
    t = np.arange(0, t_max, dt)
    N = S0 + I0 + R0
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        k1_s = -beta * S[i-1] * I[i-1] / N
        k1_i = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_s = -beta * (S[i-1] + dt * k1_s / 2) * (I[i-1] + dt * k1_i / 2) / N
        k2_i = beta * (S[i-1] + dt * k1_s / 2) * (I[i-1] + dt * k1_i / 2) / N - gamma * (I[i-1] + dt * k1_i / 2)
        S[i] = S[i-1] + dt * k2_s
        I[i] = I[i-1] + dt * k2_i
        R[i] = N - S[i] - I[i]
    return S, I, R


beta = 0.3
gamma = 0.1
S0 = 1000
I0 = 1
R0 = 0
t_max = 100
dt = 0.1

S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt)

plt.figure()
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()

