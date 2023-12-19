import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(N, I0, beta, gamma, t_max, dt):
    t = np.arange(0, t_max, dt)
    num_steps = len(t)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    for i in range(1, num_steps):
        k1_s = -beta * S[i-1] * I[i-1] / N
        k1_i = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_s = -beta * (S[i-1] + dt * k1_s/2) * (I[i-1] + dt * k1_i/2) / N
        k2_i = beta * (S[i-1] + dt * k1_s/2) * (I[i-1] + dt * k1_i/2) / N - gamma * (I[i-1] + dt * k1_i/2)
        S[i] = S[i-1] + dt * k2_s
        I[i] = I[i-1] + dt * k2_i
        R[i] = R[i-1] + gamma * (I[i-1] + dt * k1_i/2)
    return S, I, R

N = 1000
I0 = 1
beta = 0.3
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_RK2(N, I0, beta, gamma, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
