import numpy as np
import matplotlib.pyplot as plt

def sir_model(beta, gamma, S0, I0, R0, t_end, delta_t):
    N = S0 + I0 + R0
    t = np.arange(0, t_end+delta_t, delta_t)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        S[i] = S[i-1] - beta/N * S[i-1] * I[i-1] * delta_t
        I[i] = I[i-1] + (beta/N * S[i-1] * I[i-1] - gamma * I[i-1]) * delta_t
        R[i] = R[i-1] + gamma * I[i-1] * delta_t
    return S, I, R

beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0
t_end = 100
delta_t = 0.1

S, I, R = sir_model(beta, gamma, S0, I0, R0, t_end, delta_t)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()

