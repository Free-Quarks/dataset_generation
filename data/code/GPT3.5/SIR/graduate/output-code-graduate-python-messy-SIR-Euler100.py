import numpy as np
import matplotlib.pyplot as plt

def SIR_model(S0, I0, R0, beta, gamma, total_time, step_size):
    t = np.arange(0, total_time, step_size)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1]
        dI = beta * S[i-1] * I[i-1] - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + step_size * dS
        I[i] = I[i-1] + step_size * dI
        R[i] = R[i-1] + step_size * dR
    return S, I, R

S0 = 999
I0 = 1
R0 = 0
beta = 0.3
gamma = 0.1
total_time = 100
step_size = 0.1

S, I, R = SIR_model(S0, I0, R0, beta, gamma, total_time, step_size)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
