import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, N, I0, R0, t_max):
    S0 = N - I0 - R0
    dt = 0.1
    num_steps = int(t_max / dt)
    t = np.linspace(0, t_max, num_steps)
    S = np.zeros(num_steps)
    I = np.zeros(num_steps)
    R = np.zeros(num_steps)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    for i in range(1, num_steps):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    return S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
R0 = 0
t_max = 100

S, I, R = SIR_model(beta, gamma, N, I0, R0, t_max)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time steps')
plt.ylabel('Population')
plt.legend()
plt.show()
