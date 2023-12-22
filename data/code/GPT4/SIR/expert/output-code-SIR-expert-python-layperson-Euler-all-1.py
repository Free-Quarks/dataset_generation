import numpy as np
import matplotlib.pyplot as plt
import json

def sir_model_euler(S0, I0, R0, beta, gamma, dt, num_steps):
    S = np.zeros(num_steps+1)
    I = np.zeros(num_steps+1)
    R = np.zeros(num_steps+1)

    S[0] = S0
    I[0] = I0
    R[0] = R0

    for step in range(num_steps):
        S[step+1] = S[step] - dt*beta*S[step]*I[step]
        I[step+1] = I[step] + dt*(beta*S[step]*I[step] - gamma*I[step])
        R[step+1] = R[step] + dt*gamma*I[step]

    return S, I, R

S0 = 999
I0 = 1
R0 = 0
beta = 0.2
gamma = 0.1
dt = 0.05
num_steps = 1000

S, I, R = sir_model_euler(S0, I0, R0, beta, gamma, dt, num_steps)

plt.figure(figsize=[6,4])
plt.plot(S, label='S')
plt.plot(I, label='I')
plt.plot(R, label='R')
plt.legend()
plt.grid()
plt.show()
