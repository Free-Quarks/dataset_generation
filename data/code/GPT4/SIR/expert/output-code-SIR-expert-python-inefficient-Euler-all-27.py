import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S0, I0, R0, beta, gamma, num_iters):
    S, I, R = [S0], [I0], [R0]
    dt = 1.0
    for _ in range(num_iters):
        next_S = S[-1] - (beta*S[-1]*I[-1])*dt
        next_I = I[-1] + (beta*S[-1]*I[-1] - gamma*I[-1])*dt
        next_R = R[-1] + (gamma*I[-1])*dt
        S.append(next_S)
        I.append(next_I)
        R.append(next_R)

    return S, I, R

# Initial conditions
S0, I0, R0 = 999, 1, 0
# Infection rate and recovery rate
beta, gamma = 0.3, 0.1
# Time steps
num_iters = 1000

S, I, R = SIR_model(S0, I0, R0, beta, gamma, num_iters)

plt.plot(S, label='S')
plt.plot(I, label='I')
plt.plot(R, label='R')
plt.legend()
plt.show()
