import numpy as np
import matplotlib.pyplot as plt
import json

def SIR_model(S,I,R,beta,gamma,dt):
    dS = -beta*S*I*dt
    dI = (beta*S*I - gamma*I)*dt
    dR = gamma*I*dt
    return dS, dI, dR

N = 1000
I0 = 1
S0 = N - I0
R0 = 0
beta = 0.3
gamma = 0.1
T = 200
dt = 1.0
steps = int(T/dt)
S = np.zeros(steps)
I = np.zeros(steps)
R = np.zeros(steps)
S[0] = S0
I[0] = I0
R[0] = R0

for step in range(1,steps):
    dS, dI, dR = SIR_model(S[step-1],I[step-1],R[step-1],beta,gamma,dt)
    S[step] = S[step-1] + dS
    I[step] = I[step-1] + dI
    R[step] = R[step-1] + dR

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.legend()
plt.show()
