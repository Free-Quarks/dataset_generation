import numpy as np
import matplotlib.pyplot as plt

def SIR_model(beta, gamma, I0, N, t_end):
    dt = 1
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(1, len(t)):
        dS = -beta * S[i-1] * I[i-1] / N
        dI = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        dR = gamma * I[i-1]
        S[i] = S[i-1] + dt * dS
        I[i] = I[i-1] + dt * dI
        R[i] = R[i-1] + dt * dR
    
    return S, I, R

beta = 0.3
gamma = 0.1
I0 = 100
N = 1000
t_end = 100

S, I, R = SIR_model(beta, gamma, I0, N, t_end)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()
