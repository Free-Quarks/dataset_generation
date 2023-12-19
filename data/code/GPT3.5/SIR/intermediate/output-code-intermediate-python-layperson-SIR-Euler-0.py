import numpy as np
import matplotlib.pyplot as plt

def SIR_euler(S0, I0, R0, beta, gamma, t_max, dt):
    N = S0 + I0 + R0
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for t in range(t_max-1):
        dS = -beta * S[t] * I[t] / N
        dI = beta * S[t] * I[t] / N - gamma * I[t]
        dR = gamma * I[t]
        
        S[t+1] = S[t] + dt * dS
        I[t+1] = I[t] + dt * dI
        R[t+1] = R[t] + dt * dR
    
    return S, I, R

S0 = 990
I0 = 10
R0 = 0
beta = 0.2
gamma = 0.1
t_max = 100
dt = 0.1

S, I, R = SIR_euler(S0, I0, R0, beta, gamma, t_max, dt)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
