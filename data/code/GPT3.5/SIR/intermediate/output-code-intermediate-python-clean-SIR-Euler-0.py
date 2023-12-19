import numpy as np
import matplotlib.pyplot as plt


def SIR_model(N, beta, gamma, I0, t_max):
    
    S0 = N - I0
    R0 = 0
    
    S = np.zeros(t_max)
    I = np.zeros(t_max)
    R = np.zeros(t_max)
    
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    dt = 0.1
    
    for t in range(1, t_max):
        dS = -beta * S[t-1] * I[t-1] / N
        dI = beta * S[t-1] * I[t-1] / N - gamma * I[t-1]
        dR = gamma * I[t-1]
        
        S[t] = S[t-1] + dt * dS
        I[t] = I[t-1] + dt * dI
        R[t] = R[t-1] + dt * dR
    
    return S, I, R


N = 1000
beta = 0.4
gamma = 0.1
I0 = 1

S, I, R = SIR_model(N, beta, gamma, I0, t_max=100)

plt.plot(range(100), S, label='Susceptible')
plt.plot(range(100), I, label='Infected')
plt.plot(range(100), R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
