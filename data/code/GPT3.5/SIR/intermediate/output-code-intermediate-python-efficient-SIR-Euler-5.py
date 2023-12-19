import numpy as np
import matplotlib.pyplot as plt

def SIR_Euler(beta, gamma, N, I0, t_max, dt):
    t = np.linspace(0, t_max, int(t_max/dt)+1)
    S = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(len(t)-1):
        S[i+1] = S[i] - dt * beta * S[i] * I[i] / N
        I[i+1] = I[i] + dt * (beta * S[i] * I[i] / N - gamma * I[i])
        R[i+1] = R[i] + dt * gamma * I[i]
    
    return S, I, R

# Example usage:
beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
t_max = 100
dt = 0.1
S, I, R = SIR_Euler(beta, gamma, N, I0, t_max, dt)
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Number of individuals')
plt.title('SIR Model Simulation')
plt.legend()
plt.show()

