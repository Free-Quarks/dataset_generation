import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, R0, t_end, dt):
    t = np.arange(0, t_end, dt)
    n = len(t)
    S = np.zeros(n)
    I = np.zeros(n)
    R = np.zeros(n)
    
    S[0] = N - I0 - R0
    I[0] = I0
    R[0] = R0
    
    for i in range(n-1):
        k1 = -beta * S[i] * I[i] / N
        k2 = -beta * (S[i] + 0.5 * dt * k1) * (I[i] + 0.5 * dt * k1) / N
        k3 = -beta * (S[i] + 0.5 * dt * k2) * (I[i] + 0.5 * dt * k2) / N
        
        S[i+1] = S[i] + dt * (k1 + k2 + k3) / 6
        I[i+1] = I[i] + dt * (k1 + k2 + k3) / 6
        R[i+1] = R[i] + gamma * I[i+1] * dt
    
    return S, I, R

# Example usage
beta = 0.3
gamma = 0.1
N = 1000
I0 = 10
R0 = 0
t_end = 100
dt = 0.1

S, I, R = SIR_RK3(beta, gamma, N, I0, R0, t_end, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
