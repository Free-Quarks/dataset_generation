import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, T):
    dt = 0.1
    t = np.linspace(0, T, int(T/dt)+1)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(len(t)-1):
        k1 = -beta * S[i] * I[i] / N
        k2 = -beta * (S[i] + dt/2 * k1) * (I[i] + dt/2 * k1) / N
        k3 = -beta * (S[i] - dt * k1 + 2 * dt * k2) * (I[i] - dt * k1 + 2 * dt * k2) / N
        
        l1 = beta * S[i] * I[i] / N - gamma * I[i]
        l2 = beta * (S[i] + dt/2 * k1) * (I[i] + dt/2 * k1) / N - gamma * (I[i] + dt/2 * l1)
        l3 = beta * (S[i] - dt * k1 + 2 * dt * k2) * (I[i] - dt * k1 + 2 * dt * l2) / N - gamma * (I[i] - dt * k1 + 2 * dt * l2)
        
        m1 = gamma * I[i]
        m2 = gamma * (I[i] + dt/2 * l1)
        m3 = gamma * (I[i] - dt * l1 + 2 * dt * l2)
        
        S[i+1] = S[i] + dt/6 * (k1 + 4*k2 + k3)
        I[i+1] = I[i] + dt/6 * (l1 + 4*l2 + l3)
        R[i+1] = R[i] + dt/6 * (m1 + 4*m2 + m3)
    
    return S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 100

S, I, R = SIR_RK3(beta, gamma, N, I0, T)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Number of People')
plt.legend()
plt.title('SIR Model Using RK3')
plt.show()
