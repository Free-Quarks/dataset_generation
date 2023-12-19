import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, S0, I0, R0, t_max, n_steps):
    dt = t_max / n_steps
    t = np.linspace(0, t_max, n_steps + 1)
    S = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    R = np.zeros(n_steps + 1)
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(n_steps):
        k1S = -beta * S[i] * I[i]
        k1I = beta * S[i] * I[i] - gamma * I[i]
        k1R = gamma * I[i]
        
        k2S = -beta * (S[i] + dt / 2 * k1S) * (I[i] + dt / 2 * k1I)
        k2I = beta * (S[i] + dt / 2 * k1S) * (I[i] + dt / 2 * k1I) - gamma * (I[i] + dt / 2 * k1I)
        k2R = gamma * (I[i] + dt / 2 * k1I)
        
        k3S = -beta * (S[i] - dt * k1S + 2 * dt * k2S) * (I[i] - dt * k1I + 2 * dt * k2I)
        k3I = beta * (S[i] - dt * k1S + 2 * dt * k2S) * (I[i] - dt * k1I + 2 * dt * k2I) - gamma * (I[i] - dt * k1I + 2 * dt * k2I)
        k3R = gamma * (I[i] - dt * k1I + 2 * dt * k2I)
        
        S[i+1] = S[i] + dt / 6 * (k1S + 4 * k2S + k3S)
        I[i+1] = I[i] + dt / 6 * (k1I + 4 * k2I + k3I)
        R[i+1] = R[i] + dt / 6 * (k1R + 4 * k2R + k3R)
        
    return S, I, R


beta = 0.3
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0.0
t_max = 100
n_steps = 1000

S, I, R = SIR_RK3(beta, gamma, S0, I0, R0, t_max, n_steps)

plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
