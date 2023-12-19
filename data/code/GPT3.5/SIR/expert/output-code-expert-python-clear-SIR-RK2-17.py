import numpy as np
import matplotlib.pyplot as plt


def SIR_RK2(beta, gamma, S0, I0, R0, N, t_end, dt):
    t = np.arange(0, t_end, dt)
    S = np.zeros(len(t))
    I = np.zeros(len(t))
    R = np.zeros(len(t))
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    for i in range(1, len(t)):
        k1_S = -beta * S[i-1] * I[i-1] / N
        k1_I = beta * S[i-1] * I[i-1] / N - gamma * I[i-1]
        k2_S = -beta * (S[i-1] + dt * k1_S) * (I[i-1] + dt * k1_I) / N
        k2_I = beta * (S[i-1] + dt * k1_S) * (I[i-1] + dt * k1_I) / N - gamma * (I[i-1] + dt * k1_I)
        
        S[i] = S[i-1] + dt * 0.5 * (k1_S + k2_S)
        I[i] = I[i-1] + dt * 0.5 * (k1_I + k2_I)
        R[i] = R[i-1] + dt * gamma * (I[i-1] + dt * k1_I)
    
    return S, I, R


beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0
N = 1

# Simulation parameters
t_end = 100
dt = 0.1

# Run simulation
S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, N, t_end, dt)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(S, label='Susceptible')
plt.plot(I, label='Infected')
plt.plot(R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Fraction of Population')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
