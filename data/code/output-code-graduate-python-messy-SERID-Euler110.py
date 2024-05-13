import numpy as np
import matplotlib.pyplot as plt

def serid_euler(S0, E0, I0, R0, beta, sigma, gamma, N, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    S = np.zeros_like(t)
    E = np.zeros_like(t)
    I = np.zeros_like(t)
    R = np.zeros_like(t)
    
    # Set initial conditions
    S[0] = S0
    E[0] = E0
    I[0] = I0
    R[0] = R0
    
    # Euler's method
    for i in range(1, len(t)):
        dSdt = -beta * S[i-1] * I[i-1] / N
        dEdt = beta * S[i-1] * I[i-1] / N - sigma * E[i-1]
        dIdt = sigma * E[i-1] - gamma * I[i-1]
        dRdt = gamma * I[i-1]
        
        S[i] = S[i-1] + dt * dSdt
        E[i] = E[i-1] + dt * dEdt
        I[i] = I[i-1] + dt * dIdt
        R[i] = R[i-1] + dt * dRdt
    
    return t, S, E, I, R

# Example usage
S0 = 900
E0 = 100
I0 = 10
R0 = 0
beta = 0.2
sigma = 0.1
gamma = 0.05
N = 1000

t_max = 100
dt = 0.1

t, S, E, I, R = serid_euler(S0, E0, I0, R0, beta, sigma, gamma, N, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, E, label='Exposed')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.title('SERID Model Simulation')
plt.show()
