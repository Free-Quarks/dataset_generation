import numpy as np
import matplotlib.pyplot as plt

def SIR_RK3(beta, gamma, N, I0, T):
    dt = 0.1
    steps = int(T / dt)
    t = np.linspace(0, T, steps + 1)
    S = np.zeros(steps + 1)
    I = np.zeros(steps + 1)
    R = np.zeros(steps + 1)
    S[0] = N - I0
    I[0] = I0
    R[0] = 0
    
    for i in range(steps):
        S_prime = -beta * S[i] * I[i] / N
        I_prime = beta * S[i] * I[i] / N - gamma * I[i]
        R_prime = gamma * I[i]
        
        S_half = S[i] + dt / 2 * S_prime
        I_half = I[i] + dt / 2 * I_prime
        R_half = R[i] + dt / 2 * R_prime
        
        S_prime_half = -beta * S_half * I_half / N
        I_prime_half = beta * S_half * I_half / N - gamma * I_half
        R_prime_half = gamma * I_half
        
        S_new = S[i] + dt * S_prime_half
        I_new = I[i] + dt * I_prime_half
        R_new = R[i] + dt * R_prime_half
        
        S[i+1] = S_new
        I[i+1] = I_new
        R[i+1] = R_new
        
    return t, S, I, R

beta = 0.2
gamma = 0.1
N = 1000
I0 = 1
T = 100

# Run the SIR model using RK3
t, S, I, R = SIR_RK3(beta, gamma, N, I0, T)

# Plotting
plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.title('SIR Model using RK3')
plt.legend()
plt.show()
