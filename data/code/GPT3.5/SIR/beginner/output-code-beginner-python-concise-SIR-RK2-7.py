import numpy as np
import matplotlib.pyplot as plt

def SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt):
    # Initialize arrays
    t = np.arange(0, t_max, dt)
    N_steps = len(t)
    S = np.zeros(N_steps)
    I = np.zeros(N_steps)
    R = np.zeros(N_steps)
    
    # Set initial conditions
    S[0] = S0
    I[0] = I0
    R[0] = R0
    
    # Perform RK2 integration
    for i in range(1, N_steps):
        S_star = S[i-1] - beta * S[i-1] * I[i-1] * dt
        I_star = I[i-1] + beta * S[i-1] * I[i-1] * dt - gamma * I[i-1] * dt
        
        S[i] = S[i-1] - 0.5 * beta * (S[i-1] * I[i-1] + S_star * I_star) * dt
        I[i] = I[i-1] + 0.5 * beta * (S[i-1] * I[i-1] + S_star * I_star) * dt - 0.5 * gamma * (I[i-1] + I_star) * dt
        R[i] = R[i-1] + gamma * (I[i-1] + I_star) * dt
    
    # Return arrays
    return t, S, I, R

# Example usage
beta = 0.2
gamma = 0.1
S0 = 0.99
I0 = 0.01
R0 = 0
t_max = 100
dt = 0.1

t, S, I, R = SIR_RK2(beta, gamma, S0, I0, R0, t_max, dt)

plt.plot(t, S, label='Susceptible')
plt.plot(t, I, label='Infected')
plt.plot(t, R, label='Recovered')
plt.xlabel('Time')
plt.ylabel('Proportion')
plt.title('SIR Model using RK2')
plt.legend()
plt.show()
